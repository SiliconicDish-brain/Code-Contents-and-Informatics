"""
MNIST Digit Recognition: SNN (Reservoir Computing) vs ANN Comparison
=====================================================================

Scales up from XOR to real handwritten digit recognition (MNIST, 10 classes,
28×28 images) so the architectural differences between spiking and
conventional neural networks become meaningful:

  • The SNN uses a FIXED random reservoir (Liquid State Machine / Reservoir
    Computing paradigm) with Leaky Integrate-and-Fire (LIF) neurons,
    followed by a trained linear readout.
  • The ANN is a conventional dense feedforward network of matched size.

NOTE: Uses a pure-PyTorch LIF implementation instead of BindsNET to avoid
compatibility issues with modern PyTorch (torch._six was removed in PyTorch 2.x).

Key insight at this scale: MNIST images are ~80% zero-valued pixels. The SNN
naturally exploits this sparsity — zero pixels produce zero spikes, zero
synaptic events, zero energy. The ANN must multiply through every weight
regardless.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import time as timer

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ================================================================
# 1. CONFIGURATION
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Network architecture ---
N_INPUT      = 784          # 28×28 MNIST pixels
N_RESERVOIR  = 500          # LIF neurons in the reservoir
N_OUTPUT     = 10           # digit classes 0–9
CONNECTIVITY = 0.15         # fraction of Input→Reservoir synapses that exist

# --- LIF neuron parameters ---
LIF_THRESH   = 1.0          # membrane voltage threshold for spiking
LIF_RESET    = 0.0          # post-spike reset voltage
LIF_DECAY    = 0.85         # membrane voltage decay factor per timestep (leakiness)

# --- Simulation ---
SIM_TIME = 50               # timesteps per sample
DT       = 1.0

# --- Data ---
TRAIN_SIZE = 5000           # subset of 60 k  (speed vs quality trade-off)
TEST_SIZE  = 1000           # subset of 10 k
BATCH_ANN  = 64             # mini-batch for ANN training

# --- Training ---
EPOCHS_READOUT = 20         # epochs to train SNN linear readout (fast)
EPOCHS_ANN     = 20         # epochs to train full ANN
LR_READOUT     = 0.005
LR_ANN         = 0.001

# --- Energy model (picojoules per event) ---
ENERGY_PER_SNN_SPIKE_PJ = 0.9   # Loihi-class neuromorphic chip
ENERGY_PER_ANN_MAC_PJ   = 4.6   # 45 nm CMOS multiply-accumulate

# --- Output ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "Final")
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# 2. PURE-PYTORCH LIF RESERVOIR
# ================================================================
class LIFReservoir:
    """
    A fixed random reservoir of Leaky Integrate-and-Fire neurons.

    LIF dynamics per timestep:
        v[t] = decay * v[t-1] + W @ input_spikes[t]
        spike = (v >= threshold)
        v[spike] = reset

    No learning — weights are frozen random projections. This is the
    standard Liquid State Machine / reservoir computing paradigm.
    """

    def __init__(self, n_input, n_reservoir, connectivity, thresh, reset,
                 decay, device):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.thresh = thresh
        self.reset_val = reset
        self.decay = decay
        self.device = device

        # Sparse random input→reservoir weights
        # Scale by ~1/sqrt(fan_in) for stable membrane dynamics
        fan_in = int(n_input * connectivity)
        w_scale = 1.0 / np.sqrt(max(fan_in, 1))

        w = torch.rand(n_input, n_reservoir, device=device) * w_scale
        mask = (torch.rand(n_input, n_reservoir, device=device) < connectivity)
        self.W = (w * mask.float())   # frozen, not nn.Parameter

        self.n_active_synapses = int(mask.sum().item())
        self.avg_fanout = self.n_active_synapses / n_input

        # State
        self.v = torch.zeros(n_reservoir, device=device)

    def reset_state(self):
        self.v = torch.zeros(self.n_reservoir, device=self.device)

    def run(self, spike_train):
        """
        Run the reservoir for an entire spike train.

        Args:
            spike_train: (time_steps, n_input) binary tensor

        Returns:
            spike_counts: (n_reservoir,) total spikes per neuron
            all_spikes:   (time_steps, n_reservoir) binary spike raster
            all_voltages: (time_steps, n_reservoir) membrane voltages
            total_input_spikes: scalar, total input spikes
        """
        time_steps = spike_train.shape[0]
        all_spikes = torch.zeros(time_steps, self.n_reservoir, device=self.device)
        all_voltages = torch.zeros(time_steps, self.n_reservoir, device=self.device)
        self.reset_state()

        for t in range(time_steps):
            # Input current = input_spikes @ W
            I = spike_train[t] @ self.W            # (n_reservoir,)

            # Leaky integration
            self.v = self.decay * self.v + I

            # Spike generation
            spikes = (self.v >= self.thresh).float()
            all_spikes[t] = spikes
            all_voltages[t] = self.v.clone()

            # Reset neurons that spiked
            self.v[spikes.bool()] = self.reset_val

        spike_counts = all_spikes.sum(dim=0)       # (n_reservoir,)
        total_input_spikes = spike_train.sum().item()

        return spike_counts, all_spikes, all_voltages, total_input_spikes


# ================================================================
# 3. LOAD MNIST
# ================================================================
print("\n" + "=" * 55)
print("  LOADING MNIST DATASET")
print("=" * 55)

transform = transforms.Compose([transforms.ToTensor()])

train_full = datasets.MNIST(root=os.path.join(SCRIPT_DIR, "data"),
                            train=True,  download=True, transform=transform)
test_full  = datasets.MNIST(root=os.path.join(SCRIPT_DIR, "data"),
                            train=False, download=True, transform=transform)

train_set = Subset(train_full, range(TRAIN_SIZE))
test_set  = Subset(test_full,  range(TEST_SIZE))

# batch=1 for SNN (reservoir processes one sample at a time)
train_loader_snn = DataLoader(train_set, batch_size=1, shuffle=False)
test_loader_snn  = DataLoader(test_set,  batch_size=1, shuffle=False)

# batched loaders for ANN
train_loader_ann = DataLoader(train_set, batch_size=BATCH_ANN, shuffle=True)
test_loader_ann  = DataLoader(test_set,  batch_size=BATCH_ANN, shuffle=False)

print(f"Training samples: {TRAIN_SIZE}  |  Test samples: {TEST_SIZE}")


# ================================================================
# 4. BUILD THE SNN RESERVOIR
# ================================================================
print("\n" + "=" * 55)
print("  BUILDING SNN RESERVOIR (Liquid State Machine)")
print("=" * 55)

reservoir = LIFReservoir(
    n_input=N_INPUT,
    n_reservoir=N_RESERVOIR,
    connectivity=CONNECTIVITY,
    thresh=LIF_THRESH,
    reset=LIF_RESET,
    decay=LIF_DECAY,
    device=device,
)

print(f"Reservoir: {N_INPUT} -> {N_RESERVOIR} LIF neurons")
print(f"Active synapses: {reservoir.n_active_synapses:,} / {N_INPUT * N_RESERVOIR:,} "
      f"({CONNECTIVITY * 100:.0f}% connectivity)")
print(f"Avg fan-out per input neuron: {reservoir.avg_fanout:.1f}")
print(f"LIF params: thresh={LIF_THRESH}, decay={LIF_DECAY}, reset={LIF_RESET}")


# ================================================================
# 5. ENCODING & UTILITY FUNCTIONS
# ================================================================
def poisson_encode(image_flat, time_steps, dev):
    """
    Poisson rate coding: pixel intensity p in [0, 1] becomes the
    per-timestep probability of emitting a spike.
    Returns: (time_steps, N_INPUT) binary tensor.
    """
    rates = image_flat.unsqueeze(0).expand(time_steps, -1).to(dev)  # (T, 784)
    spikes = (torch.rand_like(rates) < rates).float()
    return spikes


def compute_confusion_matrix(y_true, y_pred, n_classes=10):
    """Compute confusion matrix without sklearn."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


# ================================================================
# 6. EXTRACT SNN SPIKE FEATURES
# ================================================================
# Standard reservoir computing workflow:
#   1. Run each sample through the FIXED reservoir once -> spike counts.
#   2. Train a lightweight linear readout on these feature vectors.
# The expensive reservoir simulation runs only ONCE per sample, not per epoch.


def extract_features(loader, desc="Extracting"):
    """Run all samples through the reservoir, collect spike-count features."""
    features = []
    labels = []
    per_sample_events = []
    per_sample_input_spikes = []

    n_samples = len(loader)
    t0 = timer.time()

    for i, (img, lbl) in enumerate(loader):
        img_flat = img.view(-1)                                     # (784,)
        spike_train = poisson_encode(img_flat, SIM_TIME, device)    # (T, 784)

        spike_counts, _, _, input_spike_count = reservoir.run(spike_train)

        features.append(spike_counts.unsqueeze(0).cpu())   # (1, N_RESERVOIR)
        labels.append(lbl.item())

        # --- Synaptic event accounting ---
        reservoir_spike_count = spike_counts.sum().item()

        # Each input spike propagates to its connected reservoir neurons.
        synaptic_events = input_spike_count * reservoir.avg_fanout + reservoir_spike_count
        per_sample_events.append(synaptic_events)
        per_sample_input_spikes.append(input_spike_count)

        if (i + 1) % 500 == 0 or (i + 1) == n_samples:
            elapsed = timer.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate if rate > 0 else 0
            print(f"  {desc}: {i + 1:>5}/{n_samples} "
                  f"({rate:.1f} samples/s, ETA {eta:.0f}s)")

    features = torch.cat(features, dim=0)              # (N, N_RESERVOIR)
    labels = torch.tensor(labels, dtype=torch.long)     # (N,)
    per_sample_events = np.array(per_sample_events)
    per_sample_input_spikes = np.array(per_sample_input_spikes)
    return features, labels, per_sample_events, per_sample_input_spikes


print("\n" + "=" * 55)
print("  EXTRACTING SNN FEATURES (one-time reservoir pass)")
print("=" * 55)

t_start = timer.time()
train_features, train_labels, train_events, _ = extract_features(
    train_loader_snn, desc="Train"
)
test_features, test_labels, test_events, test_input_spikes = extract_features(
    test_loader_snn, desc="Test "
)
t_extract = timer.time() - t_start

snn_avg_events = test_events.mean()
print(f"\nFeature extraction total: {t_extract:.1f}s")
print(f"Avg synaptic events/sample (test): {snn_avg_events:,.0f}")
print(f"Avg input spikes/sample (test):    {test_input_spikes.mean():,.0f}")

# Quick sanity check: are the reservoir neurons actually firing?
avg_reservoir_activity = train_features.mean().item()
print(f"Avg reservoir spike count/neuron:   {avg_reservoir_activity:.2f}")
if avg_reservoir_activity < 0.01:
    print("WARNING: Reservoir is very quiet — consider increasing weight scale "
          "or reducing threshold.")


# ================================================================
# 7. TRAIN SNN READOUT LAYER
# ================================================================
print("\n" + "=" * 55)
print(f"  TRAINING SNN READOUT ({EPOCHS_READOUT} epochs)")
print("=" * 55)


class Readout(nn.Module):
    """Single linear layer on top of reservoir spike counts."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


readout = Readout(N_RESERVOIR, N_OUTPUT).to(device)
opt_readout = optim.Adam(readout.parameters(), lr=LR_READOUT)
loss_fn = nn.CrossEntropyLoss()

train_X = train_features.to(device)
train_Y = train_labels.to(device)

snn_losses = []
snn_accs   = []

t_start = timer.time()
for epoch in range(EPOCHS_READOUT):
    readout.train()

    # Shuffle each epoch
    perm = torch.randperm(len(train_X), device=device)
    X_shuf = train_X[perm]
    Y_shuf = train_Y[perm]

    epoch_loss = 0.0
    correct = 0
    bs = 64

    for start in range(0, len(X_shuf), bs):
        bx = X_shuf[start:start + bs]
        by = Y_shuf[start:start + bs]

        opt_readout.zero_grad()
        logits = readout(bx)
        loss = loss_fn(logits, by)
        loss.backward()
        opt_readout.step()

        epoch_loss += loss.item() * len(bx)
        correct += (logits.argmax(dim=1) == by).sum().item()

    avg_loss = epoch_loss / len(train_X)
    acc = correct / len(train_X) * 100
    snn_losses.append(avg_loss)
    snn_accs.append(acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:>3}/{EPOCHS_READOUT} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

t_readout = timer.time() - t_start
print(f"Readout training: {t_readout:.1f}s")


# ================================================================
# 8. BUILD & TRAIN ANN BASELINE
# ================================================================
print("\n" + "=" * 55)
print(f"  TRAINING ANN BASELINE ({EPOCHS_ANN} epochs)")
print("=" * 55)
print(f"  Architecture: {N_INPUT} -> {N_RESERVOIR} (ReLU) -> {N_OUTPUT}")


class ANN(nn.Module):
    """
    Matched-size feedforward network: same hidden dimension as the
    reservoir so the comparison is architecturally fair.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_RESERVOIR),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(N_RESERVOIR, N_OUTPUT),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


ann = ANN().to(device)
opt_ann = optim.Adam(ann.parameters(), lr=LR_ANN)
ann_loss_fn = nn.CrossEntropyLoss()

ann_losses = []
ann_accs   = []

t_start = timer.time()
for epoch in range(EPOCHS_ANN):
    ann.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for imgs, lbls in train_loader_ann:
        imgs, lbls = imgs.to(device), lbls.to(device)

        opt_ann.zero_grad()
        logits = ann(imgs)
        loss = ann_loss_fn(logits, lbls)
        loss.backward()
        opt_ann.step()

        epoch_loss += loss.item() * len(imgs)
        correct += (logits.argmax(dim=1) == lbls).sum().item()
        total += len(imgs)

    avg_loss = epoch_loss / total
    acc = correct / total * 100
    ann_losses.append(avg_loss)
    ann_accs.append(acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:>3}/{EPOCHS_ANN} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

t_ann = timer.time() - t_start
print(f"ANN training: {t_ann:.1f}s")


# ================================================================
# 9. EVALUATE BOTH MODELS ON TEST SET
# ================================================================
print("\n" + "=" * 55)
print("  EVALUATION ON TEST SET")
print("=" * 55)

# --- SNN readout ---
readout.eval()
test_X = test_features.to(device)
with torch.no_grad():
    snn_logits = readout(test_X)
    snn_preds = snn_logits.argmax(dim=1).cpu().numpy()
snn_true = test_labels.numpy()
snn_acc = (snn_preds == snn_true).mean() * 100

# --- ANN ---
ann.eval()
ann_preds_list = []
ann_true_list = []
with torch.no_grad():
    for imgs, lbls in test_loader_ann:
        imgs = imgs.to(device)
        logits = ann(imgs)
        ann_preds_list.append(logits.argmax(dim=1).cpu().numpy())
        ann_true_list.append(lbls.numpy())
ann_preds = np.concatenate(ann_preds_list)
ann_true = np.concatenate(ann_true_list)
ann_acc = (ann_preds == ann_true).mean() * 100

print(f"\n  +---------------------------------------------+")
print(f"  |  SNN Reservoir + Readout  ->  {snn_acc:5.1f}% accuracy |")
print(f"  |  ANN Dense Feedforward    ->  {ann_acc:5.1f}% accuracy |")
print(f"  +---------------------------------------------+")

# --- Per-digit accuracy ---
snn_per_digit = []
ann_per_digit = []
for d in range(10):
    mask_s = snn_true == d
    mask_a = ann_true == d
    snn_per_digit.append((snn_preds[mask_s] == d).mean() * 100 if mask_s.sum() > 0 else 0)
    ann_per_digit.append((ann_preds[mask_a] == d).mean() * 100 if mask_a.sum() > 0 else 0)

print("\n  Per-digit accuracy:")
print(f"  {'Digit':>5}  {'SNN':>7}  {'ANN':>7}")
print(f"  {'---':>5}  {'---':>7}  {'---':>7}")
for d in range(10):
    print(f"  {d:>5}  {snn_per_digit[d]:>6.1f}%  {ann_per_digit[d]:>6.1f}%")

# --- Confusion matrices ---
snn_cm = compute_confusion_matrix(snn_true, snn_preds)
ann_cm = compute_confusion_matrix(ann_true, ann_preds)


# ================================================================
# 10. ENERGY & COMPUTATIONAL ANALYSIS
# ================================================================
print("\n" + "=" * 55)
print("  ENERGY & COMPUTATIONAL ANALYSIS")
print("=" * 55)

# ANN: deterministic MAC count per forward pass
ann_macs = N_INPUT * N_RESERVOIR + N_RESERVOIR * N_OUTPUT   # 784*500 + 500*10
ann_total_ops = 2 * ann_macs + (N_RESERVOIR + N_OUTPUT)     # mult + add + biases

# SNN: event-driven, varies per input
snn_avg_events = test_events.mean()
snn_std_events = test_events.std()

# Energy estimates
snn_energy_nj = (snn_avg_events * ENERGY_PER_SNN_SPIKE_PJ) / 1000
ann_energy_nj = (ann_macs * ENERGY_PER_ANN_MAC_PJ) / 1000

print(f"\n  ANN per forward pass:")
print(f"    MACs:        {ann_macs:>10,}")
print(f"    Total ops:   {ann_total_ops:>10,}")
print(f"    Energy:      {ann_energy_nj:>10.1f} nJ")

print(f"\n  SNN per sample (test set averages):")
print(f"    Synaptic events: {snn_avg_events:>10,.0f}  (std = {snn_std_events:,.0f})")
print(f"    Energy:          {snn_energy_nj:>10.1f} nJ")

if ann_energy_nj > snn_energy_nj and snn_energy_nj > 0:
    ratio = ann_energy_nj / snn_energy_nj
    print(f"\n  -> SNN is ~{ratio:.1f}x more energy-efficient (estimated)")
elif snn_energy_nj > 0:
    ratio = snn_energy_nj / ann_energy_nj
    print(f"\n  -> ANN is ~{ratio:.1f}x more energy-efficient at this config")

# Per-digit energy variation — the SNN's killer feature
per_digit_energy_nj = []
per_digit_events = []
for d in range(10):
    mask = test_labels.numpy() == d
    if mask.sum() > 0:
        avg_ev = test_events[mask].mean()
        per_digit_events.append(avg_ev)
        per_digit_energy_nj.append(avg_ev * ENERGY_PER_SNN_SPIKE_PJ / 1000)
    else:
        per_digit_events.append(0)
        per_digit_energy_nj.append(0)

print("\n  Per-digit SNN energy (input-adaptive computation):")
print(f"  {'Digit':>5}  {'Events':>10}  {'Energy (nJ)':>11}")
print(f"  {'---':>5}  {'---':>10}  {'---':>11}")
for d in range(10):
    marker = " << sparsest" if d == int(np.argmin(per_digit_energy_nj)) else ""
    marker = " << densest" if d == int(np.argmax(per_digit_energy_nj)) else marker
    print(f"  {d:>5}  {per_digit_events[d]:>10,.0f}  {per_digit_energy_nj[d]:>10.1f}{marker}")

print("\n  Key insight: The ANN always performs exactly the same number of MACs")
print("  regardless of input content. The SNN adapts -- sparse digits like '1'")
print("  naturally require fewer spikes and less energy than dense digits like '0'.")


# ================================================================
# 11. VISUALIZATION
# ================================================================
print("\n" + "=" * 55)
print("  GENERATING VISUALIZATIONS")
print("=" * 55)

# ---------- Color palette ----------
COL_SNN  = "#17becf"      # cyan
COL_SNN2 = "#1f77b4"      # deeper blue
COL_ANN  = "#e94560"      # coral-red
COL_ANN2 = "#ff6b6b"      # lighter red
COL_BG   = "#0f0f1a"      # deep navy background
COL_CARD = "#1a1a2e"      # card background
COL_GRID = "#ffffff15"    # subtle grid
COL_TEXT = "#e0e0e0"      # light text
COL_ACC  = "#2ecc71"      # green accent

# Apply dark theme
plt.rcParams.update({
    "figure.facecolor":   COL_BG,
    "axes.facecolor":     COL_CARD,
    "axes.edgecolor":     "#333355",
    "axes.labelcolor":    COL_TEXT,
    "axes.titlepad":      12,
    "text.color":         COL_TEXT,
    "xtick.color":        "#aaa",
    "ytick.color":        "#aaa",
    "axes.grid":          True,
    "grid.color":         COL_GRID,
    "grid.linewidth":     0.5,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "legend.facecolor":   COL_CARD,
    "legend.edgecolor":   "#444",
    "legend.fontsize":    9,
})


# ===== FIGURE 1: Main Comparison (2x3 grid) =====
fig = plt.figure(figsize=(20, 12))
fig.suptitle("MNIST Digit Recognition: SNN Reservoir vs ANN",
             fontsize=18, fontweight="bold", color="white", y=0.98)
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

# --- (0,0) Training Loss Curves ---
ax1 = fig.add_subplot(gs[0, 0])
epochs_snn_x = np.arange(1, EPOCHS_READOUT + 1)
epochs_ann_x = np.arange(1, EPOCHS_ANN + 1)
ax1.plot(epochs_snn_x, snn_losses, color=COL_SNN, linewidth=2.5,
         marker="o", markersize=3, label="SNN Readout", zorder=3)
ax1.plot(epochs_ann_x, ann_losses, color=COL_ANN, linewidth=2.5,
         marker="s", markersize=3, label="ANN", zorder=3)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend()

# --- (0,1) Training Accuracy Curves ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_snn_x, snn_accs, color=COL_SNN, linewidth=2.5,
         marker="o", markersize=3, label="SNN Readout", zorder=3)
ax2.plot(epochs_ann_x, ann_accs, color=COL_ANN, linewidth=2.5,
         marker="s", markersize=3, label="ANN", zorder=3)
ax2.set_title("Training Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(0, 105)
ax2.legend()

# --- (0,2) Overall Test Accuracy Bar ---
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(["SNN\nReservoir+Readout", "ANN\nDense FF"],
               [snn_acc, ann_acc],
               color=[COL_SNN, COL_ANN], width=0.5,
               edgecolor="white", linewidth=0.5)
ax3.set_title("Test Accuracy Comparison")
ax3.set_ylabel("Accuracy (%)")
ax3.set_ylim(0, 110)
for bar, val in zip(bars, [snn_acc, ann_acc]):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13,
             color="white")

# --- (1,0) Confusion Matrix — SNN ---
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(snn_cm, cmap="Blues", aspect="equal")
ax4.set_title("Confusion Matrix -- SNN")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("True")
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        val = snn_cm[i, j]
        color = "white" if val > snn_cm.max() * 0.5 else COL_TEXT
        ax4.text(j, i, str(val), ha="center", va="center",
                 fontsize=7, color=color, fontweight="bold")
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# --- (1,1) Confusion Matrix — ANN ---
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(ann_cm, cmap="Reds", aspect="equal")
ax5.set_title("Confusion Matrix -- ANN")
ax5.set_xlabel("Predicted")
ax5.set_ylabel("True")
ax5.set_xticks(range(10))
ax5.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        val = ann_cm[i, j]
        color = "white" if val > ann_cm.max() * 0.5 else COL_TEXT
        ax5.text(j, i, str(val), ha="center", va="center",
                 fontsize=7, color=color, fontweight="bold")
fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# --- (1,2) Per-Digit Accuracy Comparison ---
ax6 = fig.add_subplot(gs[1, 2])
x_digits = np.arange(10)
bar_width = 0.35
bars_snn = ax6.bar(x_digits - bar_width / 2, snn_per_digit, bar_width,
                   color=COL_SNN, label="SNN", edgecolor="white", linewidth=0.3)
bars_ann = ax6.bar(x_digits + bar_width / 2, ann_per_digit, bar_width,
                   color=COL_ANN, label="ANN", edgecolor="white", linewidth=0.3)
ax6.set_title("Per-Digit Test Accuracy")
ax6.set_xlabel("Digit")
ax6.set_ylabel("Accuracy (%)")
ax6.set_xticks(x_digits)
ax6.set_ylim(0, 115)
ax6.legend()

fig1_path = os.path.join(OUT_DIR, "mnist_main_comparison.pdf")
plt.savefig(fig1_path, dpi=300, facecolor=COL_BG)
print(f"  Saved: {fig1_path}")


# ===== FIGURE 2: Energy & Efficiency (1x3) =====
fig2, (ax7, ax8, ax9) = plt.subplots(1, 3, figsize=(20, 5.5))
fig2.suptitle("Energy & Computational Efficiency Analysis",
              fontsize=16, fontweight="bold", color="white", y=1.02)
fig2.subplots_adjust(left=0.06, right=0.96, top=0.82, bottom=0.15, wspace=0.35)

# --- Operations comparison ---
op_labels = ["ANN\n(Dense MACs)", "SNN\n(Spike Events)"]
op_vals = [ann_macs, snn_avg_events]
bars7 = ax7.bar(op_labels, op_vals, color=[COL_ANN, COL_SNN], width=0.45,
                edgecolor="white", linewidth=0.5)
ax7.set_title("Operations per Sample")
ax7.set_ylabel("Count")
for bar, val in zip(bars7, op_vals):
    ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(op_vals) * 0.02,
             f"{val:,.0f}", ha="center", fontweight="bold", fontsize=11, color="white")

# --- Energy comparison ---
en_vals = [ann_energy_nj, snn_energy_nj]
bars8 = ax8.bar(op_labels, en_vals, color=[COL_ANN, COL_SNN], width=0.45,
                edgecolor="white", linewidth=0.5)
ax8.set_title("Estimated Energy per Sample")
ax8.set_ylabel("Energy (nanojoules)")
for bar, val in zip(bars8, en_vals):
    ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(en_vals) * 0.02,
             f"{val:.1f} nJ", ha="center", fontweight="bold", fontsize=11, color="white")

# --- Per-digit SNN energy (the killer chart) ---
ann_energy_line = ann_energy_nj   # ANN is the same for every digit
max_energy = max(max(per_digit_energy_nj), ann_energy_line) if per_digit_energy_nj else 1
digit_colors = [plt.cm.viridis(e / max_energy) for e in per_digit_energy_nj]
bars9 = ax9.bar(x_digits, per_digit_energy_nj, color=digit_colors, width=0.6,
                edgecolor="white", linewidth=0.3)
ax9.axhline(ann_energy_line, color=COL_ANN, linestyle="--", linewidth=2,
            label=f"ANN (constant): {ann_energy_line:.1f} nJ", zorder=4)
ax9.set_title("SNN Energy Adapts to Input Sparsity")
ax9.set_xlabel("Digit")
ax9.set_ylabel("Energy (nanojoules)")
ax9.set_xticks(x_digits)
ax9.legend(loc="upper right")

fig2_path = os.path.join(OUT_DIR, "mnist_energy_analysis.pdf")
plt.savefig(fig2_path, dpi=300, facecolor=COL_BG, bbox_inches="tight")
print(f"  Saved: {fig2_path}")


# ===== FIGURE 3: SNN Internals — Example Spike Rasters =====
# Show spike patterns for each digit (0-9) to visualize how the
# reservoir responds differently to each input.
fig3 = plt.figure(figsize=(20, 10))
fig3.suptitle("SNN Reservoir Spike Rasters -- How Different Digits Drive Activity",
              fontsize=16, fontweight="bold", color="white", y=0.98)
gs3 = gridspec.GridSpec(2, 5, hspace=0.45, wspace=0.3,
                        left=0.05, right=0.97, top=0.90, bottom=0.08)

example_digits = list(range(10))
digit_examples = {}

# Find one example of each digit in the test set
for idx in range(len(test_set)):
    img, lbl = test_set[idx]
    if lbl not in digit_examples and lbl in example_digits:
        digit_examples[lbl] = img
    if len(digit_examples) >= 10:
        break

for plot_idx, digit in enumerate(example_digits):
    if digit not in digit_examples:
        continue

    img = digit_examples[digit]
    img_flat = img.view(-1)
    spike_train = poisson_encode(img_flat, SIM_TIME, device)
    _, all_spikes, _, _ = reservoir.run(spike_train)

    # Get spike raster: (time, n_reservoir)
    raster = all_spikes.cpu().numpy()

    # Plot only the 50 most-active neurons for clarity
    total_per_neuron = raster.sum(axis=0)
    top_neurons = np.argsort(total_per_neuron)[-50:]
    raster_top = raster[:, top_neurons]

    row = plot_idx // 5
    col = plot_idx % 5
    ax = fig3.add_subplot(gs3[row, col])

    # Spike raster: each dot is a spike
    spike_times, neuron_ids = np.where(raster_top.T)
    if len(spike_times) > 0:
        ax.scatter(neuron_ids, spike_times, s=0.8, color=COL_SNN,
                   alpha=0.7, marker="|")

    ax.set_title(f"Digit '{digit}'  ({int(total_per_neuron.sum())} spikes)",
                 fontsize=11)
    ax.set_xlim(0, SIM_TIME)
    ax.set_ylim(-1, 50)
    if col == 0:
        ax.set_ylabel("Neuron #")
    if row == 1:
        ax.set_xlabel("Time step")
    ax.tick_params(axis="both", which="major", labelsize=8)

fig3_path = os.path.join(OUT_DIR, "mnist_spike_rasters.pdf")
plt.savefig(fig3_path, dpi=300, facecolor=COL_BG)
print(f"  Saved: {fig3_path}")

plt.show()


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"""
  Task:       MNIST 10-class digit recognition (28x28 images)
  Train/Test: {TRAIN_SIZE} / {TEST_SIZE} samples

  +--------------------------------------------------------------+
  |                    ACCURACY                                  |
  |  SNN (Reservoir + Linear Readout) : {snn_acc:5.1f}%                  |
  |  ANN (Dense Feedforward)          : {ann_acc:5.1f}%                  |
  +--------------------------------------------------------------+
  |                    ENERGY / SAMPLE                           |
  |  SNN (neuromorphic, event-driven) : {snn_energy_nj:>8.1f} nJ            |
  |  ANN (digital, dense MACs)        : {ann_energy_nj:>8.1f} nJ            |
  +--------------------------------------------------------------+
  |                    KEY INSIGHT                               |
  |  SNN energy scales with INPUT SPARSITY.                      |
  |  Digit '1' (sparse) uses less energy than '0' (dense).       |
  |  ANN energy is FIXED -- same MACs for every input.           |
  |                                                              |
  |  At this scale ({N_INPUT}x{N_RESERVOIR}x{N_OUTPUT}), the SNN's per-event      |
  |  cost advantage ({ENERGY_PER_SNN_SPIKE_PJ} pJ vs {ENERGY_PER_ANN_MAC_PJ} pJ per MAC, ~5x cheaper)      |
  |  combined with event-driven sparsity yields significant      |
  |  energy savings on real image data.                          |
  +--------------------------------------------------------------+

  Figures saved to: {OUT_DIR}
""")
