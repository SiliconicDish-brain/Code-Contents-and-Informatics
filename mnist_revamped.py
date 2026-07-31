"""
MNIST SNN vs ANN — REVAMPED with Hyperparameter Tuning
=======================================================

The original SNN hit 73.1% accuracy. This script systematically searches
for a better reservoir configuration by tuning:

  • Reservoir size (500 → 1500 neurons)
  • Input connectivity & recurrent connectivity (true LSM)
  • LIF decay, threshold, and input scaling (prevent saturation)
  • Simulation time (temporal richness)
  • Feature normalization
  • Readout architecture (linear vs MLP)

Strategy:
  Phase 1 — Search 15 reservoir configs × 2 readout types on 3k train / 1k val
  Phase 2 — Retrain best config on 10k train, evaluate on 2k test
  Phase 3 — Train matched ANN and compare
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import time as timer
import random
import copy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ================================================================
# 1. CONFIGURATION
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- Data sizes ---
SEARCH_TRAIN = 3000
SEARCH_VAL   = 1000
FINAL_TRAIN  = 10000
FINAL_TEST   = 2000

# --- ANN baseline ---
EPOCHS_ANN = 20
LR_ANN     = 0.001
BATCH_ANN  = 64

# --- Readout training ---
EPOCHS_READOUT = 30
LR_READOUT     = 0.003
BATCH_READOUT  = 64

# --- Energy model (picojoules per event) ---
ENERGY_PER_SNN_SPIKE_PJ = 0.9
ENERGY_PER_ANN_MAC_PJ   = 4.6

# --- Output ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "Final")
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# 2. LIF RESERVOIR (with optional recurrent connections)
# ================================================================
class LIFReservoir:
    """
    Leaky Integrate-and-Fire reservoir with:
      • Sparse random input weights (Input → Reservoir)
      • Optional sparse recurrent weights (Reservoir → Reservoir)
      • Configurable decay, threshold, input scaling
    """

    def __init__(self, n_input, n_reservoir, conn_in, conn_rec,
                 thresh, reset, decay, input_scale, device):
        self.n_input = n_input
        self.n_res = n_reservoir
        self.thresh = thresh
        self.reset_val = reset
        self.decay = decay
        self.input_scale = input_scale
        self.device = device

        # --- Input weights (sparse) ---
        fan_in = max(int(n_input * conn_in), 1)
        w_scale = 1.0 / np.sqrt(fan_in)
        w_in = torch.rand(n_input, n_reservoir, device=device) * w_scale
        mask_in = (torch.rand(n_input, n_reservoir, device=device) < conn_in)
        self.W_in = w_in * mask_in.float()

        self.n_active_in = int(mask_in.sum().item())
        self.avg_fanout = self.n_active_in / n_input

        # --- Recurrent weights (sparse, both excitatory & inhibitory) ---
        self.has_recurrent = conn_rec > 0
        if self.has_recurrent:
            fan_rec = max(int(n_reservoir * conn_rec), 1)
            w_scale_rec = 0.5 / np.sqrt(fan_rec)
            w_rec = torch.randn(n_reservoir, n_reservoir, device=device) * w_scale_rec
            mask_rec = (torch.rand(n_reservoir, n_reservoir, device=device) < conn_rec)
            w_rec = w_rec * mask_rec.float()
            w_rec.fill_diagonal_(0)  # no self-connections
            self.W_rec = w_rec
            self.n_active_rec = int(mask_rec.sum().item())
        else:
            self.W_rec = None
            self.n_active_rec = 0

        # State
        self.v = torch.zeros(n_reservoir, device=device)

    def reset_state(self):
        self.v = torch.zeros(self.n_res, device=self.device)

    def run(self, spike_train):
        """
        Run reservoir for a full spike train.
        Args:   spike_train: (T, n_input) binary tensor
        Returns: spike_counts (n_res,), total_input_spikes (scalar)
        """
        T = spike_train.shape[0]
        spike_counts = torch.zeros(self.n_res, device=self.device)
        prev_spikes = torch.zeros(self.n_res, device=self.device)
        self.reset_state()

        for t in range(T):
            # Input current
            I = spike_train[t] @ self.W_in

            # Recurrent current
            if self.has_recurrent:
                I = I + prev_spikes @ self.W_rec

            # Leaky integration
            self.v = self.decay * self.v + I

            # Spike generation
            spikes = (self.v >= self.thresh).float()
            spike_counts += spikes

            # Reset spiked neurons
            self.v[spikes.bool()] = self.reset_val
            prev_spikes = spikes

        total_input_spikes = spike_train.sum().item()
        return spike_counts, total_input_spikes

    def run_with_raster(self, spike_train):
        """Same as run() but also returns the full spike raster."""
        T = spike_train.shape[0]
        all_spikes = torch.zeros(T, self.n_res, device=self.device)
        prev_spikes = torch.zeros(self.n_res, device=self.device)
        self.reset_state()

        for t in range(T):
            I = spike_train[t] @ self.W_in
            if self.has_recurrent:
                I = I + prev_spikes @ self.W_rec
            self.v = self.decay * self.v + I
            spikes = (self.v >= self.thresh).float()
            all_spikes[t] = spikes
            self.v[spikes.bool()] = self.reset_val
            prev_spikes = spikes

        spike_counts = all_spikes.sum(dim=0)
        total_input_spikes = spike_train.sum().item()
        return spike_counts, all_spikes, total_input_spikes


# ================================================================
# 3. LOAD MNIST
# ================================================================
print("\n" + "=" * 60)
print("  LOADING MNIST")
print("=" * 60)

transform = transforms.Compose([transforms.ToTensor()])
data_dir = os.path.join(SCRIPT_DIR, "data")

train_full = datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform)
test_full  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

# Search phase splits
search_train = Subset(train_full, range(SEARCH_TRAIN))
search_val   = Subset(train_full, range(SEARCH_TRAIN, SEARCH_TRAIN + SEARCH_VAL))

# Final evaluation splits
final_train = Subset(train_full, range(FINAL_TRAIN))
final_test  = Subset(test_full,  range(FINAL_TEST))

search_train_loader = DataLoader(search_train, batch_size=1, shuffle=False)
search_val_loader   = DataLoader(search_val,   batch_size=1, shuffle=False)
final_train_loader  = DataLoader(final_train,  batch_size=1, shuffle=False)
final_test_loader   = DataLoader(final_test,   batch_size=1, shuffle=False)

# ANN loaders
ann_train_loader = DataLoader(final_train, batch_size=BATCH_ANN, shuffle=True)
ann_test_loader  = DataLoader(final_test,  batch_size=BATCH_ANN, shuffle=False)

print(f"Search phase:  {SEARCH_TRAIN} train / {SEARCH_VAL} val")
print(f"Final phase:   {FINAL_TRAIN} train / {FINAL_TEST} test")


# ================================================================
# 4. UTILITY FUNCTIONS
# ================================================================
def poisson_encode(image_flat, time_steps, input_scale, dev):
    """Poisson rate coding with adjustable input scale."""
    rates = (image_flat * input_scale).unsqueeze(0).expand(time_steps, -1).to(dev)
    return (torch.rand_like(rates) < rates).float()


def extract_features(loader, reservoir, sim_time, input_scale, desc=""):
    """Extract spike-count features from reservoir for all samples."""
    features, labels, events_list = [], [], []
    n = len(loader)
    t0 = timer.time()

    for i, (img, lbl) in enumerate(loader):
        spike_train = poisson_encode(img.view(-1), sim_time, input_scale, device)
        spike_counts, input_spikes = reservoir.run(spike_train)

        features.append(spike_counts.unsqueeze(0).cpu())
        labels.append(lbl.item())

        res_spikes = spike_counts.sum().item()
        events_list.append(input_spikes * reservoir.avg_fanout + res_spikes)

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            elapsed = timer.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"    {desc}: {i+1:>5}/{n}  ({rate:.0f} s/s, ETA {eta:.0f}s)")

    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    events = np.array(events_list)
    return features, labels, events


def normalize_features(train_f, val_f):
    """Zero-mean, unit-variance normalization (fit on train)."""
    mean = train_f.mean(dim=0, keepdim=True)
    std  = train_f.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_f - mean) / std, (val_f - mean) / std, mean, std


# ================================================================
# 5. READOUT MODELS
# ================================================================
class LinearReadout(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
    def forward(self, x):
        return self.fc(x)


class MLPReadout(nn.Module):
    def __init__(self, n_in, n_out):
        n_hidden = min(n_in // 2, 256)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden // 2, n_out),
        )
    def forward(self, x):
        return self.net(x)


def train_readout(model, train_x, train_y, epochs=EPOCHS_READOUT, lr=LR_READOUT):
    """Train a readout model on pre-extracted features. Returns best val-like loss."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_x_d = train_x.to(device)
    train_y_d = train_y.to(device)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x_d), device=device)
        for start in range(0, len(perm), BATCH_READOUT):
            idx = perm[start:start + BATCH_READOUT]
            opt.zero_grad()
            loss = loss_fn(model(train_x_d[idx]), train_y_d[idx])
            loss.backward()
            opt.step()
    return model


def eval_readout(model, test_x, test_y):
    """Evaluate readout accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        preds = logits.argmax(dim=1).cpu()
    return (preds == test_y).float().mean().item() * 100


# ================================================================
# 6. DEFINE SEARCH CONFIGURATIONS
# ================================================================
print("\n" + "=" * 60)
print("  DEFINING HYPERPARAMETER SEARCH SPACE")
print("=" * 60)

# Hand-picked configs (informed by LSM/reservoir computing literature)
hand_picked = [
    {
        "tag": "HP1-LargeRecurrent",
        "n_reservoir": 1500, "conn_in": 0.15, "conn_rec": 0.10,
        "decay": 0.90, "thresh": 2.0, "sim_time": 100,
        "input_scale": 0.5, "normalize": True,
    },
    {
        "tag": "HP2-DenseLarge",
        "n_reservoir": 1000, "conn_in": 0.30, "conn_rec": 0.0,
        "decay": 0.85, "thresh": 2.0, "sim_time": 75,
        "input_scale": 0.5, "normalize": True,
    },
    {
        "tag": "HP3-HighRecurrence",
        "n_reservoir": 750, "conn_in": 0.15, "conn_rec": 0.15,
        "decay": 0.90, "thresh": 1.5, "sim_time": 100,
        "input_scale": 0.5, "normalize": True,
    },
    {
        "tag": "HP4-VeryLargeSparse",
        "n_reservoir": 1500, "conn_in": 0.10, "conn_rec": 0.05,
        "decay": 0.95, "thresh": 3.0, "sim_time": 100,
        "input_scale": 0.3, "normalize": True,
    },
    {
        "tag": "HP5-BaselineImproved",
        "n_reservoir": 500, "conn_in": 0.20, "conn_rec": 0.10,
        "decay": 0.90, "thresh": 2.0, "sim_time": 100,
        "input_scale": 0.5, "normalize": True,
    },
]

# Random configs
search_space = {
    "n_reservoir": [500, 750, 1000, 1250, 1500],
    "conn_in":     [0.10, 0.15, 0.20, 0.25, 0.30],
    "conn_rec":    [0.0, 0.0, 0.05, 0.10, 0.15],    # bias toward no/low recurrence
    "decay":       [0.75, 0.80, 0.85, 0.90, 0.95],
    "thresh":      [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "sim_time":    [50, 75, 100],
    "input_scale": [0.3, 0.5, 0.7, 1.0],
    "normalize":   [True, False],
}

random_configs = []
for i in range(10):
    cfg = {k: random.choice(v) for k, v in search_space.items()}
    cfg["tag"] = f"Rand-{i+1}"
    random_configs.append(cfg)

all_configs = hand_picked + random_configs
print(f"Total reservoir configurations to evaluate: {len(all_configs)}")
print(f"Each tested with Linear + MLP readout = {len(all_configs) * 2} evaluations")


# ================================================================
# 7. RUN HYPERPARAMETER SEARCH
# ================================================================
print("\n" + "=" * 60)
print("  PHASE 1: HYPERPARAMETER SEARCH")
print("=" * 60)

results = []
t_search_start = timer.time()

for cfg_idx, cfg in enumerate(all_configs):
    tag = cfg["tag"]
    print(f"\n--- Config {cfg_idx+1}/{len(all_configs)}: {tag} ---")
    print(f"    n={cfg['n_reservoir']}, conn_in={cfg['conn_in']}, "
          f"conn_rec={cfg['conn_rec']}, decay={cfg['decay']}, "
          f"thresh={cfg['thresh']}, T={cfg['sim_time']}, "
          f"scale={cfg['input_scale']}, norm={cfg['normalize']}")

    # Build reservoir
    res = LIFReservoir(
        n_input=784, n_reservoir=cfg["n_reservoir"],
        conn_in=cfg["conn_in"], conn_rec=cfg["conn_rec"],
        thresh=cfg["thresh"], reset=0.0, decay=cfg["decay"],
        input_scale=cfg["input_scale"], device=device,
    )

    # Extract features
    t0 = timer.time()
    train_f, train_l, _ = extract_features(
        search_train_loader, res, cfg["sim_time"], cfg["input_scale"], desc="Train"
    )
    val_f, val_l, _ = extract_features(
        search_val_loader, res, cfg["sim_time"], cfg["input_scale"], desc="Val  "
    )
    t_feat = timer.time() - t0

    # Optional normalization
    if cfg["normalize"]:
        train_fn, val_fn, _, _ = normalize_features(train_f, val_f)
    else:
        train_fn, val_fn = train_f, val_f

    # Sanity check: reservoir activity
    avg_activity = train_f.mean().item()
    print(f"    Avg spikes/neuron: {avg_activity:.2f}  "
          f"(feat extract: {t_feat:.1f}s)")

    if avg_activity < 0.01:
        print("    SKIP — reservoir is silent")
        results.append({**cfg, "acc_linear": 0.0, "acc_mlp": 0.0,
                        "activity": avg_activity, "time": t_feat})
        continue

    # --- Linear readout ---
    linear_model = LinearReadout(cfg["n_reservoir"], 10)
    train_readout(linear_model, train_fn, train_l)
    acc_lin = eval_readout(linear_model, val_fn, val_l)

    # --- MLP readout ---
    mlp_model = MLPReadout(cfg["n_reservoir"], 10)
    train_readout(mlp_model, train_fn, train_l)
    acc_mlp = eval_readout(mlp_model, val_fn, val_l)

    best = max(acc_lin, acc_mlp)
    best_type = "MLP" if acc_mlp >= acc_lin else "Linear"
    print(f"    Linear: {acc_lin:.1f}% | MLP: {acc_mlp:.1f}% | "
          f"Best: {best:.1f}% ({best_type})")

    results.append({
        **cfg,
        "acc_linear": acc_lin,
        "acc_mlp": acc_mlp,
        "best_acc": best,
        "best_readout": best_type,
        "activity": avg_activity,
        "time": t_feat,
    })

t_search = timer.time() - t_search_start


# ================================================================
# 8. RESULTS ANALYSIS
# ================================================================
print("\n" + "=" * 60)
print("  SEARCH RESULTS (sorted by best accuracy)")
print("=" * 60)

results.sort(key=lambda r: r.get("best_acc", 0), reverse=True)

print(f"\n  {'Rank':>4} {'Tag':<22} {'N':>5} {'CIn':>5} {'CRec':>5} "
      f"{'Dec':>5} {'Thr':>5} {'T':>4} {'Scl':>4} {'Nrm':>4} "
      f"{'Linear':>7} {'MLP':>7} {'Best':>7}")
print("  " + "-" * 110)

for i, r in enumerate(results):
    print(f"  {i+1:>4} {r['tag']:<22} {r['n_reservoir']:>5} "
          f"{r['conn_in']:>5.2f} {r['conn_rec']:>5.2f} "
          f"{r['decay']:>5.2f} {r['thresh']:>5.1f} {r['sim_time']:>4} "
          f"{r['input_scale']:>4.1f} {'Y' if r['normalize'] else 'N':>4} "
          f"{r.get('acc_linear',0):>6.1f}% {r.get('acc_mlp',0):>6.1f}% "
          f"{r.get('best_acc',0):>6.1f}%")

best_cfg = results[0]
print(f"\nSearch completed in {t_search:.0f}s")
print(f"\nBest config: {best_cfg['tag']} -> {best_cfg['best_acc']:.1f}% "
      f"({best_cfg['best_readout']} readout)")

# Analyze which hyperparameters matter
print("\n--- Hyperparameter Impact (avg best_acc by value) ---")
for param in ["n_reservoir", "conn_rec", "decay", "thresh", "sim_time",
              "input_scale", "normalize"]:
    vals = {}
    for r in results:
        v = r[param]
        if v not in vals:
            vals[v] = []
        vals[v].append(r.get("best_acc", 0))
    print(f"  {param}:")
    for v in sorted(vals.keys(), key=lambda x: (isinstance(x, bool), x)):
        avg = np.mean(vals[v])
        n = len(vals[v])
        print(f"    {str(v):>8} → {avg:.1f}% (n={n})")


# ================================================================
# 9. FINAL EVALUATION — Best Config on More Data
# ================================================================
print("\n" + "=" * 60)
print(f"  PHASE 2: FINAL EVALUATION (best config on {FINAL_TRAIN} train / {FINAL_TEST} test)")
print("=" * 60)

bc = best_cfg  # shorthand

# Rebuild reservoir with best hyperparameters (fresh random weights)
torch.manual_seed(SEED)
final_reservoir = LIFReservoir(
    n_input=784, n_reservoir=bc["n_reservoir"],
    conn_in=bc["conn_in"], conn_rec=bc["conn_rec"],
    thresh=bc["thresh"], reset=0.0, decay=bc["decay"],
    input_scale=bc["input_scale"], device=device,
)

print(f"Reservoir: 784 → {bc['n_reservoir']} LIF neurons")
print(f"  Input synapses:     {final_reservoir.n_active_in:,} ({bc['conn_in']*100:.0f}%)")
print(f"  Recurrent synapses: {final_reservoir.n_active_rec:,} ({bc['conn_rec']*100:.0f}%)")
print(f"  LIF: decay={bc['decay']}, thresh={bc['thresh']}")
print(f"  Sim time: {bc['sim_time']}, input scale: {bc['input_scale']}")

# Extract features
t0 = timer.time()
final_train_f, final_train_l, final_train_ev = extract_features(
    final_train_loader, final_reservoir, bc["sim_time"], bc["input_scale"], "Train"
)
final_test_f, final_test_l, final_test_ev = extract_features(
    final_test_loader, final_reservoir, bc["sim_time"], bc["input_scale"], "Test "
)
t_final_extract = timer.time() - t0
print(f"Feature extraction: {t_final_extract:.1f}s")

# Normalize
if bc["normalize"]:
    final_train_fn, final_test_fn, feat_mean, feat_std = normalize_features(
        final_train_f, final_test_f
    )
else:
    final_train_fn, final_test_fn = final_train_f, final_test_f

# Train best readout type
if bc["best_readout"] == "MLP":
    final_readout = MLPReadout(bc["n_reservoir"], 10)
else:
    final_readout = LinearReadout(bc["n_reservoir"], 10)

# Train with more epochs for final model
train_readout(final_readout, final_train_fn, final_train_l,
              epochs=50, lr=LR_READOUT)

snn_acc = eval_readout(final_readout, final_test_fn, final_test_l)

# Per-digit predictions
final_readout.eval()
with torch.no_grad():
    snn_preds = final_readout(final_test_fn.to(device)).argmax(dim=1).cpu().numpy()
snn_true = final_test_l.numpy()

print(f"\n  SNN Final Accuracy: {snn_acc:.1f}%")


# ================================================================
# 10. TRAIN ANN BASELINE (same hidden size)
# ================================================================
print("\n" + "=" * 60)
print(f"  TRAINING ANN BASELINE ({EPOCHS_ANN} epochs, {bc['n_reservoir']} hidden)")
print("=" * 60)


class ANN(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


ann = ANN(bc["n_reservoir"]).to(device)
opt_ann = optim.Adam(ann.parameters(), lr=LR_ANN)

ann_losses, ann_accs = [], []
t0 = timer.time()
for epoch in range(EPOCHS_ANN):
    ann.train()
    ep_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in ann_train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt_ann.zero_grad()
        logits = ann(imgs)
        loss = nn.CrossEntropyLoss()(logits, lbls)
        loss.backward()
        opt_ann.step()
        ep_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == lbls).sum().item()
        total += len(imgs)
    ann_losses.append(ep_loss / total)
    ann_accs.append(correct / total * 100)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:>3}/{EPOCHS_ANN} | "
              f"Loss: {ann_losses[-1]:.4f} | Acc: {ann_accs[-1]:.1f}%")

t_ann = timer.time() - t0

ann.eval()
ann_preds_list, ann_true_list = [], []
with torch.no_grad():
    for imgs, lbls in ann_test_loader:
        logits = ann(imgs.to(device))
        ann_preds_list.append(logits.argmax(1).cpu().numpy())
        ann_true_list.append(lbls.numpy())
ann_preds = np.concatenate(ann_preds_list)
ann_true = np.concatenate(ann_true_list)
ann_acc = (ann_preds == ann_true).mean() * 100

print(f"ANN training: {t_ann:.1f}s")


# ================================================================
# 11. COMPREHENSIVE COMPARISON
# ================================================================
print("\n" + "=" * 60)
print("  FINAL COMPARISON")
print("=" * 60)

# Per-digit accuracy
snn_per_digit, ann_per_digit = [], []
for d in range(10):
    ms = snn_true == d
    ma = ann_true == d
    snn_per_digit.append((snn_preds[ms] == d).mean() * 100 if ms.sum() > 0 else 0)
    ann_per_digit.append((ann_preds[ma] == d).mean() * 100 if ma.sum() > 0 else 0)

# Confusion matrices
def confusion_matrix(true, pred, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        cm[int(t)][int(p)] += 1
    return cm

snn_cm = confusion_matrix(snn_true, snn_preds)
ann_cm = confusion_matrix(ann_true, ann_preds)

# Energy
N_RES = bc["n_reservoir"]
ann_macs = 784 * N_RES + N_RES * 10
snn_avg_events = final_test_ev.mean()
snn_energy_nj = snn_avg_events * ENERGY_PER_SNN_SPIKE_PJ / 1000
ann_energy_nj = ann_macs * ENERGY_PER_ANN_MAC_PJ / 1000

per_digit_energy = []
for d in range(10):
    mask = final_test_l.numpy() == d
    if mask.sum() > 0:
        per_digit_energy.append(final_test_ev[mask].mean() * ENERGY_PER_SNN_SPIKE_PJ / 1000)
    else:
        per_digit_energy.append(0)

print(f"\n  +---------------------------------------------------+")
print(f"  |  ORIGINAL SNN (73.1%)  →  TUNED SNN: {snn_acc:5.1f}%       |")
print(f"  |  ANN Baseline:                       {ann_acc:5.1f}%       |")
print(f"  +---------------------------------------------------+")
print(f"  |  Improvement: +{snn_acc - 73.1:.1f}% from hyperparameter tuning  |")
print(f"  +---------------------------------------------------+")
print(f"  |  Energy — SNN: {snn_energy_nj:.1f} nJ  |  ANN: {ann_energy_nj:.1f} nJ     |")
if ann_energy_nj > snn_energy_nj and snn_energy_nj > 0:
    print(f"  |  SNN is ~{ann_energy_nj/snn_energy_nj:.1f}x more energy-efficient          |")
print(f"  +---------------------------------------------------+")

print(f"\n  Best config details:")
for k, v in bc.items():
    if k not in ("tag", "acc_linear", "acc_mlp", "best_acc", "best_readout",
                 "activity", "time"):
        print(f"    {k}: {v}")
print(f"    readout: {bc['best_readout']}")

print(f"\n  Per-digit accuracy:")
print(f"  {'Digit':>5}  {'SNN':>7}  {'ANN':>7}  {'Gap':>7}")
print(f"  {'---':>5}  {'---':>7}  {'---':>7}  {'---':>7}")
for d in range(10):
    gap = snn_per_digit[d] - ann_per_digit[d]
    print(f"  {d:>5}  {snn_per_digit[d]:>6.1f}%  {ann_per_digit[d]:>6.1f}%  {gap:>+6.1f}%")


# ================================================================
# 12. VISUALIZATION
# ================================================================
print("\n" + "=" * 60)
print("  GENERATING VISUALIZATIONS")
print("=" * 60)

# --- Style ---
COL_SNN  = "#17becf"
COL_ANN  = "#e94560"
COL_BG   = "#0f0f1a"
COL_CARD = "#1a1a2e"
COL_TEXT = "#e0e0e0"
COL_GOLD = "#f1c40f"

plt.rcParams.update({
    "figure.facecolor": COL_BG, "axes.facecolor": COL_CARD,
    "axes.edgecolor": "#333355", "axes.labelcolor": COL_TEXT,
    "axes.titlepad": 12, "text.color": COL_TEXT,
    "xtick.color": "#aaa", "ytick.color": "#aaa",
    "axes.grid": True, "grid.color": "#ffffff15", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.facecolor": COL_CARD, "legend.edgecolor": "#444", "legend.fontsize": 9,
})


# ===== FIGURE 1: Search Results Heatmap + Best Config Comparison =====
fig = plt.figure(figsize=(22, 14))
fig.suptitle(f"MNIST SNN Revamped — Tuned to {snn_acc:.1f}% (was 73.1%)",
             fontsize=20, fontweight="bold", color=COL_GOLD, y=0.99)
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

# (0,0) Search results bar chart (top 15)
ax1 = fig.add_subplot(gs[0, 0])
top_n = min(15, len(results))
tags = [r["tag"][:16] for r in results[:top_n]]
accs = [r.get("best_acc", 0) for r in results[:top_n]]
colors = [COL_GOLD if i == 0 else COL_SNN for i in range(top_n)]
bars = ax1.barh(range(top_n-1, -1, -1), accs[:top_n], color=colors,
                edgecolor="white", linewidth=0.3, height=0.6)
ax1.set_yticks(range(top_n-1, -1, -1))
ax1.set_yticklabels(tags, fontsize=8)
ax1.set_xlabel("Val Accuracy (%)")
ax1.set_title("Search Results (ranked)")
for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{acc:.1f}%", va="center", fontsize=8, fontweight="bold",
             color=COL_GOLD if i == 0 else "white")

# (0,1) Test Accuracy Comparison
ax2 = fig.add_subplot(gs[0, 1])
models = ["Original\nSNN", "Tuned\nSNN", "ANN"]
accs_bar = [73.1, snn_acc, ann_acc]
colors_bar = ["#555555", COL_SNN, COL_ANN]
bars2 = ax2.bar(models, accs_bar, color=colors_bar, width=0.5,
                edgecolor="white", linewidth=0.5)
ax2.set_title("Accuracy Progression")
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_ylim(0, 110)
for bar, val in zip(bars2, accs_bar):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=14,
             color=COL_GOLD if val == snn_acc else "white")

# (0,2) Per-digit accuracy comparison
ax3 = fig.add_subplot(gs[0, 2])
x_d = np.arange(10)
bw = 0.35
ax3.bar(x_d - bw/2, snn_per_digit, bw, color=COL_SNN, label="SNN (tuned)",
        edgecolor="white", linewidth=0.3)
ax3.bar(x_d + bw/2, ann_per_digit, bw, color=COL_ANN, label="ANN",
        edgecolor="white", linewidth=0.3)
ax3.set_title("Per-Digit Test Accuracy")
ax3.set_xlabel("Digit")
ax3.set_ylabel("Accuracy (%)")
ax3.set_xticks(x_d)
ax3.set_ylim(0, 115)
ax3.legend()

# (1,0) Confusion Matrix — SNN
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.imshow(snn_cm, cmap="Blues", aspect="equal")
ax4.set_title("Confusion Matrix — Tuned SNN")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("True")
ax4.set_xticks(range(10)); ax4.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        v = snn_cm[i, j]
        c = "white" if v > snn_cm.max() * 0.5 else COL_TEXT
        ax4.text(j, i, str(v), ha="center", va="center", fontsize=7,
                 color=c, fontweight="bold")
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

# (1,1) Confusion Matrix — ANN
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.imshow(ann_cm, cmap="Reds", aspect="equal")
ax5.set_title("Confusion Matrix — ANN")
ax5.set_xlabel("Predicted"); ax5.set_ylabel("True")
ax5.set_xticks(range(10)); ax5.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        v = ann_cm[i, j]
        c = "white" if v > ann_cm.max() * 0.5 else COL_TEXT
        ax5.text(j, i, str(v), ha="center", va="center", fontsize=7,
                 color=c, fontweight="bold")
fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# (1,2) Energy comparison with per-digit breakdown
ax6 = fig.add_subplot(gs[1, 2])
max_e = max(max(per_digit_energy), ann_energy_nj)
d_colors = [plt.cm.viridis(e / max_e) for e in per_digit_energy]
ax6.bar(x_d, per_digit_energy, color=d_colors, width=0.6,
        edgecolor="white", linewidth=0.3)
ax6.axhline(ann_energy_nj, color=COL_ANN, linestyle="--", linewidth=2,
            label=f"ANN (constant): {ann_energy_nj:.0f} nJ")
ax6.axhline(snn_energy_nj, color=COL_SNN, linestyle=":", linewidth=2,
            label=f"SNN (avg): {snn_energy_nj:.0f} nJ")
ax6.set_title("Energy Adapts to Input Sparsity")
ax6.set_xlabel("Digit")
ax6.set_ylabel("Energy (nJ)")
ax6.set_xticks(x_d)
ax6.legend(loc="upper right", fontsize=8)

fig1_path = os.path.join(OUT_DIR, "mnist_revamped_comparison.pdf")
plt.savefig(fig1_path, dpi=300, facecolor=COL_BG)
print(f"  Saved: {fig1_path}")


# ===== FIGURE 2: Spike Rasters for Best Config =====
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle("Tuned SNN — Reservoir Spike Rasters",
              fontsize=16, fontweight="bold", color="white", y=0.98)
gs2 = gridspec.GridSpec(2, 5, hspace=0.45, wspace=0.3,
                        left=0.05, right=0.97, top=0.90, bottom=0.08)

digit_examples = {}
for idx in range(len(final_test)):
    img, lbl = final_test[idx]
    if lbl not in digit_examples:
        digit_examples[lbl] = img
    if len(digit_examples) >= 10:
        break

for plot_idx in range(10):
    if plot_idx not in digit_examples:
        continue
    img = digit_examples[plot_idx]
    spike_train = poisson_encode(img.view(-1), bc["sim_time"], bc["input_scale"], device)
    _, all_spikes, _ = final_reservoir.run_with_raster(spike_train)

    raster = all_spikes.cpu().numpy()
    total_per_neuron = raster.sum(axis=0)
    top50 = np.argsort(total_per_neuron)[-50:]
    raster_top = raster[:, top50]

    ax = fig2.add_subplot(gs2[plot_idx // 5, plot_idx % 5])
    st, ni = np.where(raster_top.T)
    if len(st) > 0:
        ax.scatter(ni, st, s=0.8, color=COL_SNN, alpha=0.7, marker="|")
    ax.set_title(f"'{plot_idx}' ({int(total_per_neuron.sum())} spk)", fontsize=11)
    ax.set_xlim(0, bc["sim_time"])
    ax.set_ylim(-1, 50)
    if plot_idx % 5 == 0:
        ax.set_ylabel("Neuron #")
    if plot_idx >= 5:
        ax.set_xlabel("Time step")
    ax.tick_params(labelsize=8)

fig2_path = os.path.join(OUT_DIR, "mnist_revamped_rasters.pdf")
plt.savefig(fig2_path, dpi=300, facecolor=COL_BG)
print(f"  Saved: {fig2_path}")

plt.show()


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"""
  Original SNN accuracy:    73.1%
  Tuned SNN accuracy:       {snn_acc:.1f}%  (+{snn_acc - 73.1:.1f}%)
  ANN baseline accuracy:    {ann_acc:.1f}%
  Accuracy gap (SNN vs ANN): {ann_acc - snn_acc:.1f}%

  Best hyperparameters:
    Reservoir neurons:  {bc['n_reservoir']}
    Input connectivity: {bc['conn_in']}
    Recurrent connect.: {bc['conn_rec']}
    LIF decay:          {bc['decay']}
    LIF threshold:      {bc['thresh']}
    Simulation time:    {bc['sim_time']}
    Input scale:        {bc['input_scale']}
    Normalize features: {bc['normalize']}
    Readout type:       {bc['best_readout']}

  Energy efficiency:
    SNN: {snn_energy_nj:.1f} nJ/sample  |  ANN: {ann_energy_nj:.1f} nJ/sample
    SNN is ~{ann_energy_nj/max(snn_energy_nj,0.01):.1f}x more efficient

  Total search time: {t_search:.0f}s ({t_search/60:.1f} min)

  Figures saved to: {OUT_DIR}
""")
