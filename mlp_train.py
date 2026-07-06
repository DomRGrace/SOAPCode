"""
mlp_train.py

Trains the ProprioHead surprise MLP on rollout pairs.

Contrastive loss:
  success pairs: minimise MSE(ds_pred, ds_actual)
  failure pairs: hinge — push MSE above MARGIN

Run from openvla-oft/:
  python mlp_train.py

Saves: ./datasets/soap_mlp_v4.pt
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from inference import ProprioHead

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEMO_PATH    = "./datasets/soap_pairs_v3.pt"
ROLLOUT_S    = "./datasets/rollout_success_pairs.pt"
ROLLOUT_F    = "./datasets/rollout_failure_pairs.pt"
SAVE_PATH    = "./datasets/soap_mlp_v4.pt"

H_DIM       = 4096
A_DIM       = 56
DS_DIM      = 8

MARGIN          = 2.0
SUCCESS_WEIGHT  = 1.0
FAILURE_WEIGHT  = 3.0

LR              = 5e-5
WEIGHT_DECAY    = 0.2
BATCH_SIZE      = 512
EPOCHS          = 100
VAL_FRAC        = 0.3
DEVICE          = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, h_s, a_s, ds_s, h_f, a_f, ds_f, ds_mean, ds_std):
        ha_s = F.normalize(torch.cat([h_s, a_s], dim=-1), p=2, dim=-1)
        ha_f = F.normalize(torch.cat([h_f, a_f], dim=-1), p=2, dim=-1)
        ds_s = (ds_s - ds_mean) / ds_std
        ds_f = (ds_f - ds_mean) / ds_std

        self.ha     = torch.cat([ha_s, ha_f])
        self.ds     = torch.cat([ds_s, ds_f])
        self.labels = torch.cat([torch.zeros(len(ha_s)), torch.ones(len(ha_f))])  # 0=success, 1=failure

    def __len__(self): return len(self.ha)
    def __getitem__(self, idx): return self.ha[idx], self.ds[idx], self.labels[idx]


def episode_split(n_pairs, val_frac, pairs_per_episode=271):
    n_episodes = math.ceil(n_pairs / pairs_per_episode)
    n_val_eps  = max(1, int(n_episodes * val_frac))
    val_eps    = set(range(n_episodes - n_val_eps, n_episodes))
    train_idx, val_idx = [], []
    for i in range(n_pairs):
        ep = i // pairs_per_episode
        (val_idx if ep in val_eps else train_idx).append(i)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def contrastive_loss(pred, actual, labels):
    per_sample_mse = F.mse_loss(pred, actual, reduction="none").mean(dim=-1)

    s_mask = (labels == 0)
    f_mask = (labels == 1)

    loss_s = per_sample_mse[s_mask].mean() if s_mask.any() else torch.tensor(0.0, device=pred.device)
    loss_f = F.relu(MARGIN - per_sample_mse[f_mask]).mean() if f_mask.any() else torch.tensor(0.0, device=pred.device)

    return SUCCESS_WEIGHT * loss_s + FAILURE_WEIGHT * loss_f


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_mse_s, total_mse_f, n_batches = 0.0, 0.0, 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for ha, ds, labels in loader:
            ha, ds, labels = ha.to(DEVICE), ds.to(DEVICE), labels.to(DEVICE)
            pred = model(ha)
            loss = contrastive_loss(pred, ds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            with torch.no_grad():
                per_sample = F.mse_loss(pred, ds, reduction="none").mean(dim=-1)
                s_mask = (labels == 0)
                f_mask = (labels == 1)
                total_mse_s += per_sample[s_mask].mean().item() if s_mask.any() else 0.0
                total_mse_f += per_sample[f_mask].mean().item() if f_mask.any() else 0.0

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches, total_mse_s / n_batches, total_mse_f / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    demo = torch.load(DEMO_PATH)
    H_s, A_s, DS_s = demo["h"], demo["a"], demo["ds"]
    print(f"  Demo success pairs : {H_s.shape[0]:,}")

    if os.path.exists(ROLLOUT_S):
        rs   = torch.load(ROLLOUT_S)
        H_s  = torch.cat([H_s,  rs["h"]])
        A_s  = torch.cat([A_s,  rs["a"]])
        DS_s = torch.cat([DS_s, rs["ds"]])
        print(f"  + Rollout success  : {rs['h'].shape[0]:,}  → total success: {H_s.shape[0]:,}")

    rf = torch.load(ROLLOUT_F)
    H_f, A_f, DS_f = rf["h"], rf["a"], rf["ds"]
    print(f"  Rollout failure pairs: {H_f.shape[0]:,}")

    train_idx, val_idx = episode_split(len(H_s), VAL_FRAC)
    DS_tr   = DS_s[train_idx]
    ds_mean = DS_tr.mean(0)
    ds_std  = DS_tr.std(0).clamp(min=1e-6)

    n_f_val     = max(1, int(len(H_f) * VAL_FRAC))
    f_train_idx = list(range(len(H_f) - n_f_val))
    f_val_idx   = list(range(len(H_f) - n_f_val, len(H_f)))

    def make_dataset(s_idx, f_idx):
        return ContrastiveDataset(
            H_s[s_idx], A_s[s_idx], DS_s[s_idx],
            H_f[f_idx], A_f[f_idx], DS_f[f_idx],
            ds_mean, ds_std,
        )

    train_ds = make_dataset(train_idx, f_train_idx)
    val_ds   = make_dataset(val_idx,   f_val_idx)

    n_s = len(train_idx)
    n_f = len(f_train_idx)
    weights = [1.0] * n_s + [n_s / max(n_f, 1) * (FAILURE_WEIGHT / SUCCESS_WEIGHT)] * n_f
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)

    model     = ProprioHead().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining on {DEVICE}  |  {n_params:,} params")
    print(f"Train: {len(train_ds):,} pairs ({n_s} success + {n_f} failure)")
    print(f"Val  : {len(val_ds):,} pairs")
    print(f"Trivial baseline MSE: 1.0  |  Failure hinge margin: {MARGIN}\n")

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mse_s, tr_mse_f = run_epoch(model, train_loader, optimizer)
        va_loss, va_mse_s, va_mse_f = run_epoch(model, val_loader)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}"
                f"  train={tr_loss:.4f} (s={tr_mse_s:.3f} f={tr_mse_f:.3f})"
                f"  val={va_loss:.4f} (s={va_mse_s:.3f} f={va_mse_f:.3f})"
            )

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model":      model.state_dict(),
                "ds_mean":    ds_mean,
                "ds_std":     ds_std,
                "input_norm": "l2_concat_h_a",
            }, SAVE_PATH)
            print(f"  --> Saved best (val={best_val:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Model saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
