"""
latent_train.py

Trains the LatentHead world-model surprise head on rollout pairs collected by
collect_rollout_pairs.py (requires the "hn" field — see that script's h_{t+1}
collection). This is the ablation counterpart to mlp_train.py: instead of
predicting the observed proprio delta, LatentHead predicts a frozen random
projection of the VLA's own next action-hidden-state h_{t+1} from (h_t, a_t).

Contrastive loss (same scheme as mlp_train.py):
  success pairs: minimise MSE(hn_pred, hn_actual)
  failure pairs: hinge — push MSE above MARGIN

Unlike mlp_train.py, there is no demo-data source for h_{t+1} (soap_pairs_v3.pt
was collected without it), so this trains on rollout pairs only. Rollout pairs
carry real per-episode ids ("ep"), so the train/val split is done at the
episode level directly rather than via the fixed-pairs-per-episode heuristic
in mlp_train.py.

Run from openvla-oft/ (with soap_implementation/ on PYTHONPATH):
  python latent_train.py

Saves: ./datasets/soap_latent_head_v1.pt
Promote the best run to soap_implementation/checkpoints/soap_latent_head.pt
to make the "latent" mode available in adapt.py.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from latent_inference import LatentHead, make_projection, LATENT_DIM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROLLOUT_S    = "./datasets/rollout_success_pairs.pt"
ROLLOUT_F    = "./datasets/rollout_failure_pairs.pt"
SAVE_PATH    = "./datasets/soap_latent_head_v1.pt"

H_DIM       = 4096
A_DIM       = 56

PROJ_SEED       = 0     # fixed seed so train/eval/adapt.py all share the same projection
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
class ContrastiveLatentDataset(Dataset):
    def __init__(self, h_s, a_s, hn_s, h_f, a_f, hn_f, proj, hn_mean, hn_std):
        ha_s = F.normalize(torch.cat([h_s, a_s], dim=-1), p=2, dim=-1)
        ha_f = F.normalize(torch.cat([h_f, a_f], dim=-1), p=2, dim=-1)
        tgt_s = ((hn_s @ proj) - hn_mean) / hn_std
        tgt_f = ((hn_f @ proj) - hn_mean) / hn_std

        self.ha     = torch.cat([ha_s, ha_f])
        self.tgt    = torch.cat([tgt_s, tgt_f])
        self.labels = torch.cat([torch.zeros(len(ha_s)), torch.ones(len(ha_f))])  # 0=success, 1=failure

    def __len__(self): return len(self.ha)
    def __getitem__(self, idx): return self.ha[idx], self.tgt[idx], self.labels[idx]


def episode_split(ep_ids: torch.Tensor, val_frac: float):
    """Split indices by unique episode id so no episode straddles train/val."""
    unique_eps = sorted(set(ep_ids.tolist()))
    n_val_eps  = max(1, int(len(unique_eps) * val_frac))
    val_eps    = set(unique_eps[len(unique_eps) - n_val_eps:])
    train_idx, val_idx = [], []
    for i, ep in enumerate(ep_ids.tolist()):
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
        for ha, tgt, labels in loader:
            ha, tgt, labels = ha.to(DEVICE), tgt.to(DEVICE), labels.to(DEVICE)
            pred = model(ha)
            loss = contrastive_loss(pred, tgt, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            with torch.no_grad():
                per_sample = F.mse_loss(pred, tgt, reduction="none").mean(dim=-1)
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
    if not os.path.exists(ROLLOUT_S) or not os.path.exists(ROLLOUT_F):
        raise FileNotFoundError(
            "Rollout pairs not found. Run collect_rollout_pairs.py first "
            "(latent_train.py requires the 'hn' field it collects)."
        )

    rs = torch.load(ROLLOUT_S)
    rf = torch.load(ROLLOUT_F)
    H_s, A_s, HN_s, EP_s = rs["h"], rs["a"], rs["hn"], rs["ep"]
    H_f, A_f, HN_f, EP_f = rf["h"], rf["a"], rf["hn"], rf["ep"]
    print(f"  Rollout success pairs: {H_s.shape[0]:,}")
    print(f"  Rollout failure pairs: {H_f.shape[0]:,}")

    if H_s.shape[0] == 0 or H_f.shape[0] == 0:
        raise ValueError("Empty success or failure pairs — nothing to train on.")

    proj = make_projection(d=H_DIM, r=LATENT_DIM, seed=PROJ_SEED)

    s_train_idx, s_val_idx = episode_split(EP_s, VAL_FRAC)
    f_train_idx, f_val_idx = episode_split(EP_f, VAL_FRAC)

    hn_tr   = HN_s[s_train_idx] @ proj
    hn_mean = hn_tr.mean(0)
    hn_std  = hn_tr.std(0).clamp(min=1e-6)

    def make_dataset(s_idx, f_idx):
        return ContrastiveLatentDataset(
            H_s[s_idx], A_s[s_idx], HN_s[s_idx],
            H_f[f_idx], A_f[f_idx], HN_f[f_idx],
            proj, hn_mean, hn_std,
        )

    train_ds = make_dataset(s_train_idx, f_train_idx)
    val_ds   = make_dataset(s_val_idx,   f_val_idx)

    n_s = len(s_train_idx)
    n_f = len(f_train_idx)
    weights = [1.0] * n_s + [n_s / max(n_f, 1) * (FAILURE_WEIGHT / SUCCESS_WEIGHT)] * n_f
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=True)

    model     = LatentHead().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining on {DEVICE}  |  {n_params:,} params  |  latent_dim={LATENT_DIM}")
    print(f"Train: {len(train_ds):,} pairs ({n_s} success + {n_f} failure)")
    print(f"Val  : {len(val_ds):,} pairs")
    print(f"Failure hinge margin: {MARGIN}\n")

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
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save({
                "model":      model.state_dict(),
                "proj":       proj,
                "hn_mean":    hn_mean,
                "hn_std":     hn_std,
                "proj_seed":  PROJ_SEED,
                "latent_dim": LATENT_DIM,
                "input_norm": "l2_concat_h_a",
            }, SAVE_PATH)
            print(f"  --> Saved best (val={best_val:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Model saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
