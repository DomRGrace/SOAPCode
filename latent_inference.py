"""
latent_inference.py

Latent world-model surprise — an ablation alternative to the proprioceptive
ProprioHead signal (see inference.py). Instead of predicting the observed
proprio delta, LatentHead predicts the VLA's own next action-hidden-state
h_{t+1} (projected into a low-dim subspace) from (h_t, a_t). Its prediction
error drives A_mem writes the same way proprio surprise does, isolating
whether "surprise" needs to be grounded in proprioception specifically, or
whether any predictive-coding error over the model's own latent trajectory
works just as well.

IMPORTANT: h_t and a_t are concatenated and L2-normalised before use, exactly
           as in inference.py. h_{t+1} is projected through the frozen random
           matrix P saved in the checkpoint, then standardised with hn_mean/
           hn_std. Skipping these will produce meaningless surprise values.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
MLP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "soap_latent_head.pt")

H_DIM      = 4096
A_DIM      = 56
IN_DIM     = H_DIM + A_DIM   # 4152
LATENT_DIM = 64              # dim of the projected h_{t+1} prediction target
HIDDEN_MID = 1024
HIDDEN_BOT = 512
DROPOUT    = 0.4
HEAD_DROPOUT = 0.5


class LatentHead(nn.Module):
    """Same trunk shape as ProprioHead; predicts a projected next-hidden-state
    instead of a proprio delta."""

    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN_MID),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_MID, HIDDEN_BOT),
            nn.LayerNorm(HIDDEN_BOT),
            nn.GELU(),
            nn.Dropout(HEAD_DROPOUT),
        )

        self.latent_head = nn.Linear(HIDDEN_BOT, LATENT_DIM)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, ha):
        return self.latent_head(self.trunk(ha))


def make_projection(d=H_DIM, r=LATENT_DIM, seed=None):
    """Frozen random projection H_DIM -> LATENT_DIM, columns L2-normalised
    (same construction as MemoryLoRA.B_mem in adapt.py)."""
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(d, r, generator=g)
    else:
        raw = torch.randn(d, r)
    return F.normalize(raw, p=2, dim=0)


def load_latent_model(path: str = MLP_PATH):
    ckpt    = torch.load(path, map_location=DEVICE)
    model   = LatentHead().to(DEVICE)
    state   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    proj    = ckpt["proj"].to(DEVICE).float()
    hn_mean = ckpt["hn_mean"].to(DEVICE)
    hn_std  = ckpt["hn_std"].to(DEVICE)
    return model, proj, hn_mean, hn_std


def compute_latent_surprise(model, proj, hn_mean, hn_std, h_t, a_t, h_next):
    """
    Compute latent world-model surprise at timestep t.

    Args:
        model:   loaded LatentHead
        proj:    (H_DIM, LATENT_DIM) frozen projection matrix
        hn_mean: (LATENT_DIM,) normalisation mean from training
        hn_std:  (LATENT_DIM,) normalisation std from training
        h_t:     raw action hidden state at t,       shape (D,)
        a_t:     raw action chunk at t,               shape (56,)
        h_next:  raw action hidden state at t+1,      shape (D,)

    Returns:
        surprise: MSE between predicted and actual projected next hidden state
    """
    h_t    = h_t.to(DEVICE).float()
    a_t    = a_t.to(DEVICE).float()
    h_next = h_next.to(DEVICE).float()

    if h_t.dim() == 1:    h_t    = h_t.unsqueeze(0)
    if a_t.dim() == 1:    a_t    = a_t.unsqueeze(0)
    if h_next.dim() == 1: h_next = h_next.unsqueeze(0)

    ha_norm = F.normalize(torch.cat([h_t, a_t], dim=-1), p=2, dim=-1)
    hn_norm = ((h_next @ proj) - hn_mean) / hn_std

    with torch.no_grad():
        hn_pred = model(ha_norm)

    return F.mse_loss(hn_pred, hn_norm).item()
