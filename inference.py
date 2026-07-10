"""
inference.py

SOAP surprise detection using the proprio-only MLP.

IMPORTANT: h_t and a_t are concatenated and L2-normalised before use.
           ds (proprio delta) is standardized using saved ds_mean/ds_std.
           Skipping these will produce meaningless surprise values.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE   = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
MLP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "soap_mlp_v3.pt")

H_DIM      = 4096
A_DIM      = 56
IN_DIM     = H_DIM + A_DIM   # 4152
DS_DIM     = 8
HIDDEN_MID = 1024
HIDDEN_BOT = 512
DROPOUT    = 0.4
DS_HEAD_DROPOUT = 0.5


class ProprioHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN_MID),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_MID, HIDDEN_BOT),
            nn.LayerNorm(HIDDEN_BOT),
            nn.GELU(),
            nn.Dropout(DS_HEAD_DROPOUT),
        )

        self.ds_head = nn.Linear(HIDDEN_BOT, DS_DIM)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, ha):
        return self.ds_head(self.trunk(ha))


def load_model(path: str = MLP_PATH):
    ckpt    = torch.load(path, map_location=DEVICE)
    model   = ProprioHead().to(DEVICE)
    state   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    ds_mean = ckpt["ds_mean"].to(DEVICE)
    ds_std  = ckpt["ds_std"].to(DEVICE)
    return model, ds_mean, ds_std


def compute_surprise(model, ds_mean, ds_std, h_t, a_t, ds_t1):
    """
    Compute proprio surprise at timestep t.

    Args:
        model:    loaded ProprioHead
        ds_mean:  (8,) normalisation mean from training
        ds_std:   (8,) normalisation std from training
        h_t:      raw action hidden state at t,      shape (D,)
        a_t:      raw action chunk at t,             shape (56,)
        ds_t1:    actual proprio delta s_{t+1}-s_t,  shape (8,)

    Returns:
        surprise: MSE between predicted and actual normalised delta
    """
    h_t   = h_t.to(DEVICE).float()
    a_t   = a_t.to(DEVICE).float()
    ds_t1 = ds_t1.to(DEVICE).float()

    if h_t.dim() == 1:   h_t   = h_t.unsqueeze(0)
    if a_t.dim() == 1:   a_t   = a_t.unsqueeze(0)
    if ds_t1.dim() == 1: ds_t1 = ds_t1.unsqueeze(0)

    ha_norm = F.normalize(torch.cat([h_t, a_t], dim=-1), p=2, dim=-1)
    ds_norm = (ds_t1 - ds_mean) / ds_std

    with torch.no_grad():
        ds_pred = model(ha_norm)

    return F.mse_loss(ds_pred, ds_norm).item()
