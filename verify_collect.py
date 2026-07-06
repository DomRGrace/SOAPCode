"""
verify_collect.py

Runs datacollect on a single demo from one task to verify
shapes and dtypes before committing to the full collection run.
"""

import sys
import torch
import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, ".")

from experiments.robot.openvla_utils import (
    get_vla, get_action_head, get_proprio_projector,
    get_processor, normalize_proprio, prepare_images_for_vla,
)
from datacollect import get_proprio, get_images, build_processor_inputs, Cfg

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
HDF5   = "./datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
PROMPT = "In: What action should the robot take to turn on the stove and put the moka pot on it?\nOut:"


def main():
    cfg = Cfg()

    print("Loading VLA...")
    vla = get_vla(cfg); vla.eval()
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, vla.llm_dim)
    processor         = get_processor(cfg)

    with h5py.File(HDF5, "r") as f:
        demo_key = sorted(f["data"].keys())[0]
        obs = f["data"][demo_key]["obs"]
        print(f"Demo: {demo_key}  |  T={obs['agentview_rgb'].shape[0]}")

        t = 0

        # ---- h_t ----
        img_t, wrist_t = get_images(obs, t)
        s_t_raw = get_proprio(obs, t)
        pil_t   = prepare_images_for_vla([img_t, wrist_t], cfg)
        inp_t   = build_processor_inputs(processor, PROMPT, pil_t, cfg)

        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        s_t_norm = normalize_proprio(s_t_raw, proprio_norm_stats)
        proprio_t = torch.tensor(s_t_norm, dtype=torch.bfloat16).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, actions_hidden = vla.predict_action(
                input_ids         = inp_t["input_ids"],
                unnorm_key        = cfg.unnorm_key,
                proprio           = proprio_t,
                proprio_projector = proprio_projector,
                action_head       = action_head,
                noisy_action_projector = None,
                use_film          = False,
                pixel_values      = inp_t["pixel_values"],
                attention_mask    = inp_t["attention_mask"],
            )
        h_t = actions_hidden.mean(dim=1).squeeze(0).float().cpu()
        print(f"h_t  : shape={tuple(h_t.shape)}  dtype={h_t.dtype}  norm={h_t.norm():.3f}")

        # ---- z_{t+1} ----
        img_t1, wrist_t1 = get_images(obs, t + 1)
        pil_t1 = prepare_images_for_vla([img_t1, wrist_t1], cfg)
        inp_t1 = build_processor_inputs(processor, PROMPT, pil_t1, cfg)

        with torch.no_grad():
            z_t1 = vla._process_vision_features(inp_t1["pixel_values"], language_embeddings=None, use_film=False)
            z_t1 = z_t1.mean(dim=1).squeeze(0).float().cpu()
        print(f"z_t1 : shape={tuple(z_t1.shape)}  dtype={z_t1.dtype}  norm={z_t1.norm():.3f}")

        # ---- s_{t+1} ----
        s_t1 = torch.tensor(get_proprio(obs, t + 1), dtype=torch.float32)
        print(f"s_t1 : shape={tuple(s_t1.shape)}  dtype={s_t1.dtype}  values={s_t1.numpy().round(4)}")

    print("\nAll shapes look good." if h_t.shape == z_t1.shape else "\nWARNING: h and z dims differ!")


if __name__ == "__main__":
    main()
