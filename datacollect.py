"""
datacollect.py

Collects (h_t, a_t, Δs_{t+1}, z_{t+1}) training pairs from LIBERO-10 demo data
for training the ProprioHead (see inference.py / mlp_train.py). z_{t+1} (projected
visual embedding) is collected alongside but unused by the current pipeline — an
earlier design predicted both proprio delta and next-frame visual embedding jointly,
but the visual prediction target underperformed and was dropped in favor of a
proprio-only head.

  h_t      : mean-pooled action hidden states from the frozen VLA at timestep t  (D,)
  a_t      : action chunk executed at timestep t, flattened                       (56,)
  delta_s  : exact proprio delta s_{t+1} - s_t                                   (8,)
  z_t1     : mean-pooled projected visual embedding at timestep t+1               (D,)

Run from the openvla-oft directory:
  python datacollect.py
"""

import os
import sys
import numpy as np
import torch
import h5py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

sys.path.insert(0, ".")

from experiments.robot.openvla_utils import (
    get_vla,
    get_action_head,
    get_proprio_projector,
    get_processor,
    normalize_proprio,
    prepare_images_for_vla,
)
from libero.libero import benchmark

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR    = "./datasets/libero_10"
OUTPUT_PATH    = "./datasets/soap_pairs_v2.pt"
CHECKPOINT_DIR = "./datasets/soap_checkpoints_v2"
CHECKPOINT     = "moojink/openvla-7b-oft-finetuned-libero-10"
DEVICE         = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Cfg:
    pretrained_checkpoint: str          = CHECKPOINT
    load_in_8bit: bool                  = False
    load_in_4bit: bool                  = True
    use_film: bool                      = False
    use_l1_regression: bool             = True
    use_diffusion: bool                 = False
    num_images_in_input: int            = 2
    use_proprio: bool                   = True
    center_crop: bool                   = True
    lora_rank: int                      = 32
    unnorm_key: str                     = "libero_10_no_noops"
    num_diffusion_steps_train: int      = 50
    num_diffusion_steps_inference: int  = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_language(hdf5_path: Path, task_suite) -> str:
    """Match HDF5 filename to LIBERO task language instruction."""
    stem = hdf5_path.stem.replace("_demo", "")
    for i in range(task_suite.n_tasks):
        task = task_suite.get_task(i)
        if task.name.lower() == stem.lower():
            return task.language
    # Fallback: strip SCENE prefix and convert underscores
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.upper().startswith("SCENE"):
            return " ".join(parts[i + 1:])
    return stem.replace("_", " ")


def get_proprio(obs_group, t: int) -> np.ndarray:
    """Return 8-dim proprio state at timestep t.
    ee_states is already (pos, axis-angle) = 6-dim; gripper_states is 2-dim."""
    ee     = obs_group["ee_states"][t]        # (6,) pos + axis-angle
    gripper = obs_group["gripper_states"][t]  # (2,)
    return np.concatenate([ee, gripper])


def get_images(obs_group, t: int):
    """Return (agentview, wrist) uint8 numpy images at timestep t."""
    return obs_group["agentview_rgb"][t], obs_group["eye_in_hand_rgb"][t]


def build_processor_inputs(processor, prompt: str, pil_imgs, cfg: Cfg):
    """Build input dict for a single observation with optional wrist image."""
    primary, *wrists = pil_imgs
    inputs = processor(prompt, primary).to(DEVICE, dtype=torch.bfloat16)
    if wrists:
        for w in wrists:
            w_in = processor(prompt, w).to(DEVICE, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], w_in["pixel_values"]], dim=1
            )
    return inputs


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(vla, proprio_projector, action_head, processor, task_suite, cfg: Cfg):
    all_h, all_a, all_ds, all_z = [], [], [], []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hdf5_files = sorted(Path(DATASET_DIR).glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} task files\n")

    for hdf5_path in hdf5_files:
        ckpt_path = Path(CHECKPOINT_DIR) / (hdf5_path.stem + ".pt")

        # Resume: skip tasks already completed
        if ckpt_path.exists():
            print(f"Resuming: loading checkpoint {ckpt_path.name}")
            ckpt = torch.load(ckpt_path)
            all_h.append(ckpt["h"])
            all_a.append(ckpt["a"])
            all_ds.append(ckpt["ds"])
            all_z.append(ckpt["z"])
            print()
            continue

        language = get_language(hdf5_path, task_suite)
        prompt   = f"In: What action should the robot take to {language.lower()}?\nOut:"
        print(f"Task : {language}")

        task_h, task_a, task_ds, task_z = [], [], [], []

        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys())
            for demo_key in tqdm(demos, desc=hdf5_path.stem[:40]):
                obs = f["data"][demo_key]["obs"]
                T   = obs["agentview_rgb"].shape[0]

                for t in range(T - 1):
                    # ---- h_t, a_t : full forward at timestep t -----------
                    img_t, wrist_t = get_images(obs, t)
                    s_t_raw = get_proprio(obs, t)

                    pil_imgs_t = prepare_images_for_vla([img_t, wrist_t], cfg)
                    inputs_t   = build_processor_inputs(processor, prompt, pil_imgs_t, cfg)

                    proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
                    s_t_norm = normalize_proprio(s_t_raw, proprio_norm_stats)
                    proprio_t = torch.tensor(s_t_norm, dtype=torch.bfloat16).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        actions, actions_hidden = vla.predict_action(
                            input_ids         = inputs_t["input_ids"],
                            unnorm_key        = cfg.unnorm_key,
                            proprio           = proprio_t,
                            proprio_projector = proprio_projector,
                            action_head       = action_head,
                            noisy_action_projector = None,
                            use_film          = False,
                            pixel_values      = inputs_t["pixel_values"],
                            attention_mask    = inputs_t["attention_mask"],
                        )
                    # actions_hidden: (1, chunk, D) → mean-pool → (D,)
                    h_t = actions_hidden.mean(dim=1).squeeze(0).float().cpu()
                    # actions: (chunk, 7) → flatten → (56,)
                    a_t = torch.tensor(actions, dtype=torch.float32).flatten()

                    # ---- z_{t+1} : vision backbone at timestep t+1 -------
                    img_t1, wrist_t1 = get_images(obs, t + 1)
                    pil_imgs_t1 = prepare_images_for_vla([img_t1, wrist_t1], cfg)
                    inputs_t1   = build_processor_inputs(processor, prompt, pil_imgs_t1, cfg)

                    with torch.no_grad():
                        z_t1 = vla._process_vision_features(inputs_t1["pixel_values"], language_embeddings=None, use_film=False)
                        z_t1 = z_t1.mean(dim=1).squeeze(0).float().cpu()

                    # ---- Δs_{t+1} : exact delta from HDF5 ----------------
                    s_t1_raw = get_proprio(obs, t + 1)
                    delta_s  = torch.tensor(s_t1_raw - s_t_raw, dtype=torch.float32)

                    task_h.append(h_t)
                    task_a.append(a_t)
                    task_ds.append(delta_s)
                    task_z.append(z_t1)

        # Save per-task checkpoint
        task_H  = torch.stack(task_h)
        task_A  = torch.stack(task_a)
        task_DS = torch.stack(task_ds)
        task_Z  = torch.stack(task_z)
        torch.save({"h": task_H, "a": task_A, "ds": task_DS, "z": task_Z}, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path.name}  ({task_H.shape[0]:,} pairs)\n")

        all_h.append(task_H)
        all_a.append(task_A)
        all_ds.append(task_DS)
        all_z.append(task_Z)

    return torch.cat(all_h), torch.cat(all_a), torch.cat(all_ds), torch.cat(all_z)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = Cfg()

    # Load task suite for language lookup
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict["libero_10"]()

    # Load frozen model components
    print("Loading VLA...")
    vla = get_vla(cfg)
    vla.eval()

    print("Loading proprio projector...")
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)

    print("Loading action head...")
    action_head = get_action_head(cfg, vla.llm_dim)

    print("Loading processor...")
    processor = get_processor(cfg)

    # Collect pairs
    print("\nCollecting training pairs...\n")
    H, A, DS, Z = collect(vla, proprio_projector, action_head, processor, task_suite, cfg)

    print(f"Collected {H.shape[0]:,} training pairs")
    print(f"  h  : {tuple(H.shape)}")
    print(f"  a  : {tuple(A.shape)}")
    print(f"  ds : {tuple(DS.shape)}")
    print(f"  z  : {tuple(Z.shape)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save({"h": H, "a": A, "ds": DS, "z": Z}, OUTPUT_PATH)
    print(f"\nSaved → {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
