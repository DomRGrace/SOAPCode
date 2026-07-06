"""
collect_rollout_pairs.py

Runs live LIBERO-10 policy rollouts and collects (h_t, a_t, Δs_K) pairs
labelled by episode outcome. Δs_K = s_{t+K} - s_t, matching test-time computation.

Saves:
  ./datasets/rollout_success_pairs.pt
  ./datasets/rollout_failure_pairs.pt

Run from openvla-oft/:
  python collect_rollout_pairs.py
"""

import os
import sys
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_vla,
    get_action_head,
    get_proprio_projector,
    get_processor,
    normalize_proprio,
    prepare_images_for_vla,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    normalize_gripper_action,
    invert_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-10"
CKPT_DIR   = "./datasets/rollout_checkpoints"
OUT_DIR    = "./datasets"

TASK_NAME      = "libero_10"
NUM_TASKS      = 10
NUM_EPISODES   = 20      # episodes per task
NUM_STEPS_WAIT = 10
MAX_STEPS      = 520
UNNORM_KEY     = "libero_10_no_noops"
IMG_RESIZE     = 224


@dataclass
class Cfg:
    pretrained_checkpoint: str         = CHECKPOINT
    load_in_8bit: bool                 = False
    load_in_4bit: bool                 = True
    use_film: bool                     = False
    use_l1_regression: bool            = True
    use_diffusion: bool                = False
    num_images_in_input: int           = 2
    use_proprio: bool                  = True
    center_crop: bool                  = True
    lora_rank: int                     = 32
    unnorm_key: str                    = UNNORM_KEY
    num_diffusion_steps_train: int     = 50
    num_diffusion_steps_inference: int = 50
    model_family: str                  = "openvla"
    num_open_loop_steps: int           = NUM_ACTIONS_CHUNK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_vla(vla, processor, cfg, obs, task_label, proprio_projector, action_head):
    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.append(obs["wrist_image"])

    pil_imgs = prepare_images_for_vla(all_images, cfg)
    primary  = pil_imgs.pop(0)
    prompt   = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs   = processor(prompt, primary).to(DEVICE, dtype=torch.bfloat16)

    for w in pil_imgs:
        w_in = processor(prompt, w).to(DEVICE, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], w_in["pixel_values"]], dim=1)

    proprio = None
    if cfg.use_proprio:
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        proprio_normed = normalize_proprio(obs["state"], proprio_norm_stats)
        proprio = torch.tensor(proprio_normed, dtype=torch.bfloat16).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        actions, hidden = vla.predict_action(
            input_ids              = inputs["input_ids"],
            unnorm_key             = cfg.unnorm_key,
            proprio                = proprio,
            proprio_projector      = proprio_projector,
            action_head            = action_head,
            noisy_action_projector = None,
            use_film               = False,
            pixel_values           = inputs["pixel_values"],
            attention_mask         = inputs["attention_mask"],
        )

    h_t = hidden.mean(dim=1).squeeze(0).float().cpu()
    return actions, h_t


def make_obs(raw_obs):
    return {
        "full_image":  resize_image_for_policy(get_libero_image(raw_obs), IMG_RESIZE),
        "wrist_image": resize_image_for_policy(get_libero_wrist_image(raw_obs), IMG_RESIZE),
        "state": np.concatenate([
            raw_obs["robot0_eef_pos"],
            quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]),
    }


def get_proprio(raw_obs):
    return np.concatenate([
        raw_obs["robot0_eef_pos"],
        quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"],
    ])


# ---------------------------------------------------------------------------
# Single episode — returns list of (h_t, a_t, ds_K) and success bool
# ---------------------------------------------------------------------------

def run_episode(vla, processor, cfg, env, task_label, proprio_projector, action_head, initial_state):
    env.reset()
    raw_obs = env.set_init_state(initial_state)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t       = 0
    success = False
    pairs   = []   # list of (h_t, a_t, ds_K) tensors

    prev_h = None
    prev_a = None
    prev_s = None

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            raw_obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        obs   = make_obs(raw_obs)
        s_now = get_proprio(raw_obs)

        if len(action_queue) == 0:
            actions, h_t = query_vla(vla, processor, cfg, obs, task_label, proprio_projector, action_head)
            action_queue.extend(actions)
            a_t = torch.tensor(actions, dtype=torch.float32).flatten()

            if prev_h is not None:
                ds_k = torch.tensor(s_now - prev_s, dtype=torch.float32)
                pairs.append((prev_h, prev_a, ds_k))

            prev_h = h_t
            prev_a = a_t
            prev_s = s_now

        action = action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        raw_obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed_everywhere(7)
    os.makedirs(CKPT_DIR, exist_ok=True)

    cfg = Cfg()

    print("Loading VLA...")
    vla = get_vla(cfg); vla.eval()
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, vla.llm_dim)
    processor         = get_processor(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_NAME]()

    all_success_h, all_success_a, all_success_ds, all_success_ep = [], [], [], []
    all_failure_h, all_failure_a, all_failure_ds, all_failure_ep = [], [], [], []

    for task_idx in range(NUM_TASKS):
        ckpt_path = Path(CKPT_DIR) / f"task{task_idx}.pt"

        if ckpt_path.exists():
            print(f"Task {task_idx}: loading checkpoint")
            ckpt = torch.load(ckpt_path)
            all_success_h.append(ckpt["success_h"])
            all_success_a.append(ckpt["success_a"])
            all_success_ds.append(ckpt["success_ds"])
            all_failure_h.append(ckpt["failure_h"])
            all_failure_a.append(ckpt["failure_a"])
            all_failure_ds.append(ckpt["failure_ds"])
            # Backwards-compatible: old checkpoints without ep ids get dummy ids
            n_s = ckpt["success_h"].shape[0]
            n_f = ckpt["failure_h"].shape[0]
            all_success_ep.append(ckpt.get("success_ep", torch.full((n_s,), -1, dtype=torch.long)))
            all_failure_ep.append(ckpt.get("failure_ep", torch.full((n_f,), -1, dtype=torch.long)))
            print(f"  {n_s} success pairs, {n_f} failure pairs\n")
            continue

        task          = task_suite.get_task(task_idx)
        initial_states = task_suite.get_task_init_states(task_idx)
        env, task_label = get_libero_env(task, "openvla", resolution=256)
        print(f"Task {task_idx}: {task_label}")

        task_sh, task_sa, task_sds, task_sep = [], [], [], []
        task_fh, task_fa, task_fds, task_fep = [], [], [], []
        n_success, n_fail = 0, 0

        for ep in range(NUM_EPISODES):
            ep_id = task_idx * NUM_EPISODES + ep

            success, pairs = run_episode(
                vla, processor, cfg, env, task_label,
                proprio_projector, action_head, initial_states[ep],
            )
            outcome = "SUCCESS" if success else "FAIL"
            print(f"  ep {ep + 1:2d}/{NUM_EPISODES}  {outcome}  ({len(pairs)} pairs)")

            if not pairs:
                continue

            h_ep  = torch.stack([p[0] for p in pairs])
            a_ep  = torch.stack([p[1] for p in pairs])
            ds_ep = torch.stack([p[2] for p in pairs])
            id_ep = torch.full((len(pairs),), ep_id, dtype=torch.long)

            if success:
                task_sh.append(h_ep); task_sa.append(a_ep); task_sds.append(ds_ep); task_sep.append(id_ep)
                n_success += len(pairs)
            else:
                task_fh.append(h_ep); task_fa.append(a_ep); task_fds.append(ds_ep); task_fep.append(id_ep)
                n_fail += len(pairs)

        def cat_or_empty_2d(lst, dim):
            return torch.cat(lst) if lst else torch.zeros(0, dim)

        def cat_or_empty_1d(lst):
            return torch.cat(lst) if lst else torch.zeros(0, dtype=torch.long)

        sh  = cat_or_empty_2d(task_sh,  4096)
        sa  = cat_or_empty_2d(task_sa,  56)
        sds = cat_or_empty_2d(task_sds, 8)
        sep = cat_or_empty_1d(task_sep)
        fh  = cat_or_empty_2d(task_fh,  4096)
        fa  = cat_or_empty_2d(task_fa,  56)
        fds = cat_or_empty_2d(task_fds, 8)
        fep = cat_or_empty_1d(task_fep)

        torch.save({
            "success_h": sh, "success_a": sa, "success_ds": sds, "success_ep": sep,
            "failure_h": fh, "failure_a": fa, "failure_ds": fds, "failure_ep": fep,
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path.name}  ({n_success} success, {n_fail} failure pairs)\n")

        all_success_h.append(sh);  all_success_a.append(sa);  all_success_ds.append(sds);  all_success_ep.append(sep)
        all_failure_h.append(fh);  all_failure_a.append(fa);  all_failure_ds.append(fds);  all_failure_ep.append(fep)

    # Concatenate and save
    def cat_nonempty(lst):
        lst = [x for x in lst if x.shape[0] > 0]
        return torch.cat(lst) if lst else torch.zeros(0)

    SH  = cat_nonempty(all_success_h)
    SA  = cat_nonempty(all_success_a)
    SDS = cat_nonempty(all_success_ds)
    SEP = cat_nonempty(all_success_ep)
    FH  = cat_nonempty(all_failure_h)
    FA  = cat_nonempty(all_failure_a)
    FDS = cat_nonempty(all_failure_ds)
    FEP = cat_nonempty(all_failure_ep)

    torch.save({"h": SH, "a": SA, "ds": SDS, "ep": SEP}, os.path.join(OUT_DIR, "rollout_success_pairs.pt"))
    torch.save({"h": FH, "a": FA, "ds": FDS, "ep": FEP}, os.path.join(OUT_DIR, "rollout_failure_pairs.pt"))

    print(f"Success pairs : {SH.shape[0]:,}  → rollout_success_pairs.pt")
    print(f"Failure pairs : {FH.shape[0]:,}  → rollout_failure_pairs.pt")


if __name__ == "__main__":
    main()
