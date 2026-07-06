"""
adapt.py

SOAP test-time adaptation via surprise-gated Hebbian writes to B_mem A_mem.

At every VLA query step (with one-step lag):
  1. h_t, a_t extracted; ds_actual = s_now - s_prev observed.
  2. Surprise = MSE(ProprioHead(cat(prev_h, prev_a)), ds_actual_norm).
  3. delta_h = d(surprise)/d(h_adapted)  — surprise gradient.
  4. Titans momentum: S_t = eta_s * S_{t-1} - theta_s * delta_h
  5. A_mem write: A_mem *= decay; A_mem += eta * outer(B_mem.T @ S_t, h_t)
  6. Forward hook injects: h_adapted = h + (alpha/r) * (h @ A_mem.T) @ B_mem.T
     before the action head runs.

Effective weight decomposition:
  W_eff = W_merged + B_mem @ A_mem    (within LoRA subspace)

Run from soap_implementation/:
  python adapt.py --suite libero_10
  python adapt.py --suite libero_spatial
  python adapt.py --suite libero_object
  python adapt.py --suite libero_goal

Requires openvla-oft installed as a sibling directory (../openvla-oft).
Compares adapted vs. baseline success rate on the specified LIBERO suite.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
from collections import deque
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPENVLA = os.path.join(_HERE, "..", "openvla-oft")
sys.path.insert(0, _HERE)
sys.path.insert(0, _OPENVLA)

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
from inference import load_model, compute_surprise
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# ---------------------------------------------------------------------------
# Suite configs
# ---------------------------------------------------------------------------
SUITE_CONFIGS = {
    "libero_10": {
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-10",
        "unnorm_key": "libero_10_no_noops",
        "num_tasks":  10,
        "max_steps":  520,
    },
    "libero_spatial": {
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "unnorm_key": "libero_spatial_no_noops",
        "num_tasks":  10,
        "max_steps":  300,
    },
    "libero_object": {
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object",
        "unnorm_key": "libero_object_no_noops",
        "num_tasks":  10,
        "max_steps":  300,
    },
    "libero_goal": {
        "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-goal",
        "unnorm_key": "libero_goal_no_noops",
        "num_tasks":  10,
        "max_steps":  300,
    },
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
MLP_PATH   = os.path.join(_HERE, "checkpoints", "soap_mlp_v3.pt")
B_MEM_PATH = None   # set to os.path.join(_HERE, "checkpoints", "soap_b_mem.pt") to use meta-trained B_mem

NUM_EPISODES   = 50
EPISODE_OFFSET = 0
NUM_STEPS_WAIT = 10
IMG_RESIZE     = 224

# Resolved at runtime from --suite arg (see main())
CHECKPOINT = None
TASK_NAME  = None
NUM_TASKS  = None
MAX_STEPS  = None
UNNORM_KEY = None
OUT_DIR    = None

# ---------------------------------------------------------------------------
# B_mem A_mem hyperparameters
# ---------------------------------------------------------------------------
MEM_RANK  = 8       # rank of B_mem A_mem  (default: 8)
MEM_ALPHA = 8       # LoRA scaling (alpha/rank = 1.0)
ETA       = 1e-3    # Hebbian learning rate
DECAY     = 0.99    # exponential forgetting
ETA_S     = 0.9     # Titans surprise momentum (η)
THETA_S   = 1.0     # Titans surprise gradient scale (θ)
SURPRISE_THRESHOLD = 0.0   # min MSE to trigger A_mem write (trivial baseline = 1.0)
D         = 4096    # LLM hidden dim


@dataclass
class Cfg:
    pretrained_checkpoint: str         = ""
    load_in_8bit: bool                 = False
    load_in_4bit: bool                 = True
    use_film: bool                     = False
    use_l1_regression: bool            = True
    use_diffusion: bool                = False
    num_images_in_input: int           = 2
    use_proprio: bool                  = True
    center_crop: bool                  = True
    lora_rank: int                     = 32
    unnorm_key: str                    = ""
    num_diffusion_steps_train: int     = 50
    num_diffusion_steps_inference: int = 50
    model_family: str                  = "openvla"
    num_open_loop_steps: int           = NUM_ACTIONS_CHUNK


# ---------------------------------------------------------------------------
# Memory LoRA
# ---------------------------------------------------------------------------

class MemoryLoRA:
    """
    B_mem A_mem adaptation within the LoRA subspace.

    B_mem : (D, r) — random Gaussian projection, frozen.
    A_mem : (r, D) — zero-initialised, updated via surprise-gated writes.
    S_mem : (D,)   — Titans past-surprise momentum accumulator.

    Forward: h_adapted = h + (alpha/r) * (h @ A_mem.T) @ B_mem.T
    Write:   S_t = eta_s * S_{t-1} - theta_s * delta_h
             A_mem *= decay; A_mem += eta * outer(B_mem.T @ S_t, h_t)
    """

    def __init__(self, d=D, r=MEM_RANK, alpha=MEM_ALPHA, eta=ETA, decay=DECAY, b_mem_path=None):
        self.r     = r
        self.scale = alpha / r
        self.eta   = eta
        self.decay = decay

        if b_mem_path and os.path.exists(b_mem_path):
            ckpt       = torch.load(b_mem_path, map_location=DEVICE)
            self.B_mem = ckpt["B_mem"].to(DEVICE).float()
            print(f"  Loaded B_mem from {b_mem_path}")
        else:
            self.B_mem = F.normalize(torch.randn(d, r), p=2, dim=0).to(DEVICE)
        self.A_mem = torch.zeros(r, d, device=DEVICE)   # (r, D) updated
        self.S_mem = torch.zeros(d, device=DEVICE)       # (D,) Titans momentum

    def reset(self):
        self.A_mem.zero_()
        self.S_mem.zero_()

    def write(self, h_t: torch.Tensor, a_t: torch.Tensor, ds_actual: torch.Tensor,
              surprise_model, ds_mean, ds_std, random_update: bool = False) -> float:
        """Titans-style A_mem update gated by ProprioHead prediction error.

        Error signal: delta_h = d(MSE(ds_pred, ds_norm))/d(h_adapted)
        S_t = eta_s * S_{t-1} - theta_s * delta_h
        A_mem update: outer(B_mem.T @ S_t, h_t)

        random_update: replace gradient with unit Gaussian noise (ablation control).
        Returns proprio surprise MSE for logging.
        """
        h_t       = h_t.to(DEVICE).float().detach()
        a_t       = a_t.to(DEVICE).float().detach()
        ds_actual = ds_actual.to(DEVICE).float().detach()

        with torch.no_grad():
            delta     = (h_t @ self.A_mem.T) @ self.B_mem.T
            h_adapted = h_t + self.scale * delta

        h_leaf  = h_adapted.detach().requires_grad_(True)
        ha_norm = F.normalize(torch.cat([h_leaf.unsqueeze(0), a_t.unsqueeze(0)], dim=-1), p=2, dim=-1)
        ds_norm = (ds_actual.unsqueeze(0) - ds_mean) / ds_std
        ds_pred = surprise_model(ha_norm)
        loss    = F.mse_loss(ds_pred, ds_norm)
        loss.backward()
        surprise_model.zero_grad()

        if random_update:
            delta_h = torch.randn_like(h_leaf)                  # (D,) random ablation
        else:
            delta_h = h_leaf.grad.detach()                      # (D,) surprise gradient

        if loss.item() >= SURPRISE_THRESHOLD:
            # Titans momentum: S_t = η * S_{t-1} - θ * ∇ℓ
            self.S_mem = ETA_S * self.S_mem - THETA_S * delta_h

            with torch.no_grad():
                self.A_mem *= self.decay
                key = self.B_mem.T @ self.S_mem                 # (r,)
                self.A_mem += self.eta * torch.outer(key, h_t)  # (r, D)

        return loss.item()

    def adapt(self, h: torch.Tensor) -> torch.Tensor:
        """Apply A_mem correction to action hidden states. h: (B, S, D) → same shape."""
        with torch.no_grad():
            delta = (h @ self.A_mem.T) @ self.B_mem.T
        return h + self.scale * delta

    def patch_action_head(self, action_head):
        """Monkey-patch action_head.predict_action to inject B_mem A_mem."""
        mem = self
        original = action_head.predict_action

        def patched(hidden_states, *args, **kwargs):
            adapted = mem.adapt(hidden_states.to(DEVICE).float()).to(hidden_states.dtype)
            return original(adapted, *args, **kwargs)

        action_head.predict_action = patched

    @staticmethod
    def unpatch_action_head(action_head, original_fn):
        action_head.predict_action = original_fn


# ---------------------------------------------------------------------------
# VLA query
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


def make_obs(raw_obs, resize_size):
    return {
        "full_image":  resize_image_for_policy(get_libero_image(raw_obs), resize_size),
        "wrist_image": resize_image_for_policy(get_libero_wrist_image(raw_obs), resize_size),
        "state": np.concatenate([
            raw_obs["robot0_eef_pos"],
            quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]),
    }


def get_proprio_state(raw_obs):
    return np.concatenate([
        raw_obs["robot0_eef_pos"],
        quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"],
    ])


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(vla, processor, cfg, env, task_label, proprio_projector, action_head,
                surprise_model, ds_mean, ds_std, mem_lora, adapt_enabled, initial_state,
                random_update: bool = False):

    mem_lora.reset()
    env.reset()
    raw_obs = env.set_init_state(initial_state)

    action_queue  = deque(maxlen=cfg.num_open_loop_steps)
    t             = 0
    success       = False
    replay_images = []
    surprises     = []
    times         = []

    prev_h = None
    prev_a = None
    prev_s = None

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            raw_obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        replay_images.append(get_libero_image(raw_obs))
        s_now = get_proprio_state(raw_obs)
        obs   = make_obs(raw_obs, IMG_RESIZE)

        if len(action_queue) == 0:
            actions, h_t = query_vla(vla, processor, cfg, obs, task_label, proprio_projector, action_head)
            action_queue.extend(actions)
            a_t = torch.tensor(actions, dtype=torch.float32).flatten()

            times.append((t - NUM_STEPS_WAIT) / 30.0)

            if prev_h is not None:
                ds_actual = torch.tensor(s_now - prev_s, dtype=torch.float32)
                if adapt_enabled:
                    signal = mem_lora.write(prev_h.to(DEVICE), prev_a.to(DEVICE),
                                            ds_actual, surprise_model, ds_mean, ds_std,
                                            random_update=random_update)
                else:
                    signal = compute_surprise(surprise_model, ds_mean, ds_std, prev_h, prev_a, ds_actual)
                surprises.append(signal)

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

    return success, surprises, times, replay_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(task_idx, task_suite, vla, processor, cfg, proprio_projector,
             action_head, surprise_model, ds_mean, ds_std, mem_lora):
    import json

    task           = task_suite.get_task(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    env, task_label = get_libero_env(task, "openvla", resolution=256)
    print(f"\n{'=':=<60}")
    print(f"Task {task_idx}: {task_label}")
    print(f"{'=':=<60}\n")

    MODES   = ("surprise", "random", "baseline")
    results = {m: [] for m in MODES}
    ep_data = {}

    for ep in range(NUM_EPISODES):
        ep_idx = EPISODE_OFFSET + ep
        ep_data[ep_idx] = {}

        for mode in MODES:
            adapt_on = (mode != "baseline")
            rand_on  = (mode == "random")
            set_seed_everywhere(ep_idx)
            print(f"  ep {ep + 1:2d}  [{mode:8s}] ...", end=" ", flush=True)

            success, surprises, times, replay_images = run_episode(
                vla, processor, cfg, env, task_label,
                proprio_projector, action_head,
                surprise_model, ds_mean, ds_std, mem_lora, adapt_on,
                initial_states[ep_idx],
                random_update=rand_on,
            )
            results[mode].append(success)
            ep_data[ep_idx][mode] = (success, surprises, times, replay_images)
            mean_s = np.mean(surprises) if surprises else float("nan")
            print(f"{'SUCCESS' if success else 'FAIL '}  (mean surprise={mean_s:.3f})")

        # Save videos + curves if the adapted or baseline model failed
        if not ep_data[ep_idx]["surprise"][0] or not ep_data[ep_idx]["baseline"][0]:
            for mode in MODES:
                s, surprises, times, replay_images = ep_data[ep_idx][mode]
                tag = f"task{task_idx}_ep{ep_idx}_{mode}"

                vid_path = os.path.join(OUT_DIR, f"rollout_{tag}.mp4")
                try:
                    writer = imageio.get_writer(vid_path, fps=30)
                    for frame in replay_images:
                        writer.append_data(frame)
                    writer.close()
                    print(f"  Video  → {vid_path}")
                except Exception as e:
                    print(f"  Video FAILED ({mode}): {e}")

                color = "#2ecc71" if s else "#e74c3c"
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(times[1:], surprises, color=color, linewidth=1.5)
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("proprio surprise (MSE)")
                outcome = "SUCCESS" if s else "FAIL"
                ax.set_title(f"{outcome} [{mode}] — task{task_idx} ep{ep_idx} — {task_label}")
                plt.tight_layout()
                curve_path = os.path.join(OUT_DIR, f"surprise_{tag}.png")
                fig.savefig(curve_path, dpi=150)
                plt.close(fig)
                print(f"  Curve  → {curve_path}")
        print()

    n = NUM_EPISODES
    print(f"{'':=<50}")
    for mode in MODES:
        print(f"  {mode:10s} success rate: {sum(results[mode])}/{n}")
    print(f"{'':=<50}\n")

    # Save per-episode JSON
    records = []
    for ep in range(NUM_EPISODES):
        ep_idx = EPISODE_OFFSET + ep
        for mode in MODES:
            s, surprises, _, _ = ep_data[ep_idx][mode]
            records.append({
                "task_idx":          task_idx,
                "task_label":        task_label,
                "ep_idx":            ep_idx,
                "mode":              mode,
                "success":           bool(s),
                "mean_surprise":     float(np.mean(surprises)) if surprises else None,
                "mem_rank":          MEM_RANK,
                "eta":               ETA,
                "surprise_threshold": SURPRISE_THRESHOLD,
            })

    json_path = os.path.join(OUT_DIR, f"results_task{task_idx}.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  JSON → {json_path}")

    env.close() if hasattr(env, "close") else None
    return results


def main():
    global CHECKPOINT, TASK_NAME, NUM_TASKS, MAX_STEPS, UNNORM_KEY, OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10", choices=list(SUITE_CONFIGS.keys()),
                        help="LIBERO suite to evaluate on")
    args = parser.parse_args()

    suite = SUITE_CONFIGS[args.suite]
    CHECKPOINT = suite["checkpoint"]
    TASK_NAME  = args.suite
    NUM_TASKS  = suite["num_tasks"]
    MAX_STEPS  = suite["max_steps"]
    UNNORM_KEY = suite["unnorm_key"]
    OUT_DIR    = os.path.join(_HERE, "logs", f"adapt_{args.suite}")

    print(f"Suite:      {TASK_NAME}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"UNNORM_KEY: {UNNORM_KEY}")
    print(f"Max steps:  {MAX_STEPS}")
    print(f"Output dir: {OUT_DIR}")

    # Warm up CUDA/cuDNN before robosuite's EGL renderer initializes,
    # preventing CUDNN_STATUS_NOT_INITIALIZED from context ordering conflicts.
    if torch.cuda.is_available():
        _dummy = torch.zeros(1, device=DEVICE)
        torch.cuda.synchronize()
        del _dummy

    set_seed_everywhere(7)
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = Cfg(pretrained_checkpoint=CHECKPOINT, unnorm_key=UNNORM_KEY)

    print("Loading VLA...")
    vla = get_vla(cfg); vla.eval()
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, vla.llm_dim)
    processor         = get_processor(cfg)

    print("Loading proprio surprise MLP...")
    surprise_model, ds_mean, ds_std = load_model(MLP_PATH)

    print(f"Initialising MemoryLoRA  (rank={MEM_RANK}, eta={ETA}, decay={DECAY})")
    mem_lora = MemoryLoRA(b_mem_path=B_MEM_PATH)
    _original_predict_action = action_head.predict_action
    mem_lora.patch_action_head(action_head)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_NAME]()

    all_results = {}
    for task_idx in range(NUM_TASKS):
        all_results[task_idx] = run_task(
            task_idx, task_suite, vla, processor, cfg,
            proprio_projector, action_head,
            surprise_model, ds_mean, ds_std, mem_lora,
        )

    MemoryLoRA.unpatch_action_head(action_head, _original_predict_action)

    # Final aggregate summary
    MODES = ("surprise", "random", "baseline")
    totals = {m: 0 for m in MODES}
    n_total = NUM_TASKS * NUM_EPISODES
    print(f"\n{'':=<60}")
    print("  FINAL AGGREGATE")
    for m in MODES:
        total = sum(sum(all_results[t][m]) for t in range(NUM_TASKS))
        totals[m] = total
        print(f"  {m:10s}: {total}/{n_total}")
    print(f"{'':=<60}")


if __name__ == "__main__":
    main()
