# SOAP: Surprise-Weighted Episodic Memory for Test-Time Adaptation

Implementation for the paper *Surprise-Weighted Episodic Memory for Test-Time Adaptation in Long-Horizon Manipulation*.

SOAP augments a pretrained OpenVLA-OFT backbone with a low-rank episodic memory tensor updated at test time via proprioceptive prediction error, instantiating an analogue of surprise-modulated hippocampal encoding from complementary learning systems theory.

---

## Repository Structure

```
soap_implementation/
├── adapt.py                  # Main SOAP evaluation script
├── inference.py              # ProprioHead surprise computation
├── mlp_train.py              # ProprioHead training
├── collect_rollout_pairs.py  # Rollout data collection
├── datacollect.py            # Data collection utilities
├── verify_collect.py         # Collection verification
├── analyze.py                # Results analysis
└── checkpoints/
    ├── soap_mlp_v3.pt        # Trained ProprioHead MLP
    └── soap_b_mem.pt         # Meta-trained B_mem projection (optional)
```

---

## Dependencies

### 1. openvla-oft

Clone and install as a sibling directory to this repo:

```bash
git clone https://github.com/moojink/openvla-oft.git
```

The expected directory layout is:

```
your-workspace/
├── SOAP/                  # this repo
│   └── soap_implementation/
└── openvla-oft/           # sibling directory
```

### 2. LIBERO

Clone and install inside the openvla-oft directory:

```bash
cd openvla-oft
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

### 3. Python Environment

Python 3.10 is required. Using uv:

```bash
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch with CUDA 12.1 support
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install openvla-oft and its dependencies
uv pip install -e openvla-oft
uv pip install -e openvla-oft/LIBERO

# Install SOAP and remaining dependencies
uv pip install -r SOAP/soap_implementation/requirements.txt
```

---

## Replicating Experimental Results

All evaluations run 50 episodes per task across 10 tasks (500 total rollouts) and compare three modes simultaneously: SOAP (`surprise`), undirected random updates (`random`), and vanilla OpenVLA-OFT (`baseline`).

From the `soap_implementation/` directory:

```bash
PYTHONPATH=/path/to/openvla-oft/LIBERO python adapt.py --suite libero_10
```

Available suites:

| Suite | Flag |
|-------|------|
| LIBERO-Long | `--suite libero_10` |
| LIBERO-Spatial | `--suite libero_spatial` |
| LIBERO-Object | `--suite libero_object` |
| LIBERO-Goal | `--suite libero_goal` |

Checkpoints are downloaded automatically from HuggingFace on first run.

### Results (LIBERO-Long, 500 rollouts)

| Mode | Success Rate |
|------|-------------|
| SOAP | 94.2% (471/500) |
| Baseline | 92.6% (463/500) |
| Random updates | 19.6% (98/500) |

All results obtained under 4-bit quantization on an NVIDIA RTX 4070 Super.

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MEM_RANK` | 8 | Rank of B_mem / A_mem |
| `MEM_ALPHA` | 8 | LoRA scaling (alpha/rank = 1.0) |
| `ETA` | 1e-3 | Hebbian learning rate |
| `DECAY` | 0.99 | Exponential forgetting factor |
| `ETA_S` | 0.9 | Titans surprise momentum |
| `THETA_S` | 1.0 | Surprise gradient scale |
| `SURPRISE_THRESHOLD` | 0.0 | Minimum MSE to trigger A_mem write |

To use the meta-trained B_mem projection instead of random init, set `B_MEM_PATH` in `adapt.py`:

```python
B_MEM_PATH = os.path.join(_HERE, "checkpoints", "soap_b_mem.pt")
```

---

## Output

Results are written per-task to `soap_implementation/logs/adapt_{suite}/results_task{N}.json`. Each record contains task index, episode index, mode, success, and mean surprise value. Rollout videos and surprise curves are saved for any episode where the SOAP or baseline model fails.
