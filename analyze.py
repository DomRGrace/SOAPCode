"""
analyze.py

Aggregates per-episode JSON results from adapt.py across tasks and produces:
  - Per-task success rates with 95% binomial confidence intervals
  - Aggregate success rates across all tasks
  - Console table ready to transcribe into the paper

Run from openvla-oft/:
  python analyze.py --results_dir ./experiments/logs/adapt/rank_8

Expects JSON files matching: results_task*_ep*.json
"""

import json
import argparse
import math
from pathlib import Path


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_records(results_dir: Path):
    records = []
    for path in sorted(results_dir.glob("results_task*_ep*.json")):
        with open(path) as f:
            records.extend(json.load(f))
    return records


def summarise(records):
    # Group by (task_idx, mode)
    from collections import defaultdict
    groups = defaultdict(list)
    task_labels = {}
    for r in records:
        groups[(r["task_idx"], r["mode"])].append(r["success"])
        task_labels[r["task_idx"]] = r["task_label"]
    return groups, task_labels


def print_table(groups, task_labels, modes):
    tasks  = sorted(task_labels.keys())
    totals = {m: {"k": 0, "n": 0} for m in modes}

    for t in tasks:
        print(f"  task {t}  {task_labels[t][:50]}")
        for m in modes:
            results = groups.get((t, m), [])
            k, n    = sum(results), len(results)
            totals[m]["k"] += k
            totals[m]["n"] += n
            if n == 0:
                print(f"    {m:10s}: —")
            else:
                lo, hi = wilson_ci(k, n)
                print(f"    {m:10s}: {k}/{n}  ({k/n:.3f})  95% CI [{lo:.3f}, {hi:.3f}]")
        print()

    print("=" * 60)
    print("  AGGREGATE")
    for m in modes:
        k, n = totals[m]["k"], totals[m]["n"]
        if n > 0:
            lo, hi = wilson_ci(k, n)
            print(f"    {m:10s}: {k}/{n}  ({k/n:.3f})  95% CI [{lo:.3f}, {hi:.3f}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./experiments/logs/adapt/rank_8")
    parser.add_argument("--modes", nargs="+", default=["surprise", "random"],
                        help="Modes to compare (e.g. surprise random, or surprise baseline)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir)

    if not records:
        print(f"No JSON result files found in {results_dir}")
        return

    print(f"Loaded {len(records)} episode records from {results_dir}\n")
    groups, task_labels = summarise(records)
    print_table(groups, task_labels, args.modes)


if __name__ == "__main__":
    main()
