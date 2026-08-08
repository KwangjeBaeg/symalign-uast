"""
F2 Statistical Analysis — Exp01 E2 vs E3 Paired Significance Tests.

Loads original 5-seed results (geometry_ablation_results.json) and
optionally follow-up results (exp1_followup_results.json), then computes:
  - Paired t-test (5 seeds) for ZS Top-1, MAP@5, kNN-10, Sep.Ratio
  - 95% bootstrap CI for each metric difference (E3 - E2)
  - Cohen's d effect size

Usage:
    python experiments/geometry_ablation/statistical_analysis.py
    python experiments/geometry_ablation/statistical_analysis.py --followup
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from scipy import stats


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ORIG_PATH = os.path.join(RESULTS_DIR, "geometry_ablation_results.json")
FOLLOWUP_PATH = os.path.join(RESULTS_DIR, "exp1_followup_results.json")

SEED_LIST = [42, 123, 456, 789, 2026]


def load_per_seed(path: str, variants: List[str]) -> dict:
    """Load per-seed metric arrays from results JSON.

    Returns: {variant: {metric: [v_seed42, v_seed123, ...]}}
    """
    with open(path) as f:
        data = json.load(f)

    per_run = data["per_run"]
    result = {v: {} for v in variants}

    for v in variants:
        v_runs = sorted(
            [r for r in per_run if r["variant"] == v],
            key=lambda r: SEED_LIST.index(r["seed"]) if r["seed"] in SEED_LIST else r["seed"],
        )

        metrics = {
            "indist_top1": [r["indist"]["top1"] for r in v_runs],
            "zs_top1":     [r["zero_shot"]["zs_top1"] for r in v_runs],
            "map5":        [r["zero_shot"]["map5"] for r in v_runs],
            "knn10":       [r["zero_shot"]["knn10"] for r in v_runs],
            "sep_ratio":   [r["geometry"]["sep_ratio"] for r in v_runs],
            "norm_depth":  [
                r["geometry"]["norm_depth_corr"] for r in v_runs
                if r["geometry"]["norm_depth_corr"] is not None
                and not (isinstance(r["geometry"]["norm_depth_corr"], float)
                         and np.isnan(r["geometry"]["norm_depth_corr"]))
            ],
        }
        result[v] = metrics

    return result


def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for the mean of diffs."""
    rng = np.random.default_rng(rng_seed)
    boot_means = [
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (mean diff / pooled std)."""
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def run_analysis(label: str, e2: dict, e3: dict, metrics_to_test: List[str]):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"  {'Metric':<18} {'E2 mean':>10} {'E3 mean':>10} {'Diff (E3-E2)':>14} "
          f"{'p-value':>10} {'95% CI':>22} {'d':>7}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*14} {'-'*10} {'-'*22} {'-'*7}")

    for metric in metrics_to_test:
        e2_vals = np.array(e2.get(metric, []))
        e3_vals = np.array(e3.get(metric, []))

        if len(e2_vals) == 0 or len(e3_vals) == 0:
            print(f"  {metric:<18} {'N/A':>10} {'N/A':>10} {'N/A':>14} {'N/A':>10}")
            continue

        n = min(len(e2_vals), len(e3_vals))
        e2_vals = e2_vals[:n]
        e3_vals = e3_vals[:n]

        diffs = e3_vals - e2_vals
        mean_e2 = e2_vals.mean()
        mean_e3 = e3_vals.mean()
        mean_diff = diffs.mean()

        # Paired t-test (two-tailed)
        t_stat, p_val = stats.ttest_rel(e3_vals, e2_vals)

        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(diffs)

        # Cohen's d
        d = cohen_d(e3_vals, e2_vals)

        sig_marker = ""
        if p_val < 0.001:
            sig_marker = "***"
        elif p_val < 0.01:
            sig_marker = "**"
        elif p_val < 0.05:
            sig_marker = "*"
        else:
            sig_marker = "n.s."

        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
        print(
            f"  {metric:<18} {mean_e2:>10.4f} {mean_e3:>10.4f} "
            f"{mean_diff:>+14.4f} {p_val:>9.4f}{sig_marker:>1s} "
            f"{ci_str:>22} {d:>+7.3f}"
        )

    # Per-seed detail
    print(f"\n  Per-seed detail (seed | E2 | E3 | diff):")
    key_metric = "map5"
    e2_vals = e2.get(key_metric, [])
    e3_vals = e3.get(key_metric, [])
    for i, (v2, v3) in enumerate(zip(e2_vals, e3_vals)):
        seed = SEED_LIST[i] if i < len(SEED_LIST) else f"run{i}"
        print(f"    seed={seed:<5} | {v2:.4f} | {v3:.4f} | {v3-v2:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup", action="store_true",
                        help="Also load follow-up (balanced val criterion) results")
    args = parser.parse_args()

    metrics = ["zs_top1", "map5", "knn10", "sep_ratio", "indist_top1"]

    # ── Original results (accuracy val criterion) ──
    if not os.path.exists(ORIG_PATH):
        print(f"ERROR: {ORIG_PATH} not found. Run Exp01 first.")
        return

    orig = load_per_seed(ORIG_PATH, ["E2", "E3"])
    run_analysis(
        "ORIGINAL (val_criterion=accuracy) — E2 vs E3",
        orig["E2"], orig["E3"], metrics,
    )

    # ── Follow-up results (balanced val criterion) ──
    if args.followup:
        if not os.path.exists(FOLLOWUP_PATH):
            print(f"\nFollow-up results not yet available: {FOLLOWUP_PATH}")
        else:
            followup = load_per_seed(FOLLOWUP_PATH, ["E2", "E3"])
            run_analysis(
                "FOLLOW-UP (val_criterion=balanced) — E2 vs E3",
                followup["E2"], followup["E3"], metrics,
            )

            # ── Cross-criterion comparison (E3 only) ──
            print(f"\n{'='*70}")
            print("  E3: accuracy vs balanced val criterion")
            print(f"{'='*70}")
            print(f"  {'Metric':<18} {'accuracy':>12} {'balanced':>12} {'diff':>10}")
            print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")
            for metric in metrics:
                orig_vals = np.array(orig["E3"].get(metric, []))
                followup_vals = np.array(followup["E3"].get(metric, []))
                if len(orig_vals) == 0 or len(followup_vals) == 0:
                    continue
                n = min(len(orig_vals), len(followup_vals))
                diff = followup_vals[:n].mean() - orig_vals[:n].mean()
                print(f"  {metric:<18} {orig_vals[:n].mean():>12.4f} "
                      f"{followup_vals[:n].mean():>12.4f} {diff:>+10.4f}")

    print("\n  Significance legend: * p<0.05  ** p<0.01  *** p<0.001  n.s. p≥0.05")
    print("  Diff = E3 - E2  (positive = E3 higher)\n")


if __name__ == "__main__":
    main()
