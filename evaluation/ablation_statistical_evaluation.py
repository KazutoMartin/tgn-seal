import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from utils.build_result_paths import build_model_result_paths

# =====================================================
# Configuration
#
# Reflects the 2x2 matrix from run_ablation_agg_pooling.py:
# Aggregator (last, attn) x Pooling (mean, target)
# =====================================================

AGGREGATOR_VARIANTS = ["last", "attn"]
POOLING_VARIANTS = ["mean", "target"]

# The ablation script explicitly fixes these to isolate the mechanisms
CACHE_MODE = "nocache"
HOP_DEPTH = "2hop"

AVAILABLE_KEY = ["val_aps", "new_nodes_val_aps"]
AP_KEY = AVAILABLE_KEY[0]
MODE = "mean"
ALPHA = 0.01

# Filtered to Dept 1-4 as targeted in run_ablation_agg_pooling.py
EXPERIMENTS = {
    "TGN_SEAL-Dept1": dict(model="tgn", variant="seal", dataset="dept1", n_runs=10),
    "TGN_SEAL-Dept2": dict(model="tgn", variant="seal", dataset="dept2", n_runs=10),
    "TGN_SEAL-Dept3": dict(model="tgn", variant="seal", dataset="dept3", n_runs=10),
    "TGN_SEAL-Dept4": dict(model="tgn", variant="seal", dataset="dept4", n_runs=10),
}


# =====================================================
# Helper Functions
# =====================================================

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_ap(res, key=AP_KEY, mode="best"):
    aps = np.asarray(res[key])
    if mode == "best":
        return aps.max()
    elif mode == "mean":
        return aps.mean()
    else:
        raise ValueError("mode must be 'best' or 'mean'")


def extract_training_stats(res):
    epoch_times = np.asarray(res["epoch_times"], dtype=float)
    return epoch_times.mean(), epoch_times.sum(), len(epoch_times)


def paths_for(exp, agg_variant, pool_variant):
    """Reproduce the exact --prefix run_ablation_agg_pooling.py used."""
    # Prefix format: {base_prefix}-{agg_suffix}-{pool_suffix}-nocache-{n_hops}hop
    suffix_parts = [agg_variant, pool_variant, CACHE_MODE, HOP_DEPTH]
    return build_model_result_paths(
        exp["model"],
        exp["dataset"],
        exp["n_runs"],
        variant=exp["variant"],
        extra_suffix="-".join(suffix_parts),
    )


def existing_paths(paths):
    missing = [p for p in paths if not Path(p).exists()]
    return [p for p in paths if Path(p).exists()], missing


# =====================================================
# Main Evaluation
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="ablation_summary.csv", help="Optional path to write a flat summary table for further analysis.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"AP evaluation mode: {MODE.upper()}  |  AP source key: {AP_KEY}")
    print("Benchmarking Aggregator (last vs attn) x Pooling (mean vs target)")
    print(f"Fixed Context: {CACHE_MODE}, {HOP_DEPTH}")
    print("=" * 70)

    csv_rows = []

    for exp_name, exp in EXPERIMENTS.items():
        print(f"\n{'-' * 70}\n{exp_name} (dataset={exp['dataset']})\n{'-' * 70}")

        variant_aps = {}
        variant_times = {}

        for agg in AGGREGATOR_VARIANTS:
            for pool in POOLING_VARIANTS:
                combo_key = f"{agg}-{pool}"
                paths = paths_for(exp, agg, pool)
                found, missing = existing_paths(paths)
                
                if not found:
                    print(f"  [{combo_key:14s}] SKIPPED -- no result files found, e.g. {paths[0]}")
                    continue
                if missing:
                    # Gracefully handles your 2-run partial test despite expected n_runs=10
                    print(f"  [{combo_key:14s}] partial: {len(found)}/{len(paths)} run(s) found -- using what's there")

                results = [read_pickle(p) for p in found]
                aps = np.array([extract_ap(r, mode=MODE) for r in results])
                times = np.array([extract_training_stats(r) for r in results])  # [mean_epoch, total, n_epochs]

                variant_aps[combo_key] = aps
                variant_times[combo_key] = times

                print(f"  [{combo_key:14s}] AP: {aps.mean():.4f} ± {aps.std():.4f}"
                      f"   |  mean epoch time: {times[:, 0].mean():.2f}s"
                      f"   |  mean epochs to converge: {times[:, 2].mean():.1f}")

                csv_rows.append({
                    "experiment": exp_name,
                    "dataset": exp["dataset"],
                    "aggregator": agg,
                    "pooling": pool,
                    "cache_mode": CACHE_MODE,
                    "hop": HOP_DEPTH,
                    "n_runs_found": len(found),
                    "ap_mean": aps.mean(),
                    "ap_std": aps.std(),
                    "mean_epoch_time_s": times[:, 0].mean(),
                    "mean_total_time_s": times[:, 1].mean(),
                    "mean_epochs_to_converge": times[:, 2].mean(),
                })

        # Paired comparisons against the original baseline combination (last + mean)
        baseline_key = "last-mean"
        if baseline_key in variant_aps:
            baseline_aps = variant_aps[baseline_key]
            for combo_key, eval_aps in variant_aps.items():
                if combo_key == baseline_key:
                    continue
                if len(baseline_aps) < 2 or len(eval_aps) < 2:
                    print(f"  [baseline vs {combo_key:14s}] fewer than 2 paired samples found -- skipping Wilcoxon test")
                    continue
                if len(baseline_aps) != len(eval_aps):
                    print(f"  [baseline vs {combo_key:14s}] unequal sample counts ({len(baseline_aps)} vs {len(eval_aps)}) -- skipping Wilcoxon test")
                    continue
                try:
                    stat, p_value = wilcoxon(eval_aps, baseline_aps)
                    direction = "no significant AP difference"
                    if p_value < ALPHA:
                        direction = "AP significantly DIFFERENT from baseline" if eval_aps.mean() != baseline_aps.mean() else "AP tied"
                    
                    print(f"  [baseline vs {combo_key:14s}] p={p_value:.4e} ({direction})")
                except ValueError as e:
                    print(f"  [baseline vs {combo_key:14s}] Wilcoxon test not applicable ({e})")

    if args.csv and csv_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} rows to {args.csv}")
    elif args.csv:
        print(f"\nNo result files found yet -- {args.csv} not written.")


if __name__ == "__main__":
    main()