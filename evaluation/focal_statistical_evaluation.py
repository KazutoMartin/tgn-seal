import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from utils.build_result_paths import build_model_result_paths

# =====================================================
# Configuration
# =====================================================

HOP_DEPTH = 2

AGGREGATOR_VARIANTS = ["last", "attn"]
POOLING_VARIANTS = ["mean", "target"]

# Adjusted to match the exact filenames in the screenshot
LOSS_TYPE = "focal"
CACHE_MODE = "nocache"

MODE = "mean"  
ALPHA = 0.01

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


def extract_metrics(res, mode="mean"):
    """Extracts Transductive AP, Inductive AP, and epochs to converge."""
    val_aps = np.asarray(res.get("val_aps", []))
    nn_val_aps = np.asarray(res.get("new_nodes_val_aps", []))
    epoch_times = np.asarray(res.get("epoch_times", []))
    
    epochs_to_converge = len(epoch_times)
    
    if mode == "best":
        # Anchor on the epoch with the highest transductive AP
        best_epoch = val_aps.argmax() if len(val_aps) > 0 else 0
        trans_ap = val_aps[best_epoch] if len(val_aps) > 0 else 0.0
        induc_ap = nn_val_aps[best_epoch] if len(nn_val_aps) > 0 else 0.0
    elif mode == "mean":
        trans_ap = val_aps.mean() if len(val_aps) > 0 else 0.0
        induc_ap = nn_val_aps.mean() if len(nn_val_aps) > 0 else 0.0
    else:
        raise ValueError("mode must be 'best' or 'mean'")
        
    return trans_ap, induc_ap, epochs_to_converge


def paths_for(exp, agg_variant, pool_variant):
    """Reproduce the exact --prefix used in the saved results."""
    # This will generate strings like "attn-target-focal-nocache"
    suffix_parts = [agg_variant, pool_variant, LOSS_TYPE, CACHE_MODE]
    
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
    parser.add_argument("--csv", type=str, default="ablation_summary.csv", help="Optional path to write a flat summary table.")
    args = parser.parse_args()

    print("=" * 80)
    print(f"AP evaluation mode: {MODE.upper()}")
    print("Benchmarking Aggregator (last vs attn) x Pooling (mean vs target)")
    print(f"Fixed Context: {CACHE_MODE}, {HOP_DEPTH}")
    print("=" * 80)

    csv_rows = []

    for exp_name, exp in EXPERIMENTS.items():
        print(f"\n{'-' * 80}\n{exp_name} (dataset={exp['dataset']})\n{'-' * 80}")

        variant_trans_aps = {}

        for agg in AGGREGATOR_VARIANTS:
            for pool in POOLING_VARIANTS:
                combo_key = f"{agg}-{pool}"
                paths = paths_for(exp, agg, pool)
                found, missing = existing_paths(paths)
                
                if not found:
                    print(f"  [{combo_key:14s}] SKIPPED -- no result files found.")
                    continue
                if missing:
                    print(f"  [{combo_key:14s}] partial: {len(found)}/{len(paths)} run(s) found -- using what's there")

                # Extract metrics for all found runs
                results = [read_pickle(p) for p in found]
                metrics = np.array([extract_metrics(r, mode=MODE) for r in results])
                
                trans_aps = metrics[:, 0]
                induc_aps = metrics[:, 1]
                epochs_converged = metrics[:, 2]

                # Store transductive APs for baseline Wilcoxon comparison
                variant_trans_aps[combo_key] = trans_aps

                print(f"  [{combo_key:14s}] Transductive AP: {trans_aps.mean():.4f} ± {trans_aps.std():.4f} | "
                      f"Inductive AP: {induc_aps.mean():.4f} ± {induc_aps.std():.4f} | "
                      f"Epochs: {epochs_converged.mean():.1f}")

                csv_rows.append({
                    "experiment": exp_name,
                    "dataset": exp["dataset"],
                    "aggregator": agg,
                    "pooling": pool,
                    "cache_mode": CACHE_MODE,
                    "hop": HOP_DEPTH,
                    "n_runs_found": len(found),
                    "transductive_ap_mean": trans_aps.mean(),
                    "transductive_ap_std": trans_aps.std(),
                    "inductive_ap_mean": induc_aps.mean(),
                    "inductive_ap_std": induc_aps.std(),
                    "mean_epochs_to_converge": epochs_converged.mean(),
                })

        # Paired comparisons against the original baseline combination (last + mean)
        baseline_key = "last-mean"
        if baseline_key in variant_trans_aps:
            baseline_aps = variant_trans_aps[baseline_key]
            for combo_key, eval_aps in variant_trans_aps.items():
                if combo_key == baseline_key:
                    continue
                if len(baseline_aps) < 2 or len(eval_aps) < 2:
                    print(f"  [baseline vs {combo_key:14s}] < 2 paired samples -- skipping Wilcoxon")
                    continue
                if len(baseline_aps) != len(eval_aps):
                    print(f"  [baseline vs {combo_key:14s}] unequal sample counts -- skipping Wilcoxon")
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