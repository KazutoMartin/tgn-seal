import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

# =====================================================
# Configuration
#
# Parses the fixed configuration ablation runs:
# Aggregator: attn | Pooling: target | Cache: layered
# =====================================================

DEPARTMENTS = [1, 2, 3, 4]
EXPECTED_RUNS = 2
PREFIX_TEMPLATE = "tgn-seal-dept{dept}-attn-target-layered"

FIXED_PARAMS = {
    "aggregator": "attn",
    "pooling": "target",
    "cache_mode": "layered",
    "hop": "2hop"
}

MODE = "mean"

# =====================================================
# Helper Functions
# =====================================================

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_ap(res, key, mode="best"):
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

def get_paths_for_dept(dept, n_runs):
    """Constructs file paths matching the directory structure (e.g., .pkl and _1.pkl)"""
    base_prefix = PREFIX_TEMPLATE.format(dept=dept)
    paths = []
    for i in range(n_runs):
        if i == 0:
            paths.append(Path(f"resultsnew2/{base_prefix}.pkl"))
        else:
            paths.append(Path(f"resultsnew2/{base_prefix}_{i}.pkl"))
    return paths

def existing_paths(paths):
    missing = [p for p in paths if not p.exists()]
    return [p for p in paths if p.exists()], missing

# =====================================================
# Main Evaluation
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="fixed_ablation_summary.csv", help="Path to write the flat summary table.")
    args = parser.parse_args()

    print("=" * 80)
    print(f"AP evaluation mode: {MODE.upper()}")
    print("Extracting fixed runs: Attention Aggregator, Target Pooling, Layered Cache")
    print("=" * 80)

    csv_rows = []

    for dept in DEPARTMENTS:
        exp_name = f"TGN_SEAL-Dept{dept}"
        print(f"\n{'-' * 80}\n{exp_name}\n{'-' * 80}")

        paths = get_paths_for_dept(dept, EXPECTED_RUNS)
        found, missing = existing_paths(paths)
        
        if not found:
            print(f"  [Dept {dept}] SKIPPED -- no result files found.")
            continue
        if missing:
            print(f"  [Dept {dept}] partial: {len(found)}/{len(paths)} run(s) found -- using what's there.")

        results = [read_pickle(p) for p in found]
        
        # Extract both Transductive and Inductive APs
        trans_aps = np.array([extract_ap(r, key="val_aps", mode=MODE) for r in results])
        ind_aps = np.array([extract_ap(r, key="new_nodes_val_aps", mode=MODE) for r in results])
        
        times = np.array([extract_training_stats(r) for r in results])  # [mean_epoch, total, n_epochs]

        print(f"  [Dept {dept}] Transductive AP: {trans_aps.mean():.4f} ± {trans_aps.std():.4f}")
        print(f"  [Dept {dept}] Inductive AP:    {ind_aps.mean():.4f} ± {ind_aps.std():.4f}")
        print(f"            | mean epoch time: {times[:, 0].mean():.2f}s"
              f"  | mean epochs to converge: {times[:, 2].mean():.1f}")

        csv_rows.append({
            "experiment": exp_name,
            "dataset": f"dept{dept}",
            "aggregator": FIXED_PARAMS["aggregator"],
            "pooling": FIXED_PARAMS["pooling"],
            "cache_mode": FIXED_PARAMS["cache_mode"],
            "hop": FIXED_PARAMS["hop"],
            "n_runs_found": len(found),
            "trans_ap_mean": trans_aps.mean(),
            "trans_ap_std": trans_aps.std(),
            "ind_ap_mean": ind_aps.mean(),
            "ind_ap_std": ind_aps.std(),
            "mean_epoch_time_s": times[:, 0].mean(),
            "mean_total_time_s": times[:, 1].mean(),
            "mean_epochs_to_converge": times[:, 2].mean(),
        })

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