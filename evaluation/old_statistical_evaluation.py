import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from utils.build_result_paths import build_models_result_paths

# =====================================================
# Configuration
# =====================================================

NUM_RUNS = 10
AVAILABLE_DATASETS = ["calls", "CollegeMsg", "email1", "email2", "email3", "email4"]

AVAILABLE_KEY = ["val_aps", "new_nodes_val_aps"]
AP_KEY = AVAILABLE_KEY[0]
MODE = "mean"
ALPHA = 0.01

MODELS = {
    "DyRep": ("dyrep", "rnn", None),
    "JODIE": ("jodie", "rnn", None),
    "TGAT": ("tgat", None, None),
    "TGN_NoMem": ("tgn", "no-mem", None),
    "TGN_Id": ("tgn", "id", None),
    "TGN_Time": ("tgn", "time", None),
    "TGN_SEAL": ("tgn", "seal", "2h"),
    "TGN_SEAL_3": ("tgn", "seal", "3h"),
}

# Priority directories to look for result files
SEARCH_DIRS = [Path("results"), Path("resultsold")]


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


def resolve_existing_path(path_str):
    """Checks for file existence in primary path and falls back to resultsold/."""
    path = Path(path_str)
    if path.exists():
        return path

    # Check alternative directories if path starts under 'results'
    for base_dir in SEARCH_DIRS:
        if path.is_relative_to(Path("results")):
            rel_path = path.relative_to(Path("results"))
            alt_path = base_dir / rel_path
            if alt_path.exists():
                return alt_path

    return None


def fetch_model_paths(dataset):
    """Generates paths via build_models_result_paths and resolves existing files."""
    generated_paths = build_models_result_paths(dataset, NUM_RUNS, MODELS)

    resolved_results = {}
    for model_name, path_list in generated_paths.items():
        found, missing = [], []
        for p in path_list:
            resolved = resolve_existing_path(p)
            if resolved:
                found.append(resolved)
            else:
                missing.append(p)
        resolved_results[model_name] = (found, missing, path_list)

    return resolved_results


# =====================================================
# Main Evaluation
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write flat summary CSV.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"AP evaluation mode: {MODE.upper()}  |  AP source key: {AP_KEY}")
    print(f"Searching directories: {[str(d) for d in SEARCH_DIRS]}")
    print("=" * 70)

    csv_rows = []

    for dataset in AVAILABLE_DATASETS:
        print(f"\n{'-' * 70}\nDataset: {dataset}  (expected n_runs={NUM_RUNS})\n{'-' * 70}")

        path_data = fetch_model_paths(dataset)
        model_aps = {}
        model_times = {}

        for model_name in MODELS.keys():
            found, missing, original_paths = path_data[model_name]

            if not found:
                print(f"  [{model_name:14s}] SKIPPED -- no result files found (e.g., {original_paths[0]})")
                continue
            if missing:
                print(f"  [{model_name:14s}] Partial: {len(found)}/{len(original_paths)} run(s) found -- using available runs")

            results = [read_pickle(p) for p in found]
            aps = np.array([extract_ap(r, mode=MODE) for r in results])
            times = np.array([extract_training_stats(r) for r in results])  # [mean_epoch, total, n_epochs]

            model_aps[model_name] = aps
            model_times[model_name] = times

            print(f"  [{model_name:14s}] AP: {aps.mean():.4f} ± {aps.std():.4f}"
                  f"   |  mean epoch time: {times[:, 0].mean():.2f}s"
                  f"   |  mean total time: {times[:, 1].mean():.2f}s"
                  f"   |  mean epochs: {times[:, 2].mean():.1f}")

            csv_rows.append({
                "dataset": dataset,
                "model": model_name,
                "n_runs_found": len(found),
                "ap_mean": aps.mean(),
                "ap_std": aps.std(),
                "mean_epoch_time_s": times[:, 0].mean(),
                "mean_total_time_s": times[:, 1].mean(),
                "mean_epochs_to_converge": times[:, 2].mean(),
            })

        # Paired Wilcoxon Signed-Rank Tests: TGN_SEAL vs Baselines
        if "TGN_SEAL" in model_aps and len(model_aps["TGN_SEAL"]) >= 2:
            print(f"\n  --- Statistical Tests (TGN_SEAL vs Baselines on {dataset}) ---")
            tgn_seal_ap = model_aps["TGN_SEAL"]

            for baseline in MODELS.keys():
                if baseline == "TGN_SEAL" or baseline not in model_aps:
                    continue

                baseline_ap = model_aps[baseline]
                if len(baseline_ap) != len(tgn_seal_ap):
                    print(f"    [TGN_SEAL vs {baseline:12s}] Unequal sample counts ({len(tgn_seal_ap)} vs {len(baseline_ap)}) -- skipping test")
                    continue

                try:
                    stat, p_value = wilcoxon(tgn_seal_ap, baseline_ap, alternative="greater")
                    significant = "✅ YES" if p_value < ALPHA else "❌ NO"
                    print(f"    [TGN_SEAL vs {baseline:12s}] p-value={p_value:.4e} | Significant improvement? {significant}")
                except ValueError as e:
                    print(f"    [TGN_SEAL vs {baseline:12s}] Wilcoxon test not applicable ({e})")

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