import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
import numpy as np


# Regex to match "<prefix>.pkl" (run 0) and "<prefix>_<run_idx>.pkl" (run >= 1)
PKL_PATTERN = re.compile(r"^(.*?)(?:_(\d+))?\.pkl$")


def read_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_mean_std(values):
    """Compute float mean and std for a list of numeric values, ignoring None/NaN."""
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=float)
    return float(arr.mean()), float(arr.std())


def extract_ap(res: dict, key: str, mode: str = "mean"):
    """Replicates extract_ap from statistical_evaluation.py."""
    if key not in res or len(res[key]) == 0:
        return None
    aps = np.asarray(res[key], dtype=float)
    if mode == "best":
        return float(aps.max())
    elif mode == "mean":
        return float(aps.mean())
    else:
        raise ValueError("mode must be 'best' or 'mean'")


def extract_single_run_metrics(res: dict) -> dict:
    """Extracts both statistical_evaluation.py metrics and extra train_self_supervised.py metrics."""
    # 1. Core AP metrics from statistical_evaluation.py (MODE = "mean") + "best" mode
    trans_val_ap_mean = extract_ap(res, "val_aps", mode="mean")
    ind_val_ap_mean = extract_ap(res, "new_nodes_val_aps", mode="mean")

    trans_val_ap_best = extract_ap(res, "val_aps", mode="best")
    ind_val_ap_best = extract_ap(res, "new_nodes_val_aps", mode="best")

    val_aps_arr = np.asarray(res.get("val_aps", []), dtype=float)
    best_epoch = int(np.argmax(val_aps_arr)) if val_aps_arr.size > 0 else None

    # 2. Training time & convergence stats from statistical_evaluation.py
    epoch_times = np.asarray(res.get("epoch_times", []), dtype=float)
    mean_epoch_time = float(epoch_times.mean()) if epoch_times.size > 0 else None
    total_train_time = float(epoch_times.sum()) if epoch_times.size > 0 else None
    n_epochs = int(len(epoch_times))

    # 3. Cache hit rate from statistical_evaluation.py
    hit_rates = res.get("cache_hit_rates", [])
    cache_hit_rate = float(np.mean(hit_rates)) if hit_rates else None

    # 4. Test APs (evaluated on the best checkpoint at the end of training)
    test_ap = float(res["test_ap"]) if "test_ap" in res and res["test_ap"] is not None else None
    new_node_test_ap = (
        float(res["new_node_test_ap"])
        if "new_node_test_ap" in res and res["new_node_test_ap"] is not None
        else None
    )

    # 5. Training losses across epochs
    train_losses = np.asarray(res.get("train_losses", []), dtype=float)
    train_loss_mean = float(train_losses.mean()) if train_losses.size > 0 else None
    train_loss_final = float(train_losses[-1]) if train_losses.size > 0 else None
    train_loss_min = float(train_losses.min()) if train_losses.size > 0 else None

    # 6. Total epoch wall-clock times (training + validation)
    total_epoch_times = np.asarray(res.get("total_epoch_times", []), dtype=float)
    mean_total_epoch_time = float(total_epoch_times.mean()) if total_epoch_times.size > 0 else None
    total_wall_clock_time = float(total_epoch_times.sum()) if total_epoch_times.size > 0 else None
    val_overhead_epoch = (
        float(mean_total_epoch_time - mean_epoch_time)
        if (mean_total_epoch_time is not None and mean_epoch_time is not None)
        else None
    )

    # 7. Subgraph extraction, cache push, and DRNL micro-timings (ms)
    extraction_ms = np.asarray(res.get("mean_extraction_ms", []), dtype=float)
    push_ms = np.asarray(res.get("mean_push_ms", []), dtype=float)
    drnl_ms = np.asarray(res.get("mean_drnl_ms", []), dtype=float)

    mean_extraction_ms = float(extraction_ms.mean()) if extraction_ms.size > 0 else None
    mean_push_ms = float(push_ms.mean()) if push_ms.size > 0 else None
    mean_drnl_ms = float(drnl_ms.mean()) if drnl_ms.size > 0 else None

    # 8. GPU memory metrics (MB)
    epoch_gpu_mem = np.asarray(res.get("epoch_max_gpu_memory_mb", []), dtype=float)
    mean_epoch_gpu_mem_mb = float(epoch_gpu_mem.mean()) if epoch_gpu_mem.size > 0 else None

    if "max_gpu_memory_mb" in res and res["max_gpu_memory_mb"] is not None:
        peak_gpu_mem_mb = float(res["max_gpu_memory_mb"])
    elif epoch_gpu_mem.size > 0:
        peak_gpu_mem_mb = float(epoch_gpu_mem.max())
    else:
        peak_gpu_mem_mb = None

    return {
        "trans_val_ap_mean": trans_val_ap_mean,
        "ind_val_ap_mean": ind_val_ap_mean,
        "trans_val_ap_best": trans_val_ap_best,
        "ind_val_ap_best": ind_val_ap_best,
        "best_epoch": best_epoch,
        "test_ap": test_ap,
        "new_node_test_ap": new_node_test_ap,
        "mean_epoch_time_s": mean_epoch_time,
        "total_train_time_s": total_train_time,
        "epochs_to_converge": n_epochs,
        "cache_hit_rate": cache_hit_rate,
        "train_loss_mean": train_loss_mean,
        "train_loss_final": train_loss_final,
        "train_loss_min": train_loss_min,
        "mean_total_epoch_time_s": mean_total_epoch_time,
        "total_wall_clock_time_s": total_wall_clock_time,
        "val_overhead_per_epoch_s": val_overhead_epoch,
        "mean_extraction_ms": mean_extraction_ms,
        "mean_push_ms": mean_push_ms,
        "mean_drnl_ms": mean_drnl_ms,
        "peak_gpu_memory_mb": peak_gpu_mem_mb,
        "mean_epoch_gpu_memory_mb": mean_epoch_gpu_mem_mb,
    }


def discover_and_group_runs(folder: Path):
    """Groups .pkl files in folder by experiment prefix and sorts by run index."""
    grouped = defaultdict(list)
    for pkl_path in folder.glob("*.pkl"):
        match = PKL_PATTERN.match(pkl_path.name)
        if not match:
            continue
        prefix = match.group(1)
        run_idx = int(match.group(2)) if match.group(2) is not None else 0
        grouped[prefix].append((run_idx, pkl_path))

    # Sort each group's runs by run_idx (0, 1, 2, 3, 4, ...)
    for prefix in grouped:
        grouped[prefix].sort(key=lambda x: x[0])

    return dict(sorted(grouped.items()))


def aggregate_experiment_runs(run_entries, max_runs: int = 5) -> dict:
    """Aggregates metrics across at most `max_runs` runs for a single experiment prefix."""
    selected = run_entries[:max_runs]
    per_run_records = []

    for run_idx, path in selected:
        raw = read_pickle(path)
        metrics = extract_single_run_metrics(raw)
        metrics["run_idx"] = run_idx
        metrics["file"] = path.name
        per_run_records.append(metrics)

    def agg(key):
        mean_val, std_val = safe_mean_std([r[key] for r in per_run_records])
        return {"mean": mean_val, "std": std_val}

    peak_mems = [r["peak_gpu_memory_mb"] for r in per_run_records if r["peak_gpu_memory_mb"] is not None]

    return {
        "num_runs_found_in_folder": len(run_entries),
        "num_runs_used": len(selected),
        "files_used": [p.name for _, p in selected],
        # Exact metrics from statistical_evaluation.py
        "statistical_evaluation_metrics": {
            "transductive_val_ap_mean_mode": agg("trans_val_ap_mean"),
            "inductive_val_ap_mean_mode": agg("ind_val_ap_mean"),
            "epoch_train_time_s": agg("mean_epoch_time_s"),
            "total_train_time_s": agg("total_train_time_s"),
            "epochs_to_converge": agg("epochs_to_converge"),
            "cache_hit_rate": agg("cache_hit_rate"),
        },
        # Extra parameters from train_self_supervised.py not in statistical_evaluation.py
        "additional_metrics": {
            "transductive_test_ap": agg("test_ap"),
            "inductive_test_ap": agg("new_node_test_ap"),
            "transductive_val_ap_best_mode": agg("trans_val_ap_best"),
            "inductive_val_ap_best_mode": agg("ind_val_ap_best"),
            "best_checkpoint_epoch": agg("best_epoch"),
            "train_loss": {
                "epoch_mean": agg("train_loss_mean"),
                "final_epoch": agg("train_loss_final"),
                "min_epoch": agg("train_loss_min"),
            },
            "wall_clock_timing_s": {
                "total_epoch_time_s": agg("mean_total_epoch_time_s"),
                "total_run_wall_clock_time_s": agg("total_wall_clock_time_s"),
                "validation_overhead_per_epoch_s": agg("val_overhead_per_epoch_s"),
            },
            "operator_latency_ms": {
                "subgraph_extraction_ms": agg("mean_extraction_ms"),
                "cache_push_ms": agg("mean_push_ms"),
                "drnl_labeling_ms": agg("mean_drnl_ms"),
            },
            "gpu_memory_mb": {
                "run_peak_memory_mb": {
                    **agg("peak_gpu_memory_mb"),
                    "max_across_runs": float(max(peak_mems)) if peak_mems else None,
                },
                "epoch_average_peak_memory_mb": agg("mean_epoch_gpu_memory_mb"),
            },
        },
        "per_run_breakdown": per_run_records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the first 5 runs per experiment in a results folder into a single JSON file."
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Path to the input folder containing .pkl result files (e.g., results_baseline)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <folder_name>_summary.json)",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=5,
        help="Maximum number of runs to average per experiment prefix (default: 5)",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd() / f"{folder_path.name}_summary.json"
    )

    grouped_runs = discover_and_group_runs(folder_path)
    if not grouped_runs:
        print(f"No .pkl files found in {folder_path}")
        return

    summary = {
        "source_folder": str(folder_path),
        "max_runs_averaged": args.max_runs,
        "num_experiments": len(grouped_runs),
        "experiments": {},
    }

    for prefix, run_entries in grouped_runs.items():
        summary["experiments"][prefix] = aggregate_experiment_runs(
            run_entries, max_runs=args.max_runs
        )
        used = summary["experiments"][prefix]["num_runs_used"]
        total = summary["experiments"][prefix]["num_runs_found_in_folder"]
        print(f"[OK] {prefix}: averaged {used}/{total} run(s)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved aggregated JSON results to: {output_path}")


if __name__ == "__main__":
    main()