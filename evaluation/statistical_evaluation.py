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
# One entry per base experiment in run_evaluation.py's EXPERIMENT_GROUPS.
# (model, variant, dataset, extra) must reproduce the --prefix used there,
# via build_base_name's "-".join([model, variant, dataset, extra]).
# n_runs must match the --n_runs passed to that experiment.
# =====================================================

CACHE_VARIANTS = ["nocache", "cache", "layered-cache"]
HOP_VARIANTS = ["2hop", "3hop"]

MODE = "mean"
ALPHA = 0.01

EXPERIMENTS = {
    "DyRep-Dept2":               dict(model="dyrep", variant="rnn",       dataset="dept2",      extra=None, n_runs=10),
    "DyRep-CollegeMsg":          dict(model="dyrep", variant="rnn",       dataset="CollegeMsg", extra=None, n_runs=10),
    "JODIE-CollegeMsg":          dict(model="jodie", variant="rnn",       dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_NoMem-Dept4":           dict(model="tgn",   variant="no-mem",    dataset="dept4",      extra=None, n_runs=10),
    "TGN_NoMem-CollegeMsg":      dict(model="tgn",   variant="no-mem",    dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_Time-Dept4":            dict(model="tgn",   variant="time",      dataset="dept4",      extra=None, n_runs=10),
    "TGN_Time-CollegeMsg":       dict(model="tgn",   variant="time",      dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_Id-Dept4":              dict(model="tgn",   variant="id",        dataset="dept4",      extra=None, n_runs=10),
    "TGN_Id-CollegeMsg":         dict(model="tgn",   variant="id",        dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_SEAL-Dept1":            dict(model="tgn",   variant="seal",      dataset="dept1",      extra=None, n_runs=10),
    "TGN_SEAL-Dept2":            dict(model="tgn",   variant="seal",      dataset="dept2",      extra=None, n_runs=10),
    "TGN_SEAL-Dept3":            dict(model="tgn",   variant="seal",      dataset="dept3",      extra=None, n_runs=10),
    "TGN_SEAL-Dept4":            dict(model="tgn",   variant="seal",      dataset="dept4",      extra=None, n_runs=10),
    "TGN_SEAL-Calls":            dict(model="tgn",   variant="seal",      dataset="calls",      extra=None, n_runs=1),
    "TGN_SEAL_Attn-CollegeMsg":  dict(model="tgn",   variant="seal-attn", dataset="CollegeMsg", extra=None, n_runs=10),
    "TGAT-CollegeMsg":           dict(model="tgat",  variant=None,        dataset="CollegeMsg", extra=None, n_runs=10),
}


# =====================================================
# Helper Functions
# =====================================================

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_ap(res, key, mode="best"):
    vals = res.get(key)
    if vals is None or len(vals) == 0:
        return np.nan
    aps = np.asarray(vals, dtype=float)
    if mode == "best":
        return float(aps.max())
    elif mode == "mean":
        return float(aps.mean())
    else:
        raise ValueError("mode must be 'best' or 'mean'")


def extract_scalar_metric(res, key):
    """Extract scalar values such as test_ap or new_node_test_ap."""
    val = res.get(key)
    if val is None:
        return None
    return float(val)


def extract_training_stats(res):
    """Returns (mean_epoch_time, total_train_time, n_epochs, mean_total_epoch_time, total_wall_time)."""
    epoch_times = np.asarray(res.get("epoch_times", []), dtype=float)
    total_epoch_times = np.asarray(res.get("total_epoch_times", []), dtype=float)

    mean_epoch = float(epoch_times.mean()) if len(epoch_times) > 0 else 0.0
    sum_epoch = float(epoch_times.sum()) if len(epoch_times) > 0 else 0.0
    n_epochs = len(epoch_times)

    mean_total = float(total_epoch_times.mean()) if len(total_epoch_times) > 0 else mean_epoch
    sum_total = float(total_epoch_times.sum()) if len(total_epoch_times) > 0 else sum_epoch

    return mean_epoch, sum_epoch, n_epochs, mean_total, sum_total


def extract_cache_hit_rate(res):
    """Mean cache hit rate across epochs. None for nocache runs or older results."""
    rates = res.get("cache_hit_rates")
    if not rates:
        return None
    return float(np.mean(rates))


def extract_latency_stats(res, key):
    """Mean latency in milliseconds across epochs for a given timing metric."""
    times = res.get(key)
    if not times:
        return None
    valid = [t for t in times if t is not None and t > 0]
    if not valid:
        return None
    return float(np.mean(valid))


def extract_final_train_loss(res):
    """Extract the training loss of the final completed epoch."""
    losses = res.get("train_losses")
    if not losses:
        return None
    return float(losses[-1])


def extract_gpu_memory(res):
    """Extract peak allocated GPU memory in MB."""
    mem = res.get("max_gpu_memory_mb")
    if mem is not None and mem > 0:
        return float(mem)
    epoch_mem = res.get("epoch_max_gpu_memory_mb")
    if epoch_mem:
        valid = [m for m in epoch_mem if m is not None and m > 0]
        if valid:
            return float(np.max(valid))
    return None


def paths_for(exp, cache_variant, hop_variant):
    """Reproduce the exact --prefix run_evaluation.py used for this
    experiment/cache/hop combination, then turn it into result paths."""
    suffix_parts = [p for p in (exp["extra"], cache_variant, hop_variant) if p]
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
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write a flat summary table (one row per experiment/hop/cache combo) for analysis in pandas/Excel.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=MODE,
        choices=["mean", "best"],
        help="Validation AP aggregation mode across epochs ('best' or 'mean'). Default: mean.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Significance threshold for paired Wilcoxon tests. Default: 0.01.",
    )
    args = parser.parse_args()

    eval_mode = args.mode
    alpha = args.alpha

    print("=" * 78)
    print(f"Validation AP evaluation mode: {eval_mode.upper()}  |  Wilcoxon Alpha: {alpha}")
    print("Comparing nocache vs. cache vs. layered-cache, at each hop depth")
    print("=" * 78)

    csv_rows = []

    for exp_name, exp in EXPERIMENTS.items():
        for hv in HOP_VARIANTS:
            print(f"\n{'-' * 78}\n{exp_name} [{hv}]  (dataset={exp['dataset']}, n_runs={exp['n_runs']})\n{'-' * 78}")

            variant_metrics = {}
            variant_times = {}
            variant_extractions = {}

            for cv in CACHE_VARIANTS:
                paths = paths_for(exp, cv, hv)
                found, missing = existing_paths(paths)
                if not found:
                    print(f"  [{cv:14s}] SKIPPED -- no result files found, e.g. {paths[0]}")
                    continue
                if missing:
                    print(f"  [{cv:14s}] partial: {len(found)}/{len(paths)} run(s) found (expected n_runs={exp['n_runs']}) -- using what's there")

                results = [read_pickle(p) for p in found]

                # Validation APs
                trans_val_aps = np.array([extract_ap(r, key="val_aps", mode=eval_mode) for r in results])
                ind_val_aps = np.array([extract_ap(r, key="new_nodes_val_aps", mode=eval_mode) for r in results])

                # Test APs (if available in saved run)
                trans_test_aps = [extract_scalar_metric(r, "test_ap") for r in results]
                trans_test_aps = np.array([x for x in trans_test_aps if x is not None])
                ind_test_aps = [extract_scalar_metric(r, "new_node_test_ap") for r in results]
                ind_test_aps = np.array([x for x in ind_test_aps if x is not None])

                # Training losses
                final_losses = [extract_final_train_loss(r) for r in results]
                final_losses = np.array([l for l in final_losses if l is not None])

                # Timing and profiling stats: [mean_epoch, total_train, n_epochs, mean_total_epoch, total_wall]
                times = np.array([extract_training_stats(r) for r in results])

                # Cache hit rate
                hit_rates = [extract_cache_hit_rate(r) for r in results]
                hit_rates = [h for h in hit_rates if h is not None]
                mean_hit_rate = float(np.mean(hit_rates)) if hit_rates else None

                # Granular latency breakdown (ms/call)
                extract_ms = [extract_latency_stats(r, "mean_extraction_ms") for r in results]
                extract_ms = [e for e in extract_ms if e is not None]
                mean_extract_ms = float(np.mean(extract_ms)) if extract_ms else None

                push_ms = [extract_latency_stats(r, "mean_push_ms") for r in results]
                push_ms = [p for p in push_ms if p is not None]
                mean_push_ms = float(np.mean(push_ms)) if push_ms else None

                drnl_ms = [extract_latency_stats(r, "mean_drnl_ms") for r in results]
                drnl_ms = [d for d in drnl_ms if d is not None]
                mean_drnl_ms = float(np.mean(drnl_ms)) if drnl_ms else None

                # GPU memory
                gpu_mems = [extract_gpu_memory(r) for r in results]
                gpu_mems = [m for m in gpu_mems if m is not None]
                mean_gpu_mem_mb = float(np.mean(gpu_mems)) if gpu_mems else None

                # Store for comparisons
                variant_metrics[cv] = {
                    "trans_val": trans_val_aps,
                    "ind_val": ind_val_aps,
                    "trans_test": trans_test_aps,
                    "ind_test": ind_test_aps,
                }
                variant_times[cv] = times
                variant_extractions[cv] = mean_extract_ms

                # Formatted Console Display
                hit_rate_str = f"{mean_hit_rate * 100:.1f}%" if mean_hit_rate is not None else "n/a"
                print(f"  [{cv:14s}] Val Transductive AP: {trans_val_aps.mean():.4f} ± {trans_val_aps.std():.4f}")
                print(f"  [{'':14s}] Val Inductive AP:    {ind_val_aps.mean():.4f} ± {ind_val_aps.std():.4f}")
                if len(trans_test_aps) == len(results) and len(trans_test_aps) > 0:
                    print(f"  [{'':14s}] Test Transductive AP:{trans_test_aps.mean():.4f} ± {trans_test_aps.std():.4f}")
                if len(ind_test_aps) == len(results) and len(ind_test_aps) > 0:
                    print(f"  [{'':14s}] Test Inductive AP:   {ind_test_aps.mean():.4f} ± {ind_test_aps.std():.4f}")
                if len(final_losses) > 0:
                    print(f"  [{'':14s}] Final Train Loss:    {final_losses.mean():.4f} ± {final_losses.std():.4f}")

                print(f"  [{'':14s}] mean epoch time: {times[:, 0].mean():.2f}s"
                      f"   |  total train time: {times[:, 1].mean():.2f}s"
                      f"   |  epochs: {times[:, 2].mean():.1f}"
                      f"   |  cache hit rate: {hit_rate_str}")

                latency_parts = []
                if mean_extract_ms is not None:
                    latency_parts.append(f"extract: {mean_extract_ms:.2f}ms")
                if mean_push_ms is not None:
                    latency_parts.append(f"push: {mean_push_ms:.2f}ms")
                if mean_drnl_ms is not None:
                    latency_parts.append(f"drnl: {mean_drnl_ms:.2f}ms")
                if mean_gpu_mem_mb is not None:
                    latency_parts.append(f"peak GPU: {mean_gpu_mem_mb:.1f}MB")

                if latency_parts:
                    print(f"  [{'':14s}] profiling: " + "  |  ".join(latency_parts))

                csv_rows.append({
                    "experiment": exp_name,
                    "model": exp["model"],
                    "variant": exp["variant"] if exp.get("variant") else "default",
                    "dataset": exp["dataset"],
                    "hop": hv,
                    "cache_mode": cv,
                    "n_runs_found": len(found),
                    "trans_ap_mean": trans_val_aps.mean() if len(trans_val_aps) > 0 else "",
                    "trans_ap_std": trans_val_aps.std() if len(trans_val_aps) > 0 else "",
                    "ind_ap_mean": ind_val_aps.mean() if len(ind_val_aps) > 0 else "",
                    "ind_ap_std": ind_val_aps.std() if len(ind_val_aps) > 0 else "",
                    "trans_test_ap_mean": trans_test_aps.mean() if len(trans_test_aps) > 0 else "",
                    "trans_test_ap_std": trans_test_aps.std() if len(trans_test_aps) > 0 else "",
                    "ind_test_ap_mean": ind_test_aps.mean() if len(ind_test_aps) > 0 else "",
                    "ind_test_ap_std": ind_test_aps.std() if len(ind_test_aps) > 0 else "",
                    "final_train_loss_mean": final_losses.mean() if len(final_losses) > 0 else "",
                    "final_train_loss_std": final_losses.std() if len(final_losses) > 0 else "",
                    "mean_epoch_time_s": times[:, 0].mean() if len(times) > 0 else "",
                    "mean_total_time_s": times[:, 1].mean() if len(times) > 0 else "",
                    "mean_total_wall_time_s": times[:, 4].mean() if len(times) > 0 else "",
                    "mean_epochs_to_converge": times[:, 2].mean() if len(times) > 0 else "",
                    "cache_hit_rate": mean_hit_rate if mean_hit_rate is not None else "",
                    "mean_extraction_ms": mean_extract_ms if mean_extract_ms is not None else "",
                    "mean_push_ms": mean_push_ms if mean_push_ms is not None else "",
                    "mean_drnl_ms": mean_drnl_ms if mean_drnl_ms is not None else "",
                    "max_gpu_memory_mb": mean_gpu_mem_mb if mean_gpu_mem_mb is not None else "",
                })

            # Paired comparisons: does caching change AP vs. the nocache baseline, at this hop depth?
            if "nocache" in variant_metrics:
                baseline = variant_metrics["nocache"]
                baseline_extract = variant_extractions.get("nocache")

                for cv in ["cache", "layered-cache"]:
                    if cv not in variant_metrics:
                        continue

                    eval_m = variant_metrics[cv]
                    eval_extract = variant_extractions.get(cv)

                    # Compute training speedup and extraction speedup
                    speedup_parts = []
                    if variant_times["nocache"][:, 1].mean() > 0 and variant_times[cv][:, 1].mean() > 0:
                        speedup = variant_times["nocache"][:, 1].mean() / variant_times[cv][:, 1].mean()
                        speedup_parts.append(f"speedup={speedup:.2f}x total training time")

                    if baseline_extract is not None and eval_extract is not None and eval_extract > 0:
                        ext_speedup = baseline_extract / eval_extract
                        speedup_parts.append(f"extraction speedup={ext_speedup:.2f}x")

                    if speedup_parts:
                        print(f"  [nocache vs {cv:14s}] " + "  |  ".join(speedup_parts))

                    # Perform Wilcoxon for Transductive/Inductive Val and Test APs
                    test_pairs = [
                        ("Val Trans", eval_m["trans_val"], baseline["trans_val"]),
                        ("Val Induc", eval_m["ind_val"], baseline["ind_val"]),
                    ]
                    if len(eval_m["trans_test"]) > 0 and len(baseline["trans_test"]) > 0:
                        test_pairs.append(("Test Trans", eval_m["trans_test"], baseline["trans_test"]))
                    if len(eval_m["ind_test"]) > 0 and len(baseline["ind_test"]) > 0:
                        test_pairs.append(("Test Induc", eval_m["ind_test"], baseline["ind_test"]))

                    for metric_name, eval_aps, baseline_aps in test_pairs:
                        if len(baseline_aps) < 2 or len(eval_aps) < 2:
                            print(f"  [{metric_name:10s} nocache vs {cv}] fewer than 2 paired samples found -- skipping Wilcoxon test")
                            continue
                        if len(baseline_aps) != len(eval_aps):
                            print(f"  [{metric_name:10s} nocache vs {cv}] unequal sample counts ({len(baseline_aps)} vs {len(eval_aps)}) -- skipping paired Wilcoxon test")
                            continue
                        try:
                            stat, p_value = wilcoxon(eval_aps, baseline_aps)
                            direction = "no significant AP difference"
                            if p_value < alpha:
                                direction = "AP significantly DIFFERENT from nocache" if eval_aps.mean() != baseline_aps.mean() else "AP tied"

                            print(f"  [{metric_name:10s} nocache vs {cv:14s}] p={p_value:.4e} ({direction})")
                        except ValueError as e:
                            print(f"  [{metric_name:10s} nocache vs {cv}] Wilcoxon test not applicable ({e})")

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