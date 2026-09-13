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
    "DyRep-Dept2":               dict(model="dyrep", variant="rnn",      dataset="dept2",     extra=None, n_runs=10),
    "DyRep-CollegeMsg":          dict(model="dyrep", variant="rnn",      dataset="CollegeMsg", extra=None, n_runs=10),
    "JODIE-CollegeMsg":          dict(model="jodie", variant="rnn",      dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_NoMem-Dept4":           dict(model="tgn",   variant="no-mem",   dataset="dept4",     extra=None, n_runs=10),
    "TGN_NoMem-CollegeMsg":      dict(model="tgn",   variant="no-mem",   dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_Time-Dept4":            dict(model="tgn",   variant="time",     dataset="dept4",     extra=None, n_runs=10),
    "TGN_Time-CollegeMsg":       dict(model="tgn",   variant="time",     dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_Id-Dept4":              dict(model="tgn",   variant="id",       dataset="dept4",     extra=None, n_runs=10),
    "TGN_Id-CollegeMsg":         dict(model="tgn",   variant="id",       dataset="CollegeMsg", extra=None, n_runs=10),
    "TGN_SEAL-Dept1":            dict(model="tgn",   variant="seal",     dataset="dept1",     extra=None, n_runs=10),
    "TGN_SEAL-Dept2":            dict(model="tgn",   variant="seal",     dataset="dept2",     extra=None, n_runs=10),
    "TGN_SEAL-Dept3":            dict(model="tgn",   variant="seal",     dataset="dept3",     extra=None, n_runs=10),
    "TGN_SEAL-Dept4":            dict(model="tgn",   variant="seal",     dataset="dept4",     extra=None, n_runs=10),
    "TGN_SEAL-Calls":            dict(model="tgn",   variant="seal",     dataset="calls",     extra=None, n_runs=1),
    "TGN_SEAL_Attn-CollegeMsg":  dict(model="tgn",   variant="seal-attn", dataset="CollegeMsg", extra=None, n_runs=10),
    "TGAT-CollegeMsg":           dict(model="tgat",  variant=None,       dataset="CollegeMsg", extra=None, n_runs=10),
}


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


def extract_cache_hit_rate(res):
    """Mean cache hit rate across epochs. None for nocache runs, or for
    result files predating this field."""
    rates = res.get("cache_hit_rates")
    if not rates:
        return None
    return float(np.mean(rates))


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
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write a flat summary table (one row per experiment/hop/cache combo) for further analysis in pandas/Excel.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"AP evaluation mode: {MODE.upper()}")
    print("Comparing nocache vs. cache vs. layered-cache, at each hop depth")
    print("=" * 70)

    csv_rows = []

    for exp_name, exp in EXPERIMENTS.items():
        for hv in HOP_VARIANTS:
            print(f"\n{'-' * 70}\n{exp_name} [{hv}]  (dataset={exp['dataset']}, n_runs={exp['n_runs']})\n{'-' * 70}")

            variant_aps = {}
            variant_times = {}

            for cv in CACHE_VARIANTS:
                paths = paths_for(exp, cv, hv)
                found, missing = existing_paths(paths)
                if not found:
                    print(f"  [{cv:14s}] SKIPPED -- no result files found, e.g. {paths[0]}")
                    continue
                if missing:
                    print(f"  [{cv:14s}] partial: {len(found)}/{len(paths)} run(s) found (expected n_runs={exp['n_runs']}) -- using what's there")

                results = [read_pickle(p) for p in found]
                
                # Extract both Transductive and Inductive APs
                trans_aps = np.array([extract_ap(r, key="val_aps", mode=MODE) for r in results])
                ind_aps = np.array([extract_ap(r, key="new_nodes_val_aps", mode=MODE) for r in results])
                
                times = np.array([extract_training_stats(r) for r in results])  # [mean_epoch, total, n_epochs]
                hit_rates = [extract_cache_hit_rate(r) for r in results]
                hit_rates = [h for h in hit_rates if h is not None]
                mean_hit_rate = float(np.mean(hit_rates)) if hit_rates else None

                variant_aps[cv] = {"trans": trans_aps, "ind": ind_aps}
                variant_times[cv] = times

                hit_rate_str = f"{mean_hit_rate * 100:.1f}%" if mean_hit_rate is not None else "n/a"
                print(f"  [{cv:14s}] Transductive AP: {trans_aps.mean():.4f} ± {trans_aps.std():.4f}")
                print(f"  [{'':14s}] Inductive AP:    {ind_aps.mean():.4f} ± {ind_aps.std():.4f}")
                print(f"  [{'':14s}] mean epoch time: {times[:, 0].mean():.2f}s"
                      f"   |  mean total time: {times[:, 1].mean():.2f}s"
                      f"   |  mean epochs to converge: {times[:, 2].mean():.1f}"
                      f"   |  cache hit rate: {hit_rate_str}")

                csv_rows.append({
                    "experiment": exp_name,
                    "dataset": exp["dataset"],
                    "hop": hv,
                    "cache_mode": cv,
                    "n_runs_found": len(found),
                    "trans_ap_mean": trans_aps.mean(),
                    "trans_ap_std": trans_aps.std(),
                    "ind_ap_mean": ind_aps.mean(),
                    "ind_ap_std": ind_aps.std(),
                    "mean_epoch_time_s": times[:, 0].mean(),
                    "mean_total_time_s": times[:, 1].mean(),
                    "mean_epochs_to_converge": times[:, 2].mean(),
                    "cache_hit_rate": mean_hit_rate if mean_hit_rate is not None else "",
                })

            # Paired comparisons: does caching change AP vs. the nocache baseline, at this hop depth?
            if "nocache" in variant_aps:
                baseline_trans_aps = variant_aps["nocache"]["trans"]
                baseline_ind_aps = variant_aps["nocache"]["ind"]
                
                for cv in ["cache", "layered-cache"]:
                    if cv not in variant_aps:
                        continue
                        
                    eval_trans_aps = variant_aps[cv]["trans"]
                    eval_ind_aps = variant_aps[cv]["ind"]

                    # Speedup is independent of AP metric, calculate and display once per cache variant
                    speedup = baseline_trans_aps.size and (variant_times["nocache"][:, 1].mean() / variant_times[cv][:, 1].mean())
                    print(f"  [nocache vs {cv:14s}] speedup={speedup:.2f}x total training time")

                    # Perform Wilcoxon for both Transductive and Inductive APs
                    for ap_type, eval_aps, baseline_aps in [("Trans", eval_trans_aps, baseline_trans_aps), ("Induc", eval_ind_aps, baseline_ind_aps)]:
                        if len(baseline_aps) < 2 or len(eval_aps) < 2:
                            print(f"  [{ap_type} nocache vs {cv}] fewer than 2 paired samples found -- skipping Wilcoxon test")
                            continue
                        if len(baseline_aps) != len(eval_aps):
                            print(f"  [{ap_type} nocache vs {cv}] unequal sample counts ({len(baseline_aps)} vs {len(eval_aps)}) -- skipping paired Wilcoxon test")
                            continue
                        try:
                            stat, p_value = wilcoxon(eval_aps, baseline_aps)
                            direction = "no significant AP difference"
                            if p_value < ALPHA:
                                direction = "AP significantly DIFFERENT from nocache" if eval_aps.mean() != baseline_aps.mean() else "AP tied"
                            
                            print(f"  [{ap_type} nocache vs {cv:14s}] p={p_value:.4e} ({direction})")
                        except ValueError as e:
                            print(f"  [{ap_type} nocache vs {cv}] Wilcoxon test not applicable ({e})")

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