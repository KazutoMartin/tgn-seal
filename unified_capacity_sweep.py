#!/usr/bin/env python3
"""
Unified Capacity Sweep & Dynamic TTL Optimization for Flat and Layered Caches.

Features:
  1. Dynamic P50 Pair Reply Time Calculation: tau_TTL = Delta t_{P50} (dyadic conversational window).
  2. Empirical Branching Factor Estimation: M = H2_P95 / H1_P95 evaluated over tau_TTL (Layered mode).
  3. Capacity Sweeps:
     - Flat mode: Sweeps C_max.
     - Layered mode: Sweeps C1 and dynamically scales C2 = round(C1 * M).
  4. Plateau Mechanism Detection: Identifies both peak F2 and the start-of-plateau threshold
     (lowest capacity achieving >= 99% of peak F2 to minimize array extraction latency).
  5. Organized Export: Saves sweep JSONs into --json_dir and visualization grids into --plot_dir.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

from utils.data_processing import get_data
from utils.utils import (
    RandEdgeSampler,
    TemporalSubgraphCache,
    MultiLayerTemporalCache,
    get_neighbor_finder,
)

DEFAULT_DATASETS = [
    # "CollegeMsg",
    "email-Eu-core-temporal-Dept1",
    "email-Eu-core-temporal-Dept2",
    "email-Eu-core-temporal-Dept3",
    "email-Eu-core-temporal-Dept4",
    # "calls",
]


# ==============================================================================
# 1. Dynamic P50 Pair Reply Time (tau_TTL) Calculation
# ==============================================================================

def compute_p50_pair_ttl(train_data) -> Tuple[float, float]:
    """
    Computes the empirical median (P50) dyadic/pair reply time: tau_TTL = Delta t_{P50}.
    Returns:
        tau_ttl (float): Dynamic TTL in seconds.
        node_p50 (float): Node-level P50 fallback in seconds.
    """
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps

    # Ensure chronological order
    sort_idx = np.argsort(timestamps)
    sources = sources[sort_idx]
    destinations = destinations[sort_idx]
    timestamps = timestamps[sort_idx]

    node_last_seen = {}
    pair_last_seen = {}
    node_deltas = []
    pair_deltas = []

    for s, d, t in zip(sources, destinations, timestamps):
        # Node-level inter-arrival
        for n in (s, d):
            if n in node_last_seen:
                dt = t - node_last_seen[n]
                if dt > 0:
                    node_deltas.append(dt)
            node_last_seen[n] = t

        # Dyadic / Pair-level conversation reply time
        pair_key = (min(s, d), max(s, d))
        if pair_key in pair_last_seen:
            dt_pair = t - pair_last_seen[pair_key]
            if dt_pair > 0:
                pair_deltas.append(dt_pair)
        pair_last_seen[pair_key] = t

    node_p50 = float(np.percentile(node_deltas, 50)) if len(node_deltas) > 0 else 86400.0

    if len(pair_deltas) > 0:
        pair_p50 = float(np.percentile(pair_deltas, 50))
        tau_ttl = pair_p50 if pair_p50 > 0 else node_p50
    else:
        tau_ttl = node_p50

    return tau_ttl, node_p50


# ==============================================================================
# 2. Empirical Branching Factor M Calculation
# ==============================================================================

def compute_branching_factor(train_data, ttl_window: float, max_sample_nodes: int = 300) -> float:
    """
    Estimates the empirical Temporal Branching Factor M:
        M = H2^{P95} / H1^{P95}
    measured across dynamic temporal window tau_TTL.
    """
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    unique_nodes = np.unique(np.concatenate([sources, destinations]))

    max_node = max(sources.max(), destinations.max())
    adj_list = [[] for _ in range(max_node + 1)]
    for s, d, t in zip(sources, destinations, timestamps):
        adj_list[s].append((d, t))
        adj_list[d].append((s, t))

    if len(unique_nodes) > max_sample_nodes:
        sample_nodes = np.random.choice(unique_nodes, max_sample_nodes, replace=False)
    else:
        sample_nodes = unique_nodes

    h1_vols, h2_vols = [], []
    for node in sample_nodes:
        interactions = adj_list[node]
        if not interactions:
            continue
        sample_times = [t for _, t in interactions[::max(1, len(interactions) // 25)]]
        for q_time in sample_times:
            h1_nodes = set()
            h1_cnt = 0
            for neighbor, t in interactions:
                if q_time - ttl_window <= t <= q_time:
                    h1_nodes.add(neighbor)
                    h1_cnt += 1
            if h1_cnt == 0:
                continue

            h2_cnt = 0
            for h1_n in h1_nodes:
                for h2_n, t2 in adj_list[h1_n]:
                    if q_time - ttl_window <= t2 <= q_time and h2_n != node:
                        h2_cnt += 1

            h1_vols.append(h1_cnt)
            h2_vols.append(h2_cnt)

    h1_p95 = np.percentile(h1_vols, 95) if h1_vols else 1.0
    h2_p95 = np.percentile(h2_vols, 95) if h2_vols else 1.0
    multiplier = (h2_p95 / h1_p95) if h1_p95 > 0 else 1.0
    return float(multiplier)


# ==============================================================================
# 3. Capacity Step Evaluation
# ==============================================================================

def calculate_set_metrics(base_list: List, cache_list: List) -> Tuple[float, float, float]:
    base_set = set(base_list)
    cache_set = set(cache_list)

    intersect = len(base_set.intersection(cache_set))
    union = len(base_set.union(cache_set))

    recall = intersect / len(base_set) if len(base_set) > 0 else 1.0
    precision = intersect / len(cache_set) if len(cache_set) > 0 else 1.0
    iou = intersect / union if union > 0 else 1.0

    return recall, precision, iou


def evaluate_capacity_step(
    train_data,
    train_sampler,
    base_finder,
    mode: str,
    capacity_config: Dict[str, Any],
    ttl: float,
    batch_size: int = 200,
    eval_freq: int = 1,
) -> Dict[str, float]:
    """
    Evaluates one capacity point for either flat or layered cache.
    """
    if mode == "flat":
        finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
        finder.cache = TemporalSubgraphCache(
            ttl_window=ttl, max_edges=capacity_config["c_max"]
        )
    else:
        finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
        finder.cache = MultiLayerTemporalCache(
            ttl_window=ttl,
            max_edges_per_hop={1: capacity_config["c1"], 2: capacity_config["c2"]},
        )

    metrics = {"recall": [], "precision": [], "iou": [], "f2": [], "latency_ms": []}
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    for k in range(num_batch):
        start_idx = k * batch_size
        end_idx = min(num_instance, start_idx + batch_size)

        sources = train_data.sources[start_idx:end_idx]
        destinations = train_data.destinations[start_idx:end_idx]
        timestamps = train_data.timestamps[start_idx:end_idx]
        edge_idxs = train_data.edge_idxs[start_idx:end_idx]
        size = len(sources)

        if k % eval_freq == 0:
            _, negatives = train_sampler.sample(size)

            # Ground truth subgraphs
            base_pos = base_finder.extract_enclosing_subgraph(
                sources, destinations, timestamps, y=1, use_cache=False
            )
            random.seed(42)
            base_neg = base_finder.extract_enclosing_subgraph(
                sources, negatives, timestamps, y=0, use_cache=False
            )

            # Cached subgraphs & latency profiling
            start_time = time.perf_counter()
            cached_pos = finder.extract_enclosing_subgraph(
                sources, destinations, timestamps, y=1, use_cache=True
            )
            random.seed(42)
            cached_neg = finder.extract_enclosing_subgraph(
                sources, negatives, timestamps, y=0, use_cache=True
            )
            batch_latency = (time.perf_counter() - start_time) * 1000.0 / (size * 2)
            metrics["latency_ms"].append(batch_latency)

            for i in range(size):
                for b_data, c_data in [(base_pos[i], cached_pos[i]), (base_neg[i], cached_neg[i])]:
                    b_edges = b_data.edge_time.tolist()
                    c_edges = c_data.edge_time.tolist()
                    r, p, iou = calculate_set_metrics(b_edges, c_edges)

                    # Recall-heavy F2 score
                    f2 = (5.0 * p * r) / ((4.0 * p) + r) if (4.0 * p + r) > 0 else 0.0
                    metrics["recall"].append(r)
                    metrics["precision"].append(p)
                    metrics["iou"].append(iou)
                    metrics["f2"].append(f2)

        # Stream interaction edges into cache
        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            finder.cache.push_edge(s, d, t, e_idx, finder)

    hits = finder.cache.cache_hits
    misses = finder.cache.cache_misses
    total_ops = hits + misses
    hit_rate = (hits / total_ops) if total_ops > 0 else 0.0

    return {
        "recall": float(np.mean(metrics["recall"])),
        "precision": float(np.mean(metrics["precision"])),
        "iou": float(np.mean(metrics["iou"])),
        "f2": float(np.mean(metrics["f2"])),
        "latency_ms": float(np.mean(metrics["latency_ms"])),
        "cache_hit_rate": float(hit_rate),
        "cache_hits": int(hits),
        "cache_misses": int(misses),
    }


# ==============================================================================
# 4. Plateau Detection & Optimum Finding
# ==============================================================================

def find_plateau_optimum(
    records: List[Dict[str, Any]], mode: str, plateau_ratio: float = 0.99
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Finds:
      1. Peak F2 record (global maximum structural fidelity).
      2. Plateau start record: The lowest capacity point achieving >= plateau_ratio (99%)
         of peak F2, minimizing extraction latency before queue saturation.
    """
    peak_record = max(records, key=lambda x: x["f2"])
    target_f2 = peak_record["f2"] * plateau_ratio

    # Records sorted by primary capacity (c_max or c1)
    cap_key = "c_max" if mode == "flat" else "c1"
    sorted_records = sorted(records, key=lambda x: x[cap_key])

    plateau_record = peak_record
    for r in sorted_records:
        if r["f2"] >= target_f2:
            plateau_record = r
            break

    return peak_record, plateau_record


# ==============================================================================
# 5. Plotting Grid Generation
# ==============================================================================

def generate_sweep_plot(
    records: List[Dict[str, Any]],
    dataset_name: str,
    mode: str,
    tau_ttl: float,
    branching_m: float,
    peak_record: Dict[str, Any],
    plateau_record: Dict[str, Any],
    output_dir: Path,
):
    cap_key = "c_max" if mode == "flat" else "c1"
    x_label = "$C_{max}$ (Flat Queue Capacity)" if mode == "flat" else "$C_1$ (Hop-1 Edge Capacity)"

    caps = [r[cap_key] for r in records]
    ious = [r["iou"] for r in records]
    f2s = [r["f2"] for r in records]
    precisions = [r["precision"] for r in records]
    recalls = [r["recall"] for r in records]
    hit_rates = [r["cache_hit_rate"] * 100.0 for r in records]
    latencies = [r["latency_ms"] for r in records]

    opt_cap = plateau_record[cap_key]
    peak_cap = peak_record[cap_key]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mode_title = "Flat Cache" if mode == "flat" else f"Layered Cache (Branching M={branching_m:.2f}x)"
    fig.suptitle(
        f"{mode_title} Dynamic P50 TTL Sweep: {dataset_name}\n"
        f"Dynamic tau_TTL = {tau_ttl:.1f}s ({tau_ttl/3600.0:.2f}h) | "
        f"Plateau Optimum: {cap_key}={opt_cap} (F2={plateau_record['f2']:.4f}, Lat={plateau_record['latency_ms']:.2f}ms) | "
        f"Peak F2: {cap_key}={peak_cap} (F2={peak_record['f2']:.4f})",
        fontsize=13,
        fontweight="bold",
    )

    plot_configs = [
        (axes[0, 0], f2s, "Structural Fidelity (F2 Score)", "teal", "F2"),
        (axes[0, 1], ious, "Intersection over Union (IoU)", "purple", "IoU"),
        (axes[0, 2], hit_rates, "Cache Hit Rate (%)", "darkblue", "Hit Rate (%)"),
        (axes[1, 0], recalls, "Subgraph Edge Recall", "blue", "Recall"),
        (axes[1, 1], precisions, "Subgraph Edge Precision", "green", "Precision"),
        (axes[1, 2], latencies, "Extraction Latency (ms / sample)", "crimson", "Latency (ms)"),
    ]
    for ax, data, title, color, ylabel in plot_configs:
        ax.plot(caps, data, marker="o", color=color, linewidth=2, markersize=5)
        ax.axvline(
            x=opt_cap,
            color="red",
            linestyle="--",
            alpha=0.85,
            label=f"Plateau Start ({cap_key}={opt_cap})",
        )
        if peak_cap != opt_cap:
            ax.axvline(
                x=peak_cap,
                color="orange",
                linestyle=":",
                alpha=0.75,
                label=f"Peak F2 ({cap_key}={peak_cap})",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plot_file = output_dir / f"{dataset_name}_{mode}_optimal_ttl_sweep.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"  [Plot Saved] -> {plot_file}")


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================

def run_dataset_sweep(dataset_name: str, mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    print(f"\n" + "=" * 80)
    print(f"Processing Dataset: {dataset_name} | Mode: {mode.upper()}")
    print("=" * 80)

    try:
        _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    except Exception as err:
        print(f"Error loading dataset {dataset_name}: {err}")
        return None

    # Step 1: Compute Dynamic P50 Pair Reply Time TTL
    tau_ttl, node_p50 = compute_p50_pair_ttl(train_data)
    print(
        f"Dynamic TTL -> Pair Reply P50: {tau_ttl:.2f}s ({tau_ttl/3600.0:.2f} hrs, {tau_ttl/86400.0:.2f} days) | "
        f"Node P50: {node_p50:.2f}s"
    )

    # Step 2: Temporal Branching Factor M (for Layered Cache)
    branching_m = 1.0
    if mode == "layered":
        if args.branching_factor is not None:
            branching_m = args.branching_factor
            print(f"Using fixed Branching Multiplier M: {branching_m:.2f}x")
        else:
            print("Profiling empirical Hop-2 Branching Multiplier (M = H2_P95 / H1_P95)...")
            branching_m = compute_branching_factor(train_data, ttl_window=tau_ttl)
            print(f"Empirical Branching Multiplier M: {branching_m:.2f}x")

    # Step 3: Define Sweep Space
    if mode == "flat":
        cap_space = np.unique(np.linspace(args.min_c, args.max_c, args.steps, dtype=int))
        print(f"Sweeping C_max across {len(cap_space)} points: {cap_space.tolist()}")
    else:
        cap_space = np.unique(np.linspace(args.min_c1, args.max_c1, args.steps, dtype=int))
        print(f"Sweeping C1 across {len(cap_space)} points: {cap_space.tolist()}")

    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

    sweep_records = []
    header = (
        f"{'C_max':<6} | {'IoU':<7} | {'F2':<7} | {'Prec':<7} | {'Rec':<7} | {'Hit Rate':<9} | {'Lat (ms)':<8}"
        if mode == "flat"
        else f"{'C1':<5} | {'C2':<6} | {'IoU':<7} | {'F2':<7} | {'Prec':<7} | {'Rec':<7} | {'Hit Rate':<9} | {'Lat (ms)':<8}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for cap in cap_space:
        cap_int = int(cap)
        if mode == "flat":
            cfg = {"c_max": cap_int}
        else:
            c2_int = max(1, int(round(cap_int * branching_m)))
            cfg = {"c1": cap_int, "c2": c2_int}

        metrics = evaluate_capacity_step(
            train_data=train_data,
            train_sampler=train_sampler,
            base_finder=base_finder,
            mode=mode,
            capacity_config=cfg,
            ttl=tau_ttl,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq,
        )

        record = {
            **cfg,
            "branching_m": branching_m if mode == "layered" else None,
            "iou": metrics["iou"],
            "f2": metrics["f2"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "cache_hit_rate": metrics["cache_hit_rate"],
            "latency_ms": metrics["latency_ms"],
            "cache_hits": metrics["cache_hits"],
            "cache_misses": metrics["cache_misses"],
        }
        sweep_records.append(record)

        if mode == "flat":
            print(
                f"{cap_int:<6} | {metrics['iou']:.4f}  | {metrics['f2']:.4f}  | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['cache_hit_rate']*100:.2f}%    | {metrics['latency_ms']:.4f}"
            )
        else:
            print(
                f"{cap_int:<5} | {cfg['c2']:<6} | {metrics['iou']:.4f}  | {metrics['f2']:.4f}  | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['cache_hit_rate']*100:.2f}%    | {metrics['latency_ms']:.4f}"
            )

    # Step 4: Detect Plateau & Peak Optimum
    peak_rec, plateau_rec = find_plateau_optimum(sweep_records, mode=mode, plateau_ratio=args.plateau_ratio)

    # Step 5: Save JSON Results
    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{dataset_name}_{mode}_sweep.json"

    export_payload = {
        "dataset": dataset_name,
        "mode": mode,
        "tau_ttl_seconds": tau_ttl,
        "branching_multiplier_M": branching_m if mode == "layered" else None,
        "optimal_plateau_record": plateau_rec,
        "peak_f2_record": peak_rec,
        "sweep_records": sweep_records,
    }
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=4)
    print(f"\n  [JSON Saved] -> {json_path}")

    # Step 6: Generate and Save Plot
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    generate_sweep_plot(
        records=sweep_records,
        dataset_name=dataset_name,
        mode=mode,
        tau_ttl=tau_ttl,
        branching_m=branching_m,
        peak_record=peak_rec,
        plateau_record=plateau_rec,
        output_dir=plot_dir,
    )

    return {
        "dataset": dataset_name,
        "mode": mode,
        "tau_ttl": tau_ttl,
        "branching_m": branching_m if mode == "layered" else "-",
        "plateau_opt": plateau_rec,
        "peak_opt": peak_rec,
    }


def main():
    parser = argparse.ArgumentParser("Unified Dynamic TTL & Capacity Sweep Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="layered",
        choices=["flat", "layered", "both"],
        help="Cache architecture mode: 'flat', 'layered', or 'both'",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CollegeMsg"],
        help="Datasets to evaluate (or 'all' for standard suite)",
    )
    # Capacity bounds
    parser.add_argument("--min_c", type=int, default=30, help="Min capacity for flat cache")
    parser.add_argument("--max_c", type=int, default=300, help="Max capacity for flat cache")
    parser.add_argument("--min_c1", type=int, default=5, help="Min C1 capacity for layered cache")
    parser.add_argument("--max_c1", type=int, default=50, help="Max C1 capacity for layered cache")
    parser.add_argument("--steps", type=int, default=10, help="Number of linear capacity steps")
    parser.add_argument("--branching_factor", type=float, default=None, help="Manual M branching factor override")
    parser.add_argument("--plateau_ratio", type=float, default=0.99, help="Plateau threshold tolerance (e.g. 0.99 = 99%% of peak F2)")
    # Execution configs
    parser.add_argument("--batch_size", type=int, default=200, help="Replay batch size")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation batch frequency")
    # Output folders
    parser.add_argument("--json_dir", type=str, default="./sweep_results/json", help="Folder to save sweep JSONs")
    parser.add_argument("--plot_dir", type=str, default="./sweep_results/plots", help="Folder to save sweep plots")

    args = parser.parse_args()

    target_datasets = DEFAULT_DATASETS if "all" in args.datasets else args.datasets
    modes_to_run = ["flat", "layered"] if args.mode == "both" else [args.mode]

    final_summaries = []
    for d_name in target_datasets:
        for m in modes_to_run:
            res = run_dataset_sweep(d_name, m, args)
            if res is not None:
                final_summaries.append(res)

    # ==============================================================================
    # Final Recommended Capacities Summary Table
    # ==============================================================================
    print("\n" + "=" * 105)
    print("FINAL OPTIMAL CAPACITY RECOMMENDATIONS (PLATEAU THRESHOLD)")
    print("=" * 105)
    print(
        f"{'Dataset':<26} | {'Mode':<8} | {'tau_TTL (P50)':<16} | {'M':<6} | "
        f"{'Optimal Capacity':<20} | {'F2 Score':<9} | {'Latency (ms)':<10}"
    )
    print("-" * 105)

    for item in final_summaries:
        p_opt = item["plateau_opt"]
        ttl_str = f"{item['tau_ttl']:.1f}s ({item['tau_ttl']/3600.0:.1f}h)"
        m_str = f"{item['branching_m']:.2f}x" if isinstance(item["branching_m"], float) else "-"

        if item["mode"] == "flat":
            cap_str = f"C_max = {p_opt['c_max']}"
        else:
            cap_str = f"C1 = {p_opt['c1']}, C2 = {p_opt['c2']}"

        print(
            f"{item['dataset']:<26} | {item['mode']:<8} | {ttl_str:<16} | {m_str:<6} | "
            f"{cap_str:<20} | {p_opt['f2']:.4f}    | {p_opt['latency_ms']:.4f}"
        )
    print("=" * 105 + "\n")

    # ==============================================================================
    # Generate Global Summary JSON
    # ==============================================================================
    global_summary = []
    for item in final_summaries:
        p_opt = item["plateau_opt"]
        
        # Extract the correct capacity keys based on the mode
        if item["mode"] == "flat":
            optimal_caps = {"c_max": p_opt["c_max"]}
        else:
            optimal_caps = {"c1": p_opt["c1"], "c2": p_opt["c2"]}
            
        record = {
            "dataset": item["dataset"],
            "mode": item["mode"],
            "tau_ttl_seconds": item["tau_ttl"],
            "optimal_capacities": optimal_caps,
            "f2_score": p_opt["f2"],
            "precision": p_opt["precision"],
            "recall": p_opt["recall"]
        }
        global_summary.append(record)

    summary_path = Path(args.json_dir) / "global_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(global_summary, f, indent=4)
    print(f"Global sweep summary JSON saved to: {summary_path}\n")


if __name__ == "__main__":
    main()