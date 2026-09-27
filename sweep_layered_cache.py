import argparse
import json
import math
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, MultiLayerTemporalCache


def compute_branching_factor(train_data, ttl_window=86400, max_sample_nodes=300):
    """
    Empirically estimates the hop-2 branching multiplier (M = P95(Hop2) / P95(Hop1))
    over the static TTL window.
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

    # Subsample nodes for fast profiling on large graphs
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
    multiplier = (h2_p95 / h1_p95) if h1_p95 > 0 else 3.0
    return float(multiplier)


def calculate_set_metrics(base_list, cache_list):
    base_set = set(base_list)
    cache_set = set(cache_list)

    intersect = len(base_set.intersection(cache_set))
    union = len(base_set.union(cache_set))

    recall = intersect / len(base_set) if len(base_set) > 0 else 1.0
    precision = intersect / len(cache_set) if len(cache_set) > 0 else 1.0
    iou = intersect / union if union > 0 else 1.0

    return recall, precision, iou


def evaluate_capacity_step(train_data, train_sampler, base_finder, c1, c2, ttl, batch_size=200, eval_freq=1):
    """
    Simulates streaming training over the dataset, querying the multi-layer cache
    and comparing extracted subgraphs against the baseline un-cached ground truth.
    """
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    layer_finder.cache = MultiLayerTemporalCache(
        ttl_window=ttl,
        max_edges_per_hop={1: c1, 2: c2}
    )

    metrics = {
        'recall': [],
        'precision': [],
        'iou': [],
        'f2': [],
        'latency_ms': []
    }

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
            base_pos = base_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=False)
            random.seed(42)
            base_neg = base_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=False)

            # Cached subgraphs & extraction latency profiling
            start_time = time.perf_counter()
            layer_pos = layer_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=True)
            random.seed(42)
            layer_neg = layer_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=True)
            batch_latency_ms = (time.perf_counter() - start_time) * 1000.0 / (size * 2)
            metrics['latency_ms'].append(batch_latency_ms)

            # Structural fidelity metrics
            for i in range(size):
                for b_data, l_data in [(base_pos[i], layer_pos[i]), (base_neg[i], layer_neg[i])]:
                    b_edges = b_data.edge_time.tolist()
                    l_edges = l_data.edge_time.tolist()
                    r, p, iou = calculate_set_metrics(b_edges, l_edges)

                    # Recall-heavy F2 score
                    f2 = (5.0 * p * r) / ((4.0 * p) + r) if (4.0 * p + r) > 0 else 0.0

                    metrics['recall'].append(r)
                    metrics['precision'].append(p)
                    metrics['iou'].append(iou)
                    metrics['f2'].append(f2)

        # Stream interaction edges into cache
        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder)

    hits = layer_finder.cache.cache_hits
    misses = layer_finder.cache.cache_misses
    total_ops = hits + misses
    hit_rate = (hits / total_ops) if total_ops > 0 else 0.0

    return {
        'recall': float(np.mean(metrics['recall'])),
        'precision': float(np.mean(metrics['precision'])),
        'iou': float(np.mean(metrics['iou'])),
        'f2': float(np.mean(metrics['f2'])),
        'latency_ms': float(np.mean(metrics['latency_ms'])),
        'cache_hit_rate': float(hit_rate),
        'cache_hits': int(hits),
        'cache_misses': int(misses)
    }


def generate_plots(results, dataset_name, branching_factor, ttl_seconds, output_dir):
    c1_vals = [r['c1'] for r in results]
    ious = [r['iou'] for r in results]
    f2s = [r['f2'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    hit_rates = [r['cache_hit_rate'] * 100.0 for r in results]
    latencies = [r['latency_ms'] for r in results]

    # Optimal C1 point selected via max F2 score (recall-weighted)
    best_idx = int(np.argmax(f2s))
    best_c1 = c1_vals[best_idx]
    best_c2 = results[best_idx]['c2']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Multi-Layer Temporal Cache Capacity Sweep\n"
        f"Dataset: {dataset_name} | Static TTL: {ttl_seconds}s (1 day) | Branching M: {branching_factor:.2f}x\n"
        f"Recommended Optimum: C1={best_c1}, C2={best_c2} (Peak F2={f2s[best_idx]:.4f})",
        fontsize=14,
        fontweight='bold'
    )

    plot_configs = [
        (axes[0, 0], ious, 'Intersection over Union (IoU)', 'purple', 'IoU'),
        (axes[0, 1], f2s, 'F2 Score (Recall-Weighted)', 'teal', 'F2'),
        (axes[0, 2], hit_rates, 'Cache Hit Rate (%)', 'darkblue', 'Hit Rate (%)'),
        (axes[1, 0], recalls, 'Subgraph Edge Recall', 'blue', 'Recall'),
        (axes[1, 1], precisions, 'Subgraph Edge Precision', 'green', 'Precision'),
        (axes[1, 2], latencies, 'Extraction Latency (ms / sample)', 'crimson', 'Latency (ms)'),
    ]

    for ax, data, title, color, ylabel in plot_configs:
        ax.plot(c1_vals, data, marker='o', color=color, linewidth=2, markersize=5)
        ax.axvline(x=best_c1, color='red', linestyle='--', alpha=0.75, label=f'Best F2 ($C_1={best_c1}$)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('$C_1$ (Hop-1 Edge Capacity)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')

    plt.tight_layout()
    plot_path = output_dir / f"{dataset_name}_capacity_sweep.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved visualization grid to: {plot_path}")


def main():
    parser = argparse.ArgumentParser("Linear Parameter Sweep for Multi-Layer Cache Capacity")
    parser.add_argument("-d", "--data", type=str, default="wikipedia", help="Dataset name (e.g. wikipedia, reddit)")
    parser.add_argument("--min_c1", type=int, default=5, help="Minimum 1-hop layer capacity")
    parser.add_argument("--max_c1", type=int, default=50, help="Maximum 1-hop layer capacity")
    parser.add_argument("--steps", type=int, default=10, help="Number of linear steps in sweep")
    parser.add_argument("--branching_factor", type=float, default=None, help="Empirical M branching factor (auto-computed if None)")
    parser.add_argument("--ttl", type=float, default=86400.0, help="Static TTL eviction window in seconds (default: 86400)")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for interaction replay")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluation frequency (evaluates every N batches)")
    parser.add_argument("--output_dir", type=str, default="sweep_results", help="Directory for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"Loading dataset: {args.data}")
    print(f"========================================================")
    _, _, _, train_data, _, _, _, _ = get_data(args.data)

    # 1. Determine Hop-2 Branching Multiplier M
    if args.branching_factor is not None:
        m = args.branching_factor
        print(f"Using user-specified Branching Multiplier M: {m:.2f}x")
    else:
        print(f"Profiling empirical Hop-2 Branching Multiplier (TTL = {args.ttl}s)...")
        m = compute_branching_factor(train_data, ttl_window=args.ttl)
        print(f"Empirical Branching Multiplier M: {m:.2f}x")

    # 2. Configure linear sweep space
    c1_space = np.unique(np.linspace(args.min_c1, args.max_c1, args.steps, dtype=int))
    print(f"Sweeping C1 across {len(c1_space)} points: {c1_space.tolist()}")

    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

    sweep_records = []
    print(f"\n{'C1':<5} | {'C2':<6} | {'IoU':<7} | {'F2':<7} | {'Precision':<10} | {'Recall':<8} | {'Hit Rate':<9} | {'Lat (ms)':<8}")
    print("-" * 75)

    for c1 in c1_space:
        c1_int = int(c1)
        c2_int = max(1, int(math.ceil(c1_int * m)))

        metrics = evaluate_capacity_step(
            train_data=train_data,
            train_sampler=train_sampler,
            base_finder=base_finder,
            c1=c1_int,
            c2=c2_int,
            ttl=args.ttl,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq
        )

        record = {
            "c1": c1_int,
            "c2": c2_int,
            "branching_multiplier": m,
            "iou": metrics['iou'],
            "f2": metrics['f2'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "cache_hit_rate": metrics['cache_hit_rate'],
            "latency_ms": metrics['latency_ms'],
            "cache_hits": metrics['cache_hits'],
            "cache_misses": metrics['cache_misses']
        }
        sweep_records.append(record)

        print(
            f"{c1_int:<5} | {c2_int:<6} | {metrics['iou']:.4f}  | {metrics['f2']:.4f}  | "
            f"{metrics['precision']:.4f}     | {metrics['recall']:.4f}   | "
            f"{metrics['cache_hit_rate'] * 100:.2f}%    | {metrics['latency_ms']:.4f}"
        )

    # 3. Export JSON Results
    json_path = output_dir / f"{args.data}_capacity_sweep_2.json"
    payload = {
        "dataset": args.data,
        "ttl_seconds": args.ttl,
        "branching_factor_M": m,
        "c1_min": args.min_c1,
        "c1_max": args.max_c1,
        "steps": args.steps,
        "sweep_records": sweep_records
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"\nSaved structured sweep JSON to: {json_path}")

    # 4. Generate & Save Progression Plots
    generate_plots(
        results=sweep_records,
        dataset_name=args.data,
        branching_factor=m,
        ttl_seconds=args.ttl,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()