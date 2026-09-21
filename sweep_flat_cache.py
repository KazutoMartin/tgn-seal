import argparse
import json
import math
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, TemporalSubgraphCache


def calculate_set_metrics(base_list, cache_list):
    base_set = set(base_list)
    cache_set = set(cache_list)

    intersect = len(base_set.intersection(cache_set))
    union = len(base_set.union(cache_set))

    recall = intersect / len(base_set) if len(base_set) > 0 else 1.0
    precision = intersect / len(cache_set) if len(cache_set) > 0 else 1.0
    iou = intersect / union if union > 0 else 1.0

    return recall, precision, iou


def evaluate_capacity_step(train_data, train_sampler, base_finder, c_max, ttl, batch_size=200, eval_freq=1):
    """
    Simulates streaming training over the dataset, querying the flat cache
    and comparing extracted subgraphs against the baseline un-cached ground truth.
    """
    flat_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    # Manually configure the TemporalSubgraphCache with targeted capacity and TTL
    flat_finder.cache = TemporalSubgraphCache(
        ttl_window=ttl,
        max_edges=c_max
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
            flat_pos = flat_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=True)
            random.seed(42)
            flat_neg = flat_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=True)
            batch_latency_ms = (time.perf_counter() - start_time) * 1000.0 / (size * 2)
            metrics['latency_ms'].append(batch_latency_ms)

            # Structural fidelity metrics
            for i in range(size):
                for b_data, f_data in [(base_pos[i], flat_pos[i]), (base_neg[i], flat_neg[i])]:
                    b_edges = b_data.edge_time.tolist()
                    f_edges = f_data.edge_time.tolist()
                    r, p, iou = calculate_set_metrics(b_edges, f_edges)

                    # Recall-heavy F2 score
                    f2 = (5.0 * p * r) / ((4.0 * p) + r) if (4.0 * p + r) > 0 else 0.0

                    metrics['recall'].append(r)
                    metrics['precision'].append(p)
                    metrics['iou'].append(iou)
                    metrics['f2'].append(f2)

        # Stream interaction edges into cache
        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            flat_finder.cache.push_edge(s, d, t, e_idx, flat_finder)

    hits = flat_finder.cache.cache_hits
    misses = flat_finder.cache.cache_misses
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


def generate_plots(results, dataset_name, ttl_seconds, output_dir):
    c_vals = [r['c_max'] for r in results]
    ious = [r['iou'] for r in results]
    f2s = [r['f2'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    hit_rates = [r['cache_hit_rate'] * 100.0 for r in results]
    latencies = [r['latency_ms'] for r in results]

    # Optimal C_max point selected via max F2 score (recall-weighted)
    best_idx = int(np.argmax(f2s))
    best_c = c_vals[best_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Flat Temporal Cache Capacity Sweep\n"
        f"Dataset: {dataset_name} | Static TTL: {ttl_seconds}s (1 day)\n"
        f"Recommended Optimum: C_max={best_c} (Peak F2={f2s[best_idx]:.4f})",
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
        ax.plot(c_vals, data, marker='o', color=color, linewidth=2, markersize=5)
        ax.axvline(x=best_c, color='red', linestyle='--', alpha=0.75, label=f'Best F2 ($C_{{max}}={best_c}$)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('$C_{max}$ (Max Edges Capacity)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')

    plt.tight_layout()
    plot_path = output_dir / f"{dataset_name}_flat_capacity_sweep_1.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved visualization grid to: {plot_path}")


def main():
    parser = argparse.ArgumentParser("Linear Parameter Sweep for Flat Cache Capacity")
    parser.add_argument("-d", "--data", type=str, default="wikipedia", help="Dataset name (e.g. wikipedia, reddit)")
    parser.add_argument("--min_c", type=int, default=50, help="Minimum flat edge capacity")
    parser.add_argument("--max_c", type=int, default=300, help="Maximum flat edge capacity")
    parser.add_argument("--steps", type=int, default=10, help="Number of linear steps in sweep")
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

    # 1. Configure linear sweep space
    c_space = np.unique(np.linspace(args.min_c, args.max_c, args.steps, dtype=int))
    print(f"Sweeping C_max across {len(c_space)} points: {c_space.tolist()}")

    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

    sweep_records = []
    print(f"\n{'C_max':<6} | {'IoU':<7} | {'F2':<7} | {'Precision':<10} | {'Recall':<8} | {'Hit Rate':<9} | {'Lat (ms)':<8}")
    print("-" * 70)

    for c_max in c_space:
        c_int = int(c_max)

        metrics = evaluate_capacity_step(
            train_data=train_data,
            train_sampler=train_sampler,
            base_finder=base_finder,
            c_max=c_int,
            ttl=args.ttl,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq
        )

        record = {
            "c_max": c_int,
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
            f"{c_int:<6} | {metrics['iou']:.4f}  | {metrics['f2']:.4f}  | "
            f"{metrics['precision']:.4f}     | {metrics['recall']:.4f}   | "
            f"{metrics['cache_hit_rate'] * 100:.2f}%    | {metrics['latency_ms']:.4f}"
        )

    # 2. Export JSON Results
    json_path = output_dir / f"{args.data}_flat_capacity_sweep_1.json"
    payload = {
        "dataset": args.data,
        "ttl_seconds": args.ttl,
        "c_min": args.min_c,
        "c_max": args.max_c,
        "steps": args.steps,
        "sweep_records": sweep_records
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"\nSaved structured sweep JSON to: {json_path}")

    # 3. Generate & Save Progression Plots
    generate_plots(
        results=sweep_records,
        dataset_name=args.data,
        ttl_seconds=args.ttl,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()