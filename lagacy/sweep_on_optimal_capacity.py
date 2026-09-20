import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from pathlib import Path

# Import data processing and utilities based on the existing project structure
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, MultiLayerTemporalCache
from compare_capacity_heuristics import calculate_set_metrics

def calculate_dynamic_ttl(train_data, target_percentile=90, max_reply_days=3):
    """
    Measures the empirical Time-To-Return (TTR) for recurring interactions.
    Enforces a conversational cutoff to distinguish active replies from new, delayed conversations.
    """
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    
    interaction_history = {}
    return_delays = []
    max_reply_seconds = max_reply_days * 86400.0
    
    for s, d, t in zip(sources, destinations, timestamps):
        edge = tuple(sorted((int(s), int(d))))
        
        if edge in interaction_history:
            delay = t - interaction_history[edge]
            if 0 < delay <= max_reply_seconds:
                return_delays.append(delay)
                
        interaction_history[edge] = t
        
    if not return_delays:
        return 86400.0
        
    dynamic_ttl = float(np.percentile(return_delays, target_percentile))
    return dynamic_ttl

def profile_burst_topology(train_data, burst_window=7200):
    """
    Profiles the dataset using a tightly scoped sliding window (e.g., 2 hours)
    to calculate the maximum concurrent interactions a node experiences within a dense burst.
    """
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    unique_nodes = np.unique(np.concatenate([sources, destinations]))

    max_total_list, max_unique_list = [], []
    for node in unique_nodes:
        node_mask = (sources == node) | (destinations == node)
        node_edges = []
        for s, d, t in zip(sources[node_mask], destinations[node_mask], timestamps[node_mask]):
            node_edges.append((t, d if s == node else s))
        node_edges.sort(key=lambda x: x[0])
        if not node_edges:
            continue

        left = 0
        max_total, max_unique = 0, 0
        for right in range(len(node_edges)):
            # Strictly use the burst_window to define structural capacity
            while node_edges[right][0] - node_edges[left][0] > burst_window:
                left += 1
            cur_window = node_edges[left:right+1]
            tot = len(cur_window)
            uniq = len(set(n for _, n in cur_window))
            if tot > max_total: max_total = tot
            if uniq > max_unique: max_unique = uniq

        max_total_list.append(max_total)
        max_unique_list.append(max_unique)

    p95_total = np.percentile(max_total_list, 95)
    p95_unique = np.percentile(max_unique_list, 95)
    p99_unique = np.percentile(max_unique_list, 99)

    # Calculate Branching Multiplier within the same burst window
    max_node = max(sources.max(), destinations.max())
    adj_list = [[] for _ in range(max_node + 1)]
    for s, d, t in zip(sources, destinations, timestamps):
        adj_list[s].append((d, t))
        adj_list[d].append((s, t))

    h1_vols, h2_vols = [], []
    for node in unique_nodes:
        interactions = adj_list[node]
        if not interactions:
            continue
        sample_times = [t for _, t in interactions[::max(1, len(interactions)//50)]]
        for q_time in sample_times:
            h1_nodes = set()
            h1_cnt = 0
            for neighbor, t in interactions:
                if q_time - burst_window <= t <= q_time:
                    h1_nodes.add(neighbor)
                    h1_cnt += 1
            if h1_cnt == 0:
                continue

            h2_cnt = 0
            for h1_n in h1_nodes:
                for h2_n, t2 in adj_list[h1_n]:
                    if q_time - burst_window <= t2 <= q_time and h2_n != node:
                        h2_cnt += 1
            h1_vols.append(h1_cnt)
            h2_vols.append(h2_cnt)

    h1_p95 = np.percentile(h1_vols, 95) if h1_vols else 1.0
    h2_p95 = np.percentile(h2_vols, 95) if h2_vols else 1.0
    multiplier = (h2_p95 / h1_p95) if h1_p95 > 0 else 3.0

    return {
        'p95_unique': p95_unique,
        'p99_unique': p99_unique,
        'p95_total': p95_total,
        'multiplier': multiplier
    }

def sweep_evaluate_capacity(train_data, train_sampler, base_finder, c1, c2, dynamic_ttl, batch_size=200, eval_freq=4):
    """
    Decouples state updates from evaluation and calculates both IoU and F2 score.
    """
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    # The cache uses the long conversational dynamic_ttl for EVICTION
    layer_finder.cache = MultiLayerTemporalCache(
        ttl_window=dynamic_ttl,
        max_edges_per_hop={1: c1, 2: c2}
    )

    metrics = {'recall': [], 'prec': [], 'iou': [], 'f2': [], 'latency_ms': []}
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

            base_pos = base_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=False)
            random.seed(42)
            base_neg = base_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=False)

            start_time = time.perf_counter()
            layer_pos = layer_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=True)
            random.seed(42)
            layer_neg = layer_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=True)

            batch_latency_ms = (time.perf_counter() - start_time) * 1000 / (size * 2)
            metrics['latency_ms'].append(batch_latency_ms)

            for i in range(size):
                for b_data, l_data in [(base_pos[i], layer_pos[i]), (base_neg[i], layer_neg[i])]:
                    b_edges, l_edges = b_data.edge_time.tolist(), l_data.edge_time.tolist()
                    r, p, iou = calculate_set_metrics(b_edges, l_edges)
                    
                    f2 = (5 * p * r) / ((4 * p) + r) if (p + r) > 0 else 0.0
                    
                    metrics['recall'].append(r)
                    metrics['prec'].append(p)
                    metrics['iou'].append(iou)
                    metrics['f2'].append(f2)

        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder)

    return {k: float(np.mean(v)) for k, v in metrics.items()}

def main():
    parser = argparse.ArgumentParser(description="Linear Sweep with Decoupled Sizing & Eviction")
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--radius", type=int, default=15, help="Linear sweep radius around optimal C1")
    parser.add_argument("--steps", type=int, default=9, help="Number of linear steps")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate 1 in every N batches")
    parser.add_argument("--burst_window", type=int, default=7200, help="2-hour window (in seconds) for structural profiling")
    args = parser.parse_args()

    print(f"Loading data for dataset: {args.data}")
    _, _, _, train_data, _, _, _, _ = get_data(args.data)
    
    # 1. Compute conversational eviction window (TTL)
    dynamic_ttl = calculate_dynamic_ttl(train_data, target_percentile=90, max_reply_days=3)
    print(f"Empirical Eviction Window (TTL): {dynamic_ttl:.1f} seconds ({dynamic_ttl / 3600:.2f} hours)")

    # 2. Compute structural capacity sizing strictly on the 2-hour Burst Window
    print(f"Profiling dataset burst topology (Burst Window: {args.burst_window/3600:.1f} hours)...")
    params = profile_burst_topology(train_data, burst_window=dynamic_ttl)
    p95_u = params['p95_unique']
    p99_u = params['p99_unique']
    p95_t = params['p95_total']
    mult = params['multiplier']
    
    # Compute the decoupled sizing candidates
    c1_midpoint = max(1, int(round((p95_u + p95_t) / 2)))
    opt1_c1 = max(int(round(p99_u)), c1_midpoint)
    opt2_c1 = max(int(round(p99_u)), int(round(p95_t)))
    
    print(f"Topological Metrics -> P95_u: {p95_u:.1f} | P99_u: {p99_u:.1f} | P95_t: {p95_t:.1f} | M: {mult:.2f}x")
    print(f"Original Burst-Aware Optimum -> C1: {opt1_c1}")
    print(f"New Upper-Tail Optimum       -> C1: {opt2_c1}")
    
    start_c1 = max(1, min(opt1_c1, opt2_c1) - args.radius)
    end_c1 = max(opt1_c1, opt2_c1) + args.radius
    c1_sweep = np.unique(np.linspace(start_c1, end_c1, args.steps, dtype=int))
    
    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    
    results = []
    print(f"Starting sweep over C1: {c1_sweep.tolist()}")
    
    for c1 in c1_sweep:
        c1_int = int(c1)
        c2 = max(1, int(round(c1_int * mult)))
        print(f"  Evaluating C1={c1_int}, C2={c2}...")
        
        metrics = sweep_evaluate_capacity(
            train_data, train_sampler, base_finder, 
            c1=c1_int, c2=c2, dynamic_ttl=dynamic_ttl, eval_freq=args.eval_freq
        )
        
        res = {
            "c1": c1_int,
            "c2": c2,
            "recall": metrics['recall'],
            "precision": metrics['prec'],
            "iou": metrics['iou'],
            "f2": metrics['f2'],
            "latency_ms": metrics['latency_ms']
        }
        results.append(res)
        print(f"    -> IoU: {res['iou']:.4f} | F2: {res['f2']:.4f} | Latency: {res['latency_ms']:.4f} ms")
        
    output_dir = Path("results/capacity_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.data}_decoupled_sweep_results.json"
    
    with open(json_path, 'w') as f:
        json.dump({
            "dataset": args.data,
            "eviction_ttl_seconds": dynamic_ttl,
            "burst_window_seconds": args.burst_window,
            "opt1_c1_original": int(opt1_c1), 
            "opt2_c1_upper_bound": int(opt2_c1),
            "branching_factor_M": float(mult),
            "sweep_data": results
        }, f, indent=4)
        
    print(f"\nSaved results to {json_path}")
    generate_plots(results, opt1_c1, opt2_c1, dynamic_ttl, args.burst_window, args.data, output_dir)

def generate_plots(results, opt1_c1, opt2_c1, dynamic_ttl, burst_window, dataset_name, output_dir):
    c1_vals = [r['c1'] for r in results]
    ious = [r['iou'] for r in results]
    f2s = [r['f2'] for r in results]
    recalls = [r['recall'] for r in results]
    precs = [r['precision'] for r in results]
    lats = [r['latency_ms'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Multi-Layer Cache Capacity: Decoupled Sizing & Eviction\nDataset: {dataset_name} | Eviction TTL: {dynamic_ttl/3600:.1f}h | Sizing Burst: {burst_window/3600:.1f}h", fontsize=16)
    
    def mark_optimal(ax, metric_name):
        ax.axvline(x=opt1_c1, color='red', linestyle='--', linewidth=1.5, label=f'Orig. Opt ($C_1={opt1_c1}$)')
        ax.axvline(x=opt2_c1, color='blue', linestyle='-.', linewidth=1.5, label=f'Upper-Tail Opt ($C_1={opt2_c1}$)')
        ax.set_xlabel('$C_1$ (Hop 1 Capacity)')
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        
    axes[0, 0].plot(c1_vals, ious, marker='o', color='purple', linewidth=2)
    axes[0, 0].set_title('Intersection over Union (IoU)')
    mark_optimal(axes[0, 0], 'IoU')
    
    axes[0, 1].plot(c1_vals, f2s, marker='o', color='teal', linewidth=2)
    axes[0, 1].set_title('F2 Score (Recall-Weighted)')
    mark_optimal(axes[0, 1], 'F2 Score')

    axes[0, 2].plot(c1_vals, lats, marker='o', color='orange', linewidth=2)
    axes[0, 2].set_title('Extraction Latency Penalty')
    mark_optimal(axes[0, 2], 'Mean Latency (ms)')
    
    axes[1, 0].plot(c1_vals, recalls, marker='o', color='blue', linewidth=2)
    axes[1, 0].set_title('Edge Recall')
    mark_optimal(axes[1, 0], 'Recall')
    
    axes[1, 1].plot(c1_vals, precs, marker='o', color='green', linewidth=2)
    axes[1, 1].set_title('Edge Precision')
    mark_optimal(axes[1, 1], 'Precision')
    
    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    plot_path = output_dir / f"{dataset_name}_decoupled_plots.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved visual plots to {plot_path}")

if __name__ == "__main__":
    main()