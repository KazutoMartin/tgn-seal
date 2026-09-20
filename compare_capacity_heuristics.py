import numpy as np
import math
import time
import random
from tqdm import tqdm

from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, MultiLayerTemporalCache

def calculate_set_metrics(base_list, cache_list):
    base_set = set(base_list)
    cache_set = set(cache_list)
    
    intersect = len(base_set.intersection(cache_set))
    union = len(base_set.union(cache_set))
    
    recall = intersect / len(base_set) if len(base_set) > 0 else 1.0
    precision = intersect / len(cache_set) if len(cache_set) > 0 else 1.0
    iou = intersect / union if union > 0 else 1.0
    
    return recall, precision, iou

def profile_dataset(train_data, ttl_window=86400):
    """Calculates semantic capacity, raw volume, and empirical branching multiplier."""
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    unique_nodes = np.unique(np.concatenate([sources, destinations]))

    # 1. Semantic Capacity & Raw Interaction Volume
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
            while node_edges[right][0] - node_edges[left][0] > ttl_window:
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

    # 2. Empirical Hop-2 Branching Multiplier
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

    return {
        'p95_unique': p95_unique,
        'p99_unique': p99_unique,
        'p95_total': p95_total,
        'multiplier': multiplier
    }

def evaluate_capacity_configuration(train_data, train_sampler, base_finder, c1, c2, batch_size=200, sample_ratio=1):
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    layer_finder.cache = MultiLayerTemporalCache(max_edges_per_hop={1: c1, 2: c2})

    metrics = {'recall': [], 'prec': [], 'iou': [], 'latency_ms': []}
    num_instance = len(train_data.sources)
    num_batch = math.ceil((num_instance * sample_ratio) / batch_size)

    for k in range(num_batch):
        start_idx = k * batch_size
        end_idx = min(num_instance, start_idx + batch_size)

        sources = train_data.sources[start_idx:end_idx]
        destinations = train_data.destinations[start_idx:end_idx]
        timestamps = train_data.timestamps[start_idx:end_idx]
        edge_idxs = train_data.edge_idxs[start_idx:end_idx]

        size = len(sources)
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
                metrics['recall'].append(r)
                metrics['prec'].append(p)
                metrics['iou'].append(iou)

        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder)

    return {k: float(np.mean(v)) for k, v in metrics.items()}

def main():
    datasets = [
        "email-Eu-core-temporal-Dept1",
        "email-Eu-core-temporal-Dept2",
        "email-Eu-core-temporal-Dept3",
        "email-Eu-core-temporal-Dept4",
        "calls",
        "CollegeMsg"
    ]

    heuristic_scores = {
        'Arithmetic Mean (Midpoint)': {'iou': [], 'latency': [], 'rec': [], 'prec': []},
        'P99 Semantic Bound': {'iou': [], 'latency': [], 'rec': [], 'prec': []},
        'Geometric Mean': {'iou': [], 'latency': [], 'rec': [], 'prec': []},
        'Adaptive Max (Burst-Aware)': {'iou': [], 'latency': [], 'rec': [], 'prec': []}
    }

    for d_name in datasets:
        print(f"\n==================================================")
        print(f"Profiling Dataset: {d_name}")
        print(f"==================================================")
        _, _, _, train_data, _, _, _, _ = get_data(d_name)
        base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
        train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

        params = profile_dataset(train_data)
        p95_u = params['p95_unique']
        p99_u = params['p99_unique']
        p95_t = params['p95_total']
        mult = params['multiplier']

        print(f"Topology: P95(Unique)={p95_u:.1f} | P99(Unique)={p99_u:.1f} | P95(Total)={p95_t:.1f} | Branching={mult:.2f}x")

        # Candidate capacity calculations
        c1_midpoint = max(1, int(round((p95_u + p95_t) / 2)))
        c1_p99 = max(1, int(round(p99_u)))
        c1_geom = max(1, int(round(math.sqrt(p95_u * p95_t))))
        
        # New Formula: The Burstiness-Aware Capacity
        c1_adaptive = max(c1_p99, c1_midpoint)

        candidates = [
            ('Arithmetic Mean (Midpoint)', c1_midpoint, max(1, int(round(c1_midpoint * mult)))),
            ('P99 Semantic Bound', c1_p99, max(1, int(round(c1_p99 * mult)))),
            ('Geometric Mean', c1_geom, max(1, int(round(c1_geom * mult)))),
            ('Adaptive Max (Burst-Aware)', c1_adaptive, max(1, int(round(c1_adaptive * mult))))
        ]

        print(f"\n{'Heuristic':<28} | {'C1':<3} | {'C2':<4} | {'Recall':<6} | {'Prec':<6} | {'IoU':<6} | {'Lat(ms)':<7}")
        print("-" * 75)

        for name, c1, c2 in candidates:
            res = evaluate_capacity_configuration(train_data, train_sampler, base_finder, c1, c2)
            heuristic_scores[name]['iou'].append(res['iou'])
            heuristic_scores[name]['latency'].append(res['latency_ms'])
            heuristic_scores[name]['rec'].append(res['recall'])
            heuristic_scores[name]['prec'].append(res['prec'])
            print(f"{name:<28} | {c1:<3} | {c2:<4} | {res['recall']:.4f} | {res['prec']:.4f} | {res['iou']:.4f} | {res['latency_ms']:.4f}")

    print("\n\n" + "#" * 60)
    print("FINAL BENCHMARK SCOREBOARD (Average across all datasets)")
    print("#" * 60)
    print(f"{'Heuristic Strategy':<28} | {'Mean IoU':<8} | {'Mean Recall':<11} | {'Mean Prec':<9} | {'Mean Latency':<12}")
    print("-" * 78)

    ranked_results = []
    for name, data in heuristic_scores.items():
        ranked_results.append((
            name,
            np.mean(data['iou']),
            np.mean(data['rec']),
            np.mean(data['prec']),
            np.mean(data['latency'])
        ))

    ranked_results.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, m_iou, m_rec, m_prec, m_lat) in enumerate(ranked_results, 1):
        marker = "🏆 BEST" if rank == 1 else f"#{rank}"
        print(f"{name:<28} | {m_iou:.4f}   | {m_rec:.4f}      | {m_prec:.4f}    | {m_lat:.4f} ms  [{marker}]")

if __name__ == "__main__":
    main()