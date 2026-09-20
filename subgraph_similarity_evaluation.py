import numpy as np
import math
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt

# Assuming these are available in your PYTHONPATH
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler, MultiLayerTemporalCache

def calculate_set_metrics(base_list, cache_list):
    """Calculates Recall, Precision, and Jaccard IoU for two lists."""
    base_set = set(base_list)
    cache_set = set(cache_list)
    
    intersect = len(base_set.intersection(cache_set))
    union = len(base_set.union(cache_set))
    
    recall = intersect / len(base_set) if len(base_set) > 0 else 1.0
    precision = intersect / len(cache_set) if len(cache_set) > 0 else 1.0
    iou = intersect / union if union > 0 else 1.0
    
    return recall, precision, iou

def evaluate_capacity(dataset_name, train_data, train_sampler, base_finder, capacity_c, batch_size=200):
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    
    # Dynamically inject the custom capacity limit
    hop_capacities = {1: capacity_c, 2: capacity_c * 3}
    layer_finder.cache = MultiLayerTemporalCache(max_edges_per_hop=hop_capacities)
    
    metrics = {
        'layer_edge_recall': [], 'layer_edge_prec': [], 'layer_edge_iou': [], 'latency_ms': []
    }
    
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    
    for k in tqdm(range(num_batch), desc=f"Eval Capacity C={capacity_c}", leave=False):
        start_idx = k * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        
        sources = train_data.sources[start_idx:end_idx]
        destinations = train_data.destinations[start_idx:end_idx]
        timestamps = train_data.timestamps[start_idx:end_idx]
        edge_idxs = train_data.edge_idxs[start_idx:end_idx]
        
        size = len(sources)
        _, negatives = train_sampler.sample(size)
        
        # 1. Baseline Extraction
        base_pos = base_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=False)
        random.seed(42)
        base_neg = base_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=False)
        
        # 2. Layer Cache Extraction
        start_time = time.perf_counter()
        layer_pos = layer_finder.extract_enclosing_subgraph(sources, destinations, timestamps, y=1, use_cache=True)
        random.seed(42)
        layer_neg = layer_finder.extract_enclosing_subgraph(sources, negatives, timestamps, y=0, use_cache=True)
        batch_latency_ms = (time.perf_counter() - start_time) * 1000 / (size * 2)
        metrics['latency_ms'].append(batch_latency_ms)
        
        # 3. Calculate Edge Metrics
        for i in range(size):
            for b_data, l_data in [(base_pos[i], layer_pos[i]), (base_neg[i], layer_neg[i])]:
                b_edges, l_edges = b_data.edge_time.tolist(), l_data.edge_time.tolist()
                
                e_rec, e_prec, e_iou = calculate_set_metrics(b_edges, l_edges)
                metrics['layer_edge_recall'].append(e_rec)
                metrics['layer_edge_prec'].append(e_prec)
                metrics['layer_edge_iou'].append(e_iou)

        # 4. Push Step
        for s, d, t, e_idx in zip(sources, destinations, timestamps, edge_idxs):
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder)

    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    dataset_name = "email-Eu-core-temporal-Dept4"
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    
    # Fine-grained linear sweep in the critical plateau region
    capacities = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    
    results = {'recall': [], 'precision': [], 'iou': [], 'latency': []}
    
    print(f"Starting Fine-Grained Sweep for {dataset_name}...")
    for c in capacities:
        res = evaluate_capacity(dataset_name, train_data, train_sampler, base_finder, c)
        results['recall'].append(res['layer_edge_recall'])
        results['precision'].append(res['layer_edge_prec'])
        results['iou'].append(res['layer_edge_iou'])
        results['latency'].append(res['latency_ms'])
        print(f"C={c:2d} | Rec: {res['layer_edge_recall']:.4f} | Prec: {res['layer_edge_prec']:.4f} | IoU: {res['layer_edge_iou']:.4f} | Latency: {res['latency_ms']:.4f}ms")

    x_data = np.array(capacities)
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Recall vs Precision Trade-off
    plt.subplot(1, 2, 1)
    plt.plot(x_data, results['recall'], marker='o', color='green', label='Edge Recall (Missing Data)')
    plt.plot(x_data, results['precision'], marker='s', color='red', label='Edge Precision (Stale Data)')
    plt.plot(x_data, results['iou'], marker='^', color='blue', linestyle='--', label='Jaccard IoU (Combined)')
    
    plt.title("Subgraph Fidelity: Recall vs Precision Trade-off")
    plt.xlabel("Base Capacity (Edges per Hop 1)")
    plt.ylabel("Score [0.0 - 1.0]")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Subplot 2: The Latency/Fidelity Pareto Frontier
    plt.subplot(1, 2, 2)
    plt.plot(results['latency'], results['iou'], marker='o', color='purple', linestyle='-')
    for i, txt in enumerate(capacities):
        plt.annotate(f"c={txt}", (results['latency'][i], results['iou'][i]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.title("Fine-Grained Computational Pareto Frontier")
    plt.xlabel("Mean Extraction Latency (ms / subgraph)")
    plt.ylabel("Edge Jaccard Similarity (IoU)")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("fine_grained_capacity_optimization.png", dpi=300)
    print("\nHigh-resolution optimization plots saved to 'fine_grained_capacity_optimization.png'")

if __name__ == "__main__":
    main()