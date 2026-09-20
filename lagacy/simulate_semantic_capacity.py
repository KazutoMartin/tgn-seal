import numpy as np
from tqdm import tqdm
from utils.data_processing import get_data

def simulate_semantic_capacity(dataset_name="email-Eu-core-temporal-Dept4", ttl_window=86400):
    # Load the exact training split to match the queueing theory verify script
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    
    unique_nodes = np.unique(np.concatenate([sources, destinations]))
    
    max_total_edges_list = []
    max_unique_neighbors_list = []
    
    print(f"Simulating Information Entropy for {len(unique_nodes)} nodes...")
    
    for node in tqdm(unique_nodes, desc="Scanning Temporal Windows"):
        # Extract all chronological timestamps involving this specific node
        node_mask = (sources == node) | (destinations == node)
        
        # Build chronological interaction pairs (timestamp, interacting_neighbor)
        node_edges = []
        for s, d, t in zip(sources[node_mask], destinations[node_mask], timestamps[node_mask]):
            neighbor = d if s == node else s
            node_edges.append((t, neighbor))
            
        node_edges.sort(key=lambda x: x[0]) 
        
        if len(node_edges) == 0:
            continue
            
        left = 0
        max_total = 0
        max_unique = 0
        
        # Slide a time window (W) over the node's history
        for right in range(len(node_edges)):
            # Advance the left pointer until the window is within the TTL
            while node_edges[right][0] - node_edges[left][0] > ttl_window:
                left += 1
            
            current_window = node_edges[left:right+1]
            
            # Hypothesis Metrics: Raw Volume vs Semantic Diversity
            total_edges = len(current_window)
            unique_neighbors = len(set(neighbor for _, neighbor in current_window))
            
            if total_edges > max_total:
                max_total = total_edges
            if unique_neighbors > max_unique:
                max_unique = unique_neighbors
                
        max_total_edges_list.append(max_total)
        max_unique_neighbors_list.append(max_unique)
        
    # Calculate Queueing Theory Parameters
    p95_total = np.percentile(max_total_edges_list, 95)
    p99_total = np.percentile(max_total_edges_list, 99)
    
    p95_unique = np.percentile(max_unique_neighbors_list, 95)
    p99_unique = np.percentile(max_unique_neighbors_list, 99)
    
    print(f"\n--- Information Entropy & Semantic Diversity ---")
    print(f"Dataset: {dataset_name} | TTL Window (W): {ttl_window}s")
    print(f"95th Percentile - Total Active Edges (Raw Volume): {p95_total:.2f}")
    print(f"95th Percentile - Unique Neighbors (Semantic Capacity): {p95_unique:.2f}")
    print(f"99th Percentile - Total Active Edges: {p99_total:.2f}")
    print(f"99th Percentile - Unique Neighbors: {p99_unique:.2f}")
    
    print("\n--- Theoretical Conclusion ---")
    print("If P95(Unique) matches the empirical Pareto optimum (C=30), it proves the cache")
    print("achieves structural parity natively. The fixed FIFO capacity organically acts as")
    print("a high-pass filter, dropping repetitive edge bursts while retaining structural breadth.")

if __name__ == "__main__":
    simulate_semantic_capacity()