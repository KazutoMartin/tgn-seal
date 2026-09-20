import numpy as np
from utils.data_processing import get_data

def verify_theoretical_capacity(dataset_name="email-Eu-core-temporal-Dept4", ttl_window=86400):
    # Load the exact training split
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    
    unique_nodes = np.unique(np.concatenate([sources, destinations]))
    max_edges_per_window = []
    
    print(f"Scanning temporal windows for {len(unique_nodes)} nodes...")
    
    for node in unique_nodes:
        # Extract all chronological timestamps involving this specific node
        node_mask = (sources == node) | (destinations == node)
        node_times = np.sort(timestamps[node_mask])
        
        if len(node_times) == 0:
            continue
            
        # Slide a time window (W) over the node's history to find peak density
        left = 0
        max_in_window = 0
        
        for right in range(len(node_times)):
            # Advance the left pointer until the window is within the TTL
            while node_times[right] - node_times[left] > ttl_window:
                left += 1
            
            current_window_size = right - left + 1
            if current_window_size > max_in_window:
                max_in_window = current_window_size
                
        max_edges_per_window.append(max_in_window)
        
    # Calculate Queueing Theory Parameters
    mean_edges = np.mean(max_edges_per_window)
    p95_edges = np.percentile(max_edges_per_window, 95)
    p99_edges = np.percentile(max_edges_per_window, 99)
    absolute_max = np.max(max_edges_per_window)
    
    print(f"\n--- Theoretical Capacity Analysis: {dataset_name} ---")
    print(f"TTL Window (W): {ttl_window} seconds")
    print(f"Average Active Edges per Window (d_act): {mean_edges:.2f}")
    print(f"95th Percentile (λ_95 * W): {p95_edges:.2f}")
    print(f"99th Percentile: {p99_edges:.2f}")
    print(f"Absolute Maximum Burst: {absolute_max}")
    print("\nConclusion: The theoretical C_opt should closely match the 95th/99th percentile.")

if __name__ == "__main__":
    verify_theoretical_capacity(dataset_name="email-Eu-core-temporal-Dept4", ttl_window=86400)