import numpy as np
from tqdm import tqdm
from utils.data_processing import get_data

def simulate_hop2_branching(dataset_name="email-Eu-core-temporal-Dept4", ttl_window=86400):
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    
    sources = train_data.sources
    destinations = train_data.destinations
    timestamps = train_data.timestamps
    
    # Pre-build an adjacency list for fast temporal lookups
    max_node = max(sources.max(), destinations.max())
    adj_list = [[] for _ in range(max_node + 1)]
    for s, d, t in zip(sources, destinations, timestamps):
        adj_list[s].append((d, t))
        adj_list[d].append((s, t))
        
    unique_nodes = np.unique(np.concatenate([sources, destinations]))
    
    hop1_active_volumes = []
    hop2_active_volumes = []
    
    for node in tqdm(unique_nodes, desc="Measuring Effective Branching Factor"):
        node_interactions = adj_list[node]
        if not node_interactions: continue
            
        # Check a sample of random timestamps for this node to find average window densities
        sample_times = [t for _, t in node_interactions[::max(1, len(node_interactions)//50)]]
        
        for query_time in sample_times:
            # 1. Find Hop 1 volume within TTL
            hop1_nodes = set()
            hop1_count = 0
            for neighbor, t in node_interactions:
                if query_time - ttl_window <= t <= query_time:
                    hop1_nodes.add(neighbor)
                    hop1_count += 1
            
            if hop1_count == 0: continue
                
            # 2. Find Hop 2 volume within TTL
            hop2_count = 0
            for h1_node in hop1_nodes:
                for h2_neighbor, t2 in adj_list[h1_node]:
                    if query_time - ttl_window <= t2 <= query_time and h2_neighbor != node:
                        hop2_count += 1
                        
            hop1_active_volumes.append(hop1_count)
            hop2_active_volumes.append(hop2_count)

    h1_p95 = np.percentile(hop1_active_volumes, 95)
    h2_p95 = np.percentile(hop2_active_volumes, 95)
    
    multiplier = h2_p95 / h1_p95 if h1_p95 > 0 else 0
    
    print("\n--- Effective Temporal Branching Factor ---")
    print(f"P95 Active Hop 1 Edges: {h1_p95:.2f}")
    print(f"P95 Active Hop 2 Edges: {h2_p95:.2f}")
    print(f"\nEmpirical Multiplier: {multiplier:.2f}x")
    print(f"Update evaluation script to: hop_capacities = {{1: capacity_c, 2: int(capacity_c * {multiplier:.2f})}}")

if __name__ == "__main__":
    simulate_hop2_branching()