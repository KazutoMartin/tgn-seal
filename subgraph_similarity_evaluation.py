import numpy as np
import math
from tqdm import tqdm
import random

# Assuming these are available in your PYTHONPATH
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler

def evaluate_exact_training_fidelity(dataset_name="email-Eu-core-temporal-Dept1", batch_size=200):
    # 1. Load data and isolate the training split exactly like the training script
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name) #[cite: 2]
    
    # 2. Initialize independent finders
    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False) #[cite: 2]
    flat_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    
    # 3. Initialize the Negative Sampler 
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations) #[cite: 2]
    
    metrics = {
        'flat_node_recall': [], 'flat_edge_recall': [],
        'layer_node_recall': [], 'layer_edge_recall': []
    }
    
    num_instance = len(train_data.sources) #[cite: 2]
    num_batch = math.ceil(num_instance / batch_size) #[cite: 2]
    
    # 4. Iterate over batches exactly like the training loop
    for k in tqdm(range(num_batch), desc="Evaluating Batches"):
        start_idx = k * batch_size #[cite: 2]
        end_idx = min(num_instance, start_idx + batch_size) #[cite: 2]
        
        sources_batch = train_data.sources[start_idx:end_idx] #[cite: 2]
        destinations_batch = train_data.destinations[start_idx:end_idx] #[cite: 2]
        timestamps_batch = train_data.timestamps[start_idx:end_idx] #[cite: 2]
        edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx] #[cite: 2]
        
        size = len(sources_batch) #[cite: 2]
        
        # Sample negatives for the batch
        _, negatives_batch = train_sampler.sample(size) #[cite: 2]
        
        # 5. Extraction Step (Mimicking tgn.compute_edge_probabilities)
        # Baseline (No Cache)
        base_pos = base_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=False) #[cite: 1]
        random.seed(42)
        base_neg = base_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=False) #[cite: 1]
        
        # Flat Cache
        flat_pos = flat_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=True) #[cite: 1]
        random.seed(42)
        flat_neg = flat_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=True) #[cite: 1]
        
        # Layer Cache
        layer_pos = layer_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=True) #[cite: 1]
        random.seed(42)
        layer_neg = layer_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=True) #[cite: 1]
        
        # 6. Calculate Recall for both positive and negative subgraphs
        for i in range(size):
            for b_data, f_data, l_data in [(base_pos[i], flat_pos[i], layer_pos[i]), (base_neg[i], flat_neg[i], layer_neg[i])]:
                base_nodes = set(b_data.nodes.tolist())
                base_edges = set(b_data.edge_time.tolist())
                
                if len(base_nodes) > 0:
                    metrics['flat_node_recall'].append(len(base_nodes.intersection(f_data.nodes.tolist())) / len(base_nodes))
                    metrics['layer_node_recall'].append(len(base_nodes.intersection(l_data.nodes.tolist())) / len(base_nodes))
                    
                if len(base_edges) > 0:
                    metrics['flat_edge_recall'].append(len(base_edges.intersection(f_data.edge_time.tolist())) / len(base_edges))
                    metrics['layer_edge_recall'].append(len(base_edges.intersection(l_data.edge_time.tolist())) / len(base_edges))
        
        # 7. Push Step (End of Batch)
        # Exclusively push the batch's positive edges into the cache after extraction is complete
        for s, d, t, e_idx in zip(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch): #[cite: 2]
            flat_finder.cache.push_edge(s, d, t, e_idx, flat_finder) #[cite: 1, 2]
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder) #[cite: 1, 2]

    # 8. Calculate Mean and Standard Deviation
    flat_node_mean, flat_node_std = np.mean(metrics['flat_node_recall']), np.std(metrics['flat_node_recall'])
    flat_edge_mean, flat_edge_std = np.mean(metrics['flat_edge_recall']), np.std(metrics['flat_edge_recall'])
    
    layer_node_mean, layer_node_std = np.mean(metrics['layer_node_recall']), np.std(metrics['layer_node_recall'])
    layer_edge_mean, layer_edge_std = np.mean(metrics['layer_edge_recall']), np.std(metrics['layer_edge_recall'])

    print(f"\n--- Recall Metrics (Mean ± Std Dev) - {dataset_name}---")
    print(f"Flat Cache   - Nodes: {flat_node_mean:.4f} ± {flat_node_std:.4f} | Edges: {flat_edge_mean:.4f} ± {flat_edge_std:.4f}")
    print(f"Layer Cache  - Nodes: {layer_node_mean:.4f} ± {layer_node_std:.4f} | Edges: {layer_edge_mean:.4f} ± {layer_edge_std:.4f}")
if __name__ == "__main__":
    for i in range(1, 5):
        evaluate_exact_training_fidelity(dataset_name=f"email-Eu-core-temporal-Dept{i}")