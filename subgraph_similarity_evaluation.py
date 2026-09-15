import numpy as np
import math
from tqdm import tqdm
import random

# Assuming these are available in your PYTHONPATH
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder, RandEdgeSampler

def evaluate_exact_training_fidelity(dataset_name="email-Eu-core-temporal-Dept1", batch_size=200):
    # 1. Load data and isolate the training split exactly like the training script
    _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
    
    # 2. Initialize independent finders
    base_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    flat_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=False)
    layer_finder = get_neighbor_finder(train_data, uniform=False, use_layered_cache=True)
    
    # 3. Initialize the Negative Sampler 
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    
    metrics = {
        'flat_node_recall': [], 'flat_edge_recall': [], 'flat_drnl_agreement': [],
        'layer_node_recall': [], 'layer_edge_recall': [], 'layer_drnl_agreement': []
    }
    
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / batch_size)
    
    # 4. Iterate over batches exactly like the training loop
    for k in tqdm(range(num_batch), desc="Evaluating Batches"):
        start_idx = k * batch_size
        end_idx = min(num_instance, start_idx + batch_size)
        
        sources_batch = train_data.sources[start_idx:end_idx]
        destinations_batch = train_data.destinations[start_idx:end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
        
        size = len(sources_batch)
        
        # Sample negatives for the batch
        _, negatives_batch = train_sampler.sample(size)
        
        # 5. Extraction Step (Mimicking tgn.compute_edge_probabilities)
        # Baseline (No Cache)
        base_pos = base_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=False)
        random.seed(42)
        base_neg = base_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=False)
        
        # Flat Cache
        flat_pos = flat_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=True)
        random.seed(42)
        flat_neg = flat_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=True)
        
        # Layer Cache
        layer_pos = layer_finder.extract_enclosing_subgraph(sources_batch, destinations_batch, timestamps_batch, y=1, use_cache=True)
        random.seed(42)
        layer_neg = layer_finder.extract_enclosing_subgraph(sources_batch, negatives_batch, timestamps_batch, y=0, use_cache=True)
        
        # 6. Calculate Recall and DRNL Agreement for both positive and negative subgraphs
        for i in range(size):
            for b_data, f_data, l_data in [(base_pos[i], flat_pos[i], layer_pos[i]), (base_neg[i], flat_neg[i], layer_neg[i])]:
                
                base_nodes = b_data.nodes.tolist()
                flat_nodes = f_data.nodes.tolist()
                layer_nodes = l_data.nodes.tolist()
                
                base_nodes_set = set(base_nodes)
                base_edges_set = set(b_data.edge_time.tolist())
                
                if len(base_nodes_set) > 0:
                    metrics['flat_node_recall'].append(len(base_nodes_set.intersection(flat_nodes)) / len(base_nodes_set))
                    metrics['layer_node_recall'].append(len(base_nodes_set.intersection(layer_nodes)) / len(base_nodes_set))
                    
                    # DRNL Agreement Calculation
                    # Create mappings from node ID to DRNL label 'z'
                    b_z_dict = dict(zip(base_nodes, b_data.z.tolist()))
                    f_z_dict = dict(zip(flat_nodes, f_data.z.tolist()))
                    l_z_dict = dict(zip(layer_nodes, l_data.z.tolist()))
                    
                    # A match means the node is present in the cache extraction AND its DRNL label is identical
                    f_drnl_matches = sum(1 for n in base_nodes if n in f_z_dict and b_z_dict[n] == f_z_dict[n])
                    l_drnl_matches = sum(1 for n in base_nodes if n in l_z_dict and b_z_dict[n] == l_z_dict[n])
                    
                    metrics['flat_drnl_agreement'].append(f_drnl_matches / len(base_nodes_set))
                    metrics['layer_drnl_agreement'].append(l_drnl_matches / len(base_nodes_set))
                    
                if len(base_edges_set) > 0:
                    metrics['flat_edge_recall'].append(len(base_edges_set.intersection(f_data.edge_time.tolist())) / len(base_edges_set))
                    metrics['layer_edge_recall'].append(len(base_edges_set.intersection(l_data.edge_time.tolist())) / len(base_edges_set))
        
        # 7. Push Step (End of Batch)
        # Exclusively push the batch's positive edges into the cache after extraction is complete
        for s, d, t, e_idx in zip(sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch):
            flat_finder.cache.push_edge(s, d, t, e_idx, flat_finder)
            layer_finder.cache.push_edge(s, d, t, e_idx, layer_finder)

    # 8. Calculate Mean and Standard Deviation
    flat_node_mean, flat_node_std = np.mean(metrics['flat_node_recall']), np.std(metrics['flat_node_recall'])
    flat_edge_mean, flat_edge_std = np.mean(metrics['flat_edge_recall']), np.std(metrics['flat_edge_recall'])
    flat_drnl_mean, flat_drnl_std = np.mean(metrics['flat_drnl_agreement']), np.std(metrics['flat_drnl_agreement'])
    
    layer_node_mean, layer_node_std = np.mean(metrics['layer_node_recall']), np.std(metrics['layer_node_recall'])
    layer_edge_mean, layer_edge_std = np.mean(metrics['layer_edge_recall']), np.std(metrics['layer_edge_recall'])
    layer_drnl_mean, layer_drnl_std = np.mean(metrics['layer_drnl_agreement']), np.std(metrics['layer_drnl_agreement'])

    print(f"\n--- Metrics (Mean ± Std Dev) - {dataset_name}---")
    print(f"Flat Cache   - Nodes: {flat_node_mean:.4f} ± {flat_node_std:.4f} | Edges: {flat_edge_mean:.4f} ± {flat_edge_std:.4f} | DRNL: {flat_drnl_mean:.4f} ± {flat_drnl_std:.4f}")
    print(f"Layer Cache  - Nodes: {layer_node_mean:.4f} ± {layer_node_std:.4f} | Edges: {layer_edge_mean:.4f} ± {layer_edge_std:.4f} | DRNL: {layer_drnl_mean:.4f} ± {layer_drnl_std:.4f}")

if __name__ == "__main__":
    for i in range(1, 5):
        evaluate_exact_training_fidelity(dataset_name=f"email-Eu-core-temporal-Dept{i}")