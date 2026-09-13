import argparse
import numpy as np
import torch
from utils.data_processing import get_data
from utils.utils import get_neighbor_finder


def parse_args():
    parser = argparse.ArgumentParser("Temporal Graph Density Evaluator")
    parser.add_argument("-d", "--data", type=str, default="email-Eu-core-temporal-Dept4", help="Dataset name")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of interactions to sample for local metrics")
    parser.add_argument("--n_hops", type=int, default=2, help="Number of hops for enclosing subgraphs")
    parser.add_argument("--n_degree", type=int, default=10, help="Number of temporal neighbors sampled")
    return parser.parse_args()


def evaluate_dataset_density(dataset_name, n_samples=1000, n_hops=2, n_degree=10):
    _, _, full_data, train_data, val_data, test_data, _, _ = get_data(dataset_name)

    # 1. Global Topological Metrics
    sources = full_data.sources
    destinations = full_data.destinations
    timestamps = full_data.timestamps

    all_nodes = np.unique(np.concatenate([sources, destinations]))
    num_nodes = len(all_nodes)
    num_temporal_edges = len(sources)

    # Collapse multi-edges to measure static topological footprint
    directed_edges = set(zip(sources, destinations))
    num_unique_directed = len(directed_edges)
    undirected_edges = set((min(u, v), max(u, v)) for u, v in directed_edges if u != v)
    num_unique_undirected = len(undirected_edges)

    # Standard Directed Density
    max_possible_edges = num_nodes * (num_nodes - 1)
    global_static_density = num_unique_directed / max_possible_edges if max_possible_edges > 0 else 0.0
    global_temporal_density = num_temporal_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    temporal_multiplicity = num_temporal_edges / num_unique_directed if num_unique_directed > 0 else 0.0

    # 2. Node Degree Statistics (Static Footprint)
    degrees = np.zeros(int(all_nodes.max()) + 1, dtype=np.int64)
    for u, v in undirected_edges:
        degrees[u] += 1
        degrees[v] += 1
    active_degrees = degrees[all_nodes]

    # 3. Local Subgraph Density Metrics (Sampled Enclosing Subgraphs)
    neighbor_finder = get_neighbor_finder(full_data, uniform=False)
    sample_indices = np.random.choice(
        num_temporal_edges, size=min(n_samples, num_temporal_edges), replace=False
    )
    sample_src = sources[sample_indices]
    sample_dst = destinations[sample_indices]
    sample_ts = timestamps[sample_indices]

    subgraphs = neighbor_finder.extract_enclosing_subgraph(
        sample_src, sample_dst, sample_ts, y=1, hop=n_hops, n_neighbors=n_degree, use_cache=False
    )

    local_densities = []
    subgraph_node_counts = []
    subgraph_edge_counts = []
    anchor_degree_sums = []

    for idx, sub in enumerate(subgraphs):
        v_sub = len(sub.nodes)
        # sub.edge_index is shape [2, E]
        e_sub = sub.edge_index.shape[1] if sub.edge_index.numel() > 0 else 0

        # Formula: |E_sub| / (|V_sub| * (|V_sub| - 1))
        if v_sub > 1:
            d_local = e_sub / (v_sub * (v_sub - 1))
        else:
            d_local = 0.0

        local_densities.append(d_local)
        subgraph_node_counts.append(v_sub)
        subgraph_edge_counts.append(e_sub)
        anchor_degree_sums.append(degrees[sample_src[idx]] + degrees[sample_dst[idx]])

    local_densities = np.array(local_densities)
    subgraph_node_counts = np.array(subgraph_node_counts)
    anchor_degree_sums = np.array(anchor_degree_sums)

    print(f"\n=======================================================")
    print(f" DENSITY EVALUATION REPORT: {dataset_name.upper()}")
    print(f"=======================================================")
    print(f"**Global Dataset Statistics**")
    print(f"  * Total Active Nodes: {num_nodes:,}")
    print(f"  * Total Temporal Interactions: {num_temporal_edges:,}")
    print(f"  * Unique Directed Edges: {num_unique_directed:,}")
    print(f"  * Edge Multiplicity (|E_temp| / |E_unique|): {temporal_multiplicity:.2f}x")
    print(f"  * Global Static Graph Density: {global_static_density:.6e}")
    print(f"  * Global Temporal Event Density: {global_temporal_density:.6e}")
    print(f"  * Degree (Mean ± Std): {active_degrees.mean():.2f} ± {active_degrees.std():.2f} (Median: {np.median(active_degrees):.0f}, Max: {active_degrees.max():.0f})")

    print(f"\n**Local Subgraph Density Distribution ({n_hops}-hop enclosing, {len(local_densities)} samples)**")
    print(f"  * Mean Subgraph Density: {local_densities.mean():.4f}")
    print(f"  * Median Subgraph Density (p50): {np.median(local_densities):.4f}")
    print(f"  * 10th Percentile (Sparse Boundary): {np.percentile(local_densities, 10):.4f}")
    print(f"  * 25th Percentile: {np.percentile(local_densities, 25):.4f}")
    print(f"  * 75th Percentile: {np.percentile(local_densities, 75):.4f}")
    print(f"  * 90th Percentile (Dense Core): {np.percentile(local_densities, 90):.4f}")
    print(f"  * Average Subgraph Size: {subgraph_node_counts.mean():.1f} nodes, {np.mean(subgraph_edge_counts):.1f} edges")
    print(f"  * Anchor Degree Sum (Mean / Median): {anchor_degree_sums.mean():.1f} / {np.median(anchor_degree_sums):.1f}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset_density(args.data, args.n_samples, args.n_hops, args.n_degree)