import numpy as np
import random
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from collections import deque

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
            same_index = [
                idx for idx, (i, j) in enumerate(zip(src_index, dst_index)) if i == j
            ]
            dst_index[same_index] = 1

        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def remove_redundant_edge(neighbors, edge_idxs, edge_times, edge_index):
    unique_edge_label = np.unique(edge_idxs)
    unique_edge_time = np.zeros(len(unique_edge_label)).astype(np.float32)
    unique_edge_index = np.zeros((2, len(unique_edge_label))).astype(np.int64)
    for i, label in enumerate(unique_edge_label):
        idx = np.where(edge_idxs == label)[0][0]
        unique_edge_time[i] = edge_times[idx]
        unique_edge_index[0][i] = edge_index[0][idx]
        unique_edge_index[1][i] = edge_index[1][idx]

    return np.unique(neighbors), unique_edge_label, unique_edge_time, unique_edge_index


def relabel_enclosing_subgraph(nodes, src_node, dst_node, edge_index):
    new_edge_index = np.zeros(edge_index.shape).astype(np.int64)
    for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        new_edge_index[0][i] = np.where(nodes == src)[0][0]
        new_edge_index[1][i] = np.where(nodes == dst)[0][0]

    return (
        np.where(nodes == src_node)[0][0],
        np.where(nodes == dst_node)[0][0],
        new_edge_index,
    )


# def drnl_node_labeling(src, dst, edge_index, num_nodes=None):
#     src, dst = (dst, src) if src > dst else (src, dst)
#     edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)
#     adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

#     idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
#     # adj without src
#     adj_wo_src = adj[idx, :][:, idx]

#     idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
#     # adj without dst
#     adj_wo_dst = adj[idx, :][:, idx]

#     dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
#     dist2src = np.insert(dist2src, dst, 0, axis=0)
#     dist2src = torch.from_numpy(dist2src)

#     dist2dst = shortest_path(
#         adj_wo_src, directed=False, unweighted=True, indices=dst - 1
#     )
#     dist2dst = np.insert(dist2dst, src, 0, axis=0)
#     dist2dst = torch.from_numpy(dist2dst)

#     dist = dist2src + dist2dst
#     dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode="floor"), dist % 2

#     z = 1 + torch.min(dist2src, dist2dst)
#     z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
#     z[src] = 1.0
#     z[dst] = 1.0
#     z[torch.isnan(z)] = 0.0

#     return z.to(torch.long)

def drnl_node_labeling(src, dst, edge_index, num_nodes=None):
    # Ensure consistent ordering
    src, dst = (dst, src) if src > dst else (src, dst)
    
    # Make the graph undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)
    
    # Dynamically resolve number of nodes if not provided
    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else max(src, dst) + 1
        
    # 1. Create a dense boolean adjacency matrix for rapid parallel lookup
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = True
    
    def get_shortest_path(start_node, exclude_node):
        dist = torch.full((num_nodes,), float('inf'), device=edge_index.device)
        dist[start_node] = 0.0
        
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        visited[exclude_node] = True
        visited[start_node] = True
        
        current_frontier = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        current_frontier[start_node] = True
        
        # PyTorch-Native Breadth-First Search (BFS)
        for d in range(1, num_nodes):
            if not current_frontier.any():
                break
            
            # Vectorized neighbor discovery 
            next_frontier = adj[current_frontier].any(dim=0)
            
            # Remove already visited nodes
            next_frontier = next_frontier & ~visited
            
            if not next_frontier.any():
                break
                
            dist[next_frontier] = float(d)
            visited |= next_frontier
            current_frontier = next_frontier
            
        return dist

    # 2. Compute distances independently
    dist2src = get_shortest_path(src, dst)
    dist2dst = get_shortest_path(dst, src)
    
    dist = dist2src + dist2dst
    
    # 3. Compute DRNL equation
    dist_over_2 = torch.div(dist, 2, rounding_mode="floor")
    dist_mod_2 = dist % 2
    
    z = 1.0 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1.0)
    
    # 4. Enforce strict labeling constraints
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isinf(z) | torch.isnan(z)] = 0.0
    
    return z.to(torch.long)


def get_node_max_ts(source_nodes, edge_times, edge_index, timestamp):
    nodes_ts = []
    for n in source_nodes:
        idx = np.concatenate(
            (np.where(edge_index[0] == n)[0], np.where(edge_index[1] == n)[0]), axis=0
        )
        if len(idx) != 0:
            nodes_ts.append(np.max(edge_times[idx]))
        else:
            nodes_ts.append(timestamp)
    return np.array(nodes_ts)


def get_neighbor_finder(data, uniform, max_node_idx=None, use_layered_cache=False):
    max_node_idx = (
        max(data.sources.max(), data.destinations.max())
        if max_node_idx is None
        else max_node_idx
    )
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform, use_layered_cache=use_layered_cache)




class TemporalSubgraphCache:
    def __init__(self, ttl_window=86400, max_edges=150):
        self.ttl_window = ttl_window
        self.max_edges = max_edges
        self.max_nodes = max_edges * 2  # 2 nodes per edge
        self.subgraph_cache = {}  
        self.ttl_tracker = {}     
        self.cache_hits = 0       
        self.cache_misses = 0

    def get_subgraph(self, node_id, timestamp, neighbor_finder, y, hop, n_neighbors):
        # Ensure time difference is positive (no time-travel) and within the TTL
        if node_id in self.ttl_tracker and (0 <= timestamp - self.ttl_tracker[node_id] <= self.ttl_window):
            self.cache_hits += 1
            cache = self.subgraph_cache[node_id]
            
            # Cast the deques to numpy arrays only at the exact moment the model needs them
            return {
                'nodes': np.array(cache['nodes'], dtype=np.int64),
                'edge_idxs': np.array(cache['edge_idxs'], dtype=np.int64),
                'edge_times': np.array(cache['edge_times'], dtype=np.float32),
                'edge_index': [list(cache['edge_index_0']), list(cache['edge_index_1'])]
            }
        
        self.cache_misses += 1
        n_nodes, e_idxs, e_times, e_index = neighbor_finder.get_k_hop_temporal_neighbor(
            [node_id], [timestamp], y, hop, n_neighbors
        )

        # Because get_k_hop_temporal_neighbor might return a numpy array or a list,
        # I didn't want to change the existing code.
        if isinstance(e_index, np.ndarray):
            e_index = e_index.tolist()

        # Initialize the cache using deques for ultra-fast, zero-copy FIFO buffering
        subgraph_data = {
            'nodes': deque(n_nodes.tolist() if isinstance(n_nodes, np.ndarray) else n_nodes, maxlen=self.max_nodes),
            'edge_idxs': deque(e_idxs.tolist() if isinstance(e_idxs, np.ndarray) else e_idxs, maxlen=self.max_edges),
            'edge_times': deque(e_times.tolist() if isinstance(e_times, np.ndarray) else e_times, maxlen=self.max_edges),
            'edge_index_0': deque(e_index[0], maxlen=self.max_edges),
            'edge_index_1': deque(e_index[1], maxlen=self.max_edges)
        }
        self.subgraph_cache[node_id] = subgraph_data
        self.ttl_tracker[node_id] = timestamp
        
        # Return standard arrays for the immediate extraction step
        return {
            'nodes': np.array(subgraph_data['nodes'], dtype=np.int64),
            'edge_idxs': np.array(subgraph_data['edge_idxs'], dtype=np.int64),
            'edge_times': np.array(subgraph_data['edge_times'], dtype=np.float32),
            'edge_index': [list(subgraph_data['edge_index_0']), list(subgraph_data['edge_index_1'])]
        }

    def push_edge(self, src, dst, ts, edge_idx, neighbor_finder, y=1, hop=2, n_neighbors=10):
        affected_nodes = set([src, dst])

        if hop - 1 > 0:
            current_frontier = set([src, dst])
            for _ in range(hop - 1):
                next_frontier = set()
                for node in current_frontier:
                    neighbors, _, _ = neighbor_finder.find_before(node, ts)
                    if len(neighbors) > n_neighbors:
                        neighbors = neighbors[-n_neighbors:]
                        
                    new_neighbors = set(neighbors) - affected_nodes
                    next_frontier.update(new_neighbors)
                    affected_nodes.update(new_neighbors)
                    
                current_frontier = next_frontier
                if not current_frontier:
                    break

        for node in affected_nodes:
            if node in self.ttl_tracker and (ts - self.ttl_tracker[node] <= self.ttl_window):
                cache = self.subgraph_cache[node]
                
                # --- O(1) PERFORMANCE ---
                # The deque automatically drops the oldest items in C-code, zero slicing required.
                cache['nodes'].extend([src, dst])
                cache['edge_times'].append(ts)
                cache['edge_idxs'].append(edge_idx)
                
                cache['edge_index_0'].append(src)
                cache['edge_index_1'].append(dst)
                
                self.ttl_tracker[node] = ts

    def reset_cache(self):
        self.subgraph_cache = {}  
        self.ttl_tracker = {}     
        self.cache_hits = 0       
        self.cache_misses = 0


class MultiLayerTemporalCache:
    def __init__(self, ttl_window=86400, max_edges_per_hop={1: 20, 2: 60, 3: 180}):
        self.ttl_window = ttl_window
        self.max_edges_per_hop = max_edges_per_hop
        self.subgraph_cache = {}  
        self.ttl_tracker = {}     
        self.cache_hits = 0       
        self.cache_misses = 0

    def _init_node_cache(self):
        """Initializes deques partitioned by hop layer for a specific node."""
        node_cache = {}
        for hop, max_e in self.max_edges_per_hop.items():
            max_n = max_e * 2
            node_cache[hop] = {
                'nodes': deque(maxlen=max_n),
                'edge_idxs': deque(maxlen=max_e),
                'edge_times': deque(maxlen=max_e),
                'edge_index_0': deque(maxlen=max_e),
                'edge_index_1': deque(maxlen=max_e)
            }
        return node_cache

    def get_subgraph(self, node_id, timestamp, neighbor_finder, y, hop, n_neighbors):
        #1. Cache Hit Logic
        if node_id in self.ttl_tracker and (0 <= timestamp - self.ttl_tracker[node_id] <= self.ttl_window):
            self.cache_hits += 1
            cache = self.subgraph_cache[node_id]
            
            # Find which hops actually exist in the cache for this node
            active_hops = [h for h in range(1, hop + 1) if h in cache]
            
            # Calculate exact total sizes upfront (Nodes vs. Edges)
            total_nodes = sum(len(cache[h]['nodes']) for h in active_hops)
            total_edges = sum(len(cache[h]['edge_idxs']) for h in active_hops)
            
            # Pre-allocate fixed-size NumPy arrays 
            nodes = np.empty(total_nodes, dtype=np.int64)
            edge_idxs = np.empty(total_edges, dtype=np.int64)
            edge_times = np.empty(total_edges, dtype=np.float32)
            ei_0 = np.empty(total_edges, dtype=np.int64)
            ei_1 = np.empty(total_edges, dtype=np.int64)
            
            # Fill the arrays using dual moving pointers (Slice Assignment)
            n_ptr = 0
            e_ptr = 0
            
            for h in active_hops:
                n_len = len(cache[h]['nodes'])
                e_len = len(cache[h]['edge_idxs'])
                
                # Assign nodes
                if n_len > 0:
                    next_n_ptr = n_ptr + n_len
                    nodes[n_ptr:next_n_ptr] = cache[h]['nodes']
                    n_ptr = next_n_ptr
                    
                # Assign edges
                if e_len > 0:
                    next_e_ptr = e_ptr + e_len
                    
                    edge_idxs[e_ptr:next_e_ptr] = cache[h]['edge_idxs']
                    edge_times[e_ptr:next_e_ptr] = cache[h]['edge_times']
                    ei_0[e_ptr:next_e_ptr] = cache[h]['edge_index_0']
                    ei_1[e_ptr:next_e_ptr] = cache[h]['edge_index_1']
                    
                    e_ptr = next_e_ptr
                    
            return {
                'nodes': nodes,
                'edge_idxs': edge_idxs,
                'edge_times': edge_times,
                'edge_index': [ei_0.tolist(), ei_1.tolist()] 
            }
        # 2. Cache Miss Logic
        self.cache_misses += 1
        layered_neighbors = neighbor_finder.get_layered_k_hop_temporal_neighbor(
            [node_id], [timestamp], y, hop, n_neighbors
        )

        self.subgraph_cache[node_id] = self._init_node_cache()
        cache = self.subgraph_cache[node_id]

        nodes_all, edge_idxs_all, edge_times_all, ei_0_all, ei_1_all = [], [], [], [], []

        for h in range(1, hop + 1):
            if h in layered_neighbors:
                n_nodes, e_idxs, e_times, e_index = layered_neighbors[h]
                if isinstance(e_index, np.ndarray):
                    e_index = e_index.tolist()

                n_list = n_nodes.tolist() if isinstance(n_nodes, np.ndarray) else n_nodes
                e_idxs_list = e_idxs.tolist() if isinstance(e_idxs, np.ndarray) else e_idxs
                e_times_list = e_times.tolist() if isinstance(e_times, np.ndarray) else e_times
                
                if h in cache:
                    cache[h]['nodes'].extend(n_list)
                    cache[h]['edge_idxs'].extend(e_idxs_list)
                    cache[h]['edge_times'].extend(e_times_list)
                    cache[h]['edge_index_0'].extend(e_index[0])
                    cache[h]['edge_index_1'].extend(e_index[1])

                nodes_all.extend(n_list)
                edge_idxs_all.extend(e_idxs_list)
                edge_times_all.extend(e_times_list)
                ei_0_all.extend(e_index[0])
                ei_1_all.extend(e_index[1])

        self.ttl_tracker[node_id] = timestamp
        
        return {
            'nodes': np.array(nodes_all, dtype=np.int64),
            'edge_idxs': np.array(edge_idxs_all, dtype=np.int64),
            'edge_times': np.array(edge_times_all, dtype=np.float32),
            'edge_index': [ei_0_all, ei_1_all]
        }

    def push_edge(self, src, dst, ts, edge_idx, neighbor_finder, y=1, hop=2, n_neighbors=10):
        visited = set([src, dst])
        current_frontier = set([src, dst])
        
        # Route new interactions strictly to the cache layer corresponding to the hop distance
        for current_hop in range(1, hop + 1):
            for node in current_frontier:
                if node in self.ttl_tracker and (ts - self.ttl_tracker[node] <= self.ttl_window):
                    if current_hop in self.subgraph_cache[node]:
                        cache_layer = self.subgraph_cache[node][current_hop]
                        cache_layer['nodes'].extend([src, dst])
                        cache_layer['edge_times'].append(ts)
                        cache_layer['edge_idxs'].append(edge_idx)
                        cache_layer['edge_index_0'].append(src)
                        cache_layer['edge_index_1'].append(dst)
                        self.ttl_tracker[node] = ts

            if current_hop < hop:
                next_frontier = set()
                for node in current_frontier:
                    neighbors, _, _ = neighbor_finder.find_before(node, ts)
                    if len(neighbors) > n_neighbors:
                        neighbors = neighbors[-n_neighbors:]
                        
                    new_neighbors = set(neighbors) - visited
                    next_frontier.update(new_neighbors)
                    visited.update(new_neighbors)
                    
                current_frontier = next_frontier
                if not current_frontier:
                    break

    def reset_cache(self):
        self.subgraph_cache = {}  
        self.ttl_tracker = {}     
        self.cache_hits = 0       
        self.cache_misses = 0

class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None, use_layered_cache=False):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # We sort the list based on timestamp
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbors]))
            self.node_to_edge_timestamps.append(
                np.array([x[2] for x in sorted_neighbors])
            )

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

        if use_layered_cache:
            self.cache = MultiLayerTemporalCache()
        else:
            self.cache = TemporalSubgraphCache()

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph.
        The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i],
        )

    def find_equal_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time + 0.01)

        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i],
        )

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of
        each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert len(source_nodes) == len(timestamps)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int64)
        # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i]
        # with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.float32)
        # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i]
        # and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(np.int64)
        # each entry in position (i,j) represent the interaction index of an interaction between
        # user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(
                source_node, timestamp
            )
            # extracts all neighbors, interactions indexes and timestamps of all interactions of
            # user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(
                        0, len(source_neighbors), n_neighbors
                    )

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert len(source_neighbors) <= n_neighbors
                    assert len(source_edge_times) <= n_neighbors
                    assert len(source_edge_idxs) <= n_neighbors

                    neighbors[i, n_neighbors - len(source_neighbors) :] = (
                        source_neighbors
                    )
                    edge_times[i, n_neighbors - len(source_edge_times) :] = (
                        source_edge_times
                    )
                    edge_idxs[i, n_neighbors - len(source_edge_idxs) :] = (
                        source_edge_idxs
                    )

        return neighbors, edge_idxs, edge_times

    def get_temporal_neighbor_coo_format(
        self, source_nodes, timestamps, y, n_neighbors=10
    ):
        neighbors = []
        edge_times = []
        edge_idxs = []
        edge_index = [[], []]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            if y == 1:
                source_neighbors, source_edge_idxs, source_edge_times = (
                    self.find_equal_before(source_node, timestamp)
                )
            else:
                source_neighbors, source_edge_idxs, source_edge_times = (
                    self.find_before(source_node, timestamp)
                )

            if len(source_neighbors) > 0 and n_neighbors > 0:
                source_neighbors = source_neighbors[-n_neighbors:]
                source_edge_times = source_edge_times[-n_neighbors:]
                source_edge_idxs = source_edge_idxs[-n_neighbors:]

                assert len(source_neighbors) <= n_neighbors
                assert len(source_edge_times) <= n_neighbors
                assert len(source_edge_idxs) <= n_neighbors

                neighbors.extend(source_neighbors)
                edge_times.extend(source_edge_times)
                edge_idxs.extend(source_edge_idxs)
                edge_index[0].extend(
                    np.repeat(source_node, len(source_neighbors)).tolist()
                )
                edge_index[1].extend(source_neighbors)

        return neighbors, edge_idxs, edge_times, edge_index

    def get_k_hop_temporal_neighbor(
        self, source_nodes, timestamps, y, hop=2, n_neighbors=10
    ):
        assert hop >= 1

        neighbors, edge_idxs, edge_times, edge_index = (
            self.get_temporal_neighbor_coo_format(
                source_nodes, timestamps, y, n_neighbors
            )
        )

        if hop == 1:
            return neighbors, edge_idxs, edge_times, edge_index

        else:
            sub_neighbors, sub_edge_idxs, sub_edge_times, sub_edge_index = (
                self.get_k_hop_temporal_neighbor(
                    neighbors, edge_times, y, hop - 1, n_neighbors
                )
            )

            neighbors = np.append(neighbors, sub_neighbors, axis=0)
            edge_idxs = np.append(edge_idxs, sub_edge_idxs, axis=0)
            edge_times = np.append(edge_times, sub_edge_times, axis=0)
            edge_index = np.append(edge_index, sub_edge_index, axis=1)

            return neighbors, edge_idxs, edge_times, edge_index

    def extract_enclosing_subgraph(
        self, src_nodes, dst_nodes, edge_times, y, hop=2, n_neighbors=10, use_cache=False
    ):
        data_list = []

        for i, (src, dst, ts) in enumerate(zip(src_nodes, dst_nodes, edge_times)):
            if src == dst:
                dst = random.randint(0, dst - 1)

            if use_cache:
                # CACHE ROUTING
                src_subgraph = self.cache.get_subgraph(src, ts, self, y, hop, n_neighbors)
                dst_subgraph = self.cache.get_subgraph(dst, ts, self, y, hop, n_neighbors)

                sub_nodes = np.concatenate([src_subgraph['nodes'], dst_subgraph['nodes'], [src, dst]])
                sub_edge_idxs = np.concatenate([src_subgraph['edge_idxs'], dst_subgraph['edge_idxs']])
                sub_edge_times = np.concatenate([src_subgraph['edge_times'], dst_subgraph['edge_times']])
                
                # Convert back to 2D numpy array for downstream processing
                sub_edge_index = np.array([
                    src_subgraph['edge_index'][0] + dst_subgraph['edge_index'][0],
                    src_subgraph['edge_index'][1] + dst_subgraph['edge_index'][1]
                ], dtype=np.int64)
                # END CACHE ROUTING
            else:
                sub_nodes, sub_edge_idxs, sub_edge_times, sub_edge_index = (
                self.get_k_hop_temporal_neighbor(
                    [src, dst], [ts, ts], y, hop, n_neighbors
                    )
                )

                sub_nodes = np.append(sub_nodes, src)
                sub_nodes = np.append(sub_nodes, dst)

            sub_nodes, sub_edge_idxs, sub_edge_times, sub_edge_index = (
                remove_redundant_edge(
                    sub_nodes, sub_edge_idxs, sub_edge_times, sub_edge_index
                )
            )

            sub_nodes = np.unique(np.concatenate([sub_nodes, sub_edge_index[0], sub_edge_index[1], [src, dst]]))

            node_timestamps = get_node_max_ts(
                sub_nodes, sub_edge_times, sub_edge_index, ts
            )

            src_mapping, dst_mapping, sub_edge_index = relabel_enclosing_subgraph(
                sub_nodes, src, dst, sub_edge_index
            )

            sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.int64)

            mask1 = (sub_edge_index[0] != src_mapping) | (
                sub_edge_index[1] != dst_mapping
            )
            mask2 = (sub_edge_index[0] != dst_mapping) | (
                sub_edge_index[1] != src_mapping
            )
            mask = mask1 & mask2
            sub_edge_index = sub_edge_index[:, mask]
            if len(mask) > 1:
                sub_edge_times = sub_edge_times[mask1 & mask2]
            elif len(mask) == 1 and not mask[0]:
                sub_edge_times = np.asarray([])

            z = drnl_node_labeling(
                src_mapping, dst_mapping, sub_edge_index, len(sub_nodes)
            )

            data = Data(
                nodes=sub_nodes.astype(np.int32),
                node_timestamps=node_timestamps,
                edge_index=sub_edge_index,
                edge_time=sub_edge_times,
                y=y,
                z=z,
            )
            data_list.append(data)

        return data_list
    def get_layered_k_hop_temporal_neighbor(
        self, source_nodes, timestamps, y, hop=2, n_neighbors=10
    ):
        """
        Recursively extracts temporal neighborhoods separated by hop level.
        Returns a dictionary mapping hop depth -> (neighbors, edge_idxs, edge_times, edge_index)
        """
        assert hop >= 1

        neighbors, edge_idxs, edge_times, edge_index = (
            self.get_temporal_neighbor_coo_format(
                source_nodes, timestamps, y, n_neighbors
            )
        )

        result = {1: (neighbors, edge_idxs, edge_times, edge_index)}

        if hop > 1 and len(neighbors) > 0:
            sub_layered = self.get_layered_k_hop_temporal_neighbor(
                neighbors, edge_times, y, hop - 1, n_neighbors
            )
            for sub_hop, data in sub_layered.items():
                result[sub_hop + 1] = data

        return result

