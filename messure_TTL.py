import numpy as np
from utils.data_processing import get_data


DATASETS = [
    "email-Eu-core-temporal-Dept1",
    "email-Eu-core-temporal-Dept2",
    "email-Eu-core-temporal-Dept3",
    "email-Eu-core-temporal-Dept4",
    "calls",
    "CollegeMsg"
]


def format_duration(seconds: float) -> str:
    """Formats raw seconds into human-readable days, hours, and minutes."""
    days = seconds / 86400.0
    hours = seconds / 3600.0
    minutes = seconds / 60.0

    if days >= 1.0:
        return f"{days:6.2f} days  ({hours:7.2f} hrs)"
    elif hours >= 1.0:
        return f"{hours:6.2f} hours ({days:7.3f} days)"
    else:
        return f"{minutes:6.2f} mins  ({hours:7.3f} hrs)"


def compute_inter_arrival_times(sources, destinations, timestamps):
    """
    Computes empirical inter-arrival intervals:
      1. Node-level: Time between any consecutive interactions involving a node.
         (Directly determines whether a node's cache entry remains warm in MultiLayerTemporalCache).
      2. Dyadic/Pair-level: Time between interactions between the same pair of nodes (conversational reply time).
    """
    # Ensure chronological order
    sort_idx = np.argsort(timestamps)
    sources = sources[sort_idx]
    destinations = destinations[sort_idx]
    timestamps = timestamps[sort_idx]

    node_last_seen = {}
    pair_last_seen = {}

    node_deltas = []
    pair_deltas = []

    for s, d, t in zip(sources, destinations, timestamps):
        # 1. Node-level inter-arrival (matches ttl_tracker[node])
        for node in (s, d):
            if node in node_last_seen:
                dt = t - node_last_seen[node]
                if dt > 0:  # Exclude concurrent multi-edge events
                    node_deltas.append(dt)
            node_last_seen[node] = t

        # 2. Dyadic (pair-level) conversation / reply inter-arrival
        pair_key = (min(s, d), max(s, d))
        if pair_key in pair_last_seen:
            dt_pair = t - pair_last_seen[pair_key]
            if dt_pair > 0:
                pair_deltas.append(dt_pair)
        pair_last_seen[pair_key] = t

    return np.array(node_deltas), np.array(pair_deltas)


def main():
    percentiles = [50, 75, 90, 95]

    print("=" * 95)
    print(f"{'Dataset':<30} | {'Metric':<22} | {'P50 (Median)':<18} | {'P90':<18} | {'P95 (Optimal)':<18}")
    print("=" * 95)

    dataset_recommendations = {}

    for dataset_name in DATASETS:
        try:
            # Load training portion matching sweep_layered_cache.py
            _, _, _, train_data, _, _, _, _ = get_data(dataset_name)
        except Exception as err:
            print(f"Failed to load {dataset_name}: {err}")
            continue

        node_deltas, pair_deltas = compute_inter_arrival_times(
            train_data.sources, train_data.destinations, train_data.timestamps
        )

        if len(node_deltas) == 0:
            print(f"{dataset_name:<30} | No temporal edges found.")
            continue

        node_p = np.percentile(node_deltas, percentiles)
        pair_p = np.percentile(pair_deltas, percentiles) if len(pair_deltas) > 0 else [0] * len(percentiles)

        # Store P95 node TTL (in seconds) as the optimal setting for MultiLayerTemporalCache
        dataset_recommendations[dataset_name] = {
            "p95_seconds": float(node_p[3]),
            "p95_days": float(node_p[3] / 86400.0),
            "p95_hours": float(node_p[3] / 3600.0)
        }

        # Print Node-level row (matches cache hit condition)
        print(
            f"{dataset_name:<30} | {'Node Active TTL':<22} | "
            f"{format_duration(node_p[0]):<18} | "
            f"{format_duration(node_p[2]):<18} | "
            f"{format_duration(node_p[3]):<18}"
        )

        # Print Dyad/Pair row (conversation reply time)
        if len(pair_deltas) > 0:
            print(
                f"{'':<30} | {'Pair Reply Time':<22} | "
                f"{format_duration(pair_p[0]):<18} | "
                f"{format_duration(pair_p[2]):<18} | "
                f"{format_duration(pair_p[3]):<18}"
            )
        print("-" * 95)

    print("\n\nRecommended P95 TTL settings dictionary for your cache experiments:")
    print("OPTIMAL_DATASET_TTLS = {")
    for name, stats in dataset_recommendations.items():
        print(f'    "{name}": {stats["p95_seconds"]:.1f},  # {stats["p95_days"]:.2f} days ({stats["p95_hours"]:.1f} hours)')
    print("}")


if __name__ == "__main__":
    main()