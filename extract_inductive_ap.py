import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np

def get_base_prefix(filename_stem):
    """Strips the trailing run identifier (e.g., '_1', '_2') to group identical configs."""
    return re.sub(r'_\d+$', '', filename_stem)

def main():
    parser = argparse.ArgumentParser(description="Extract and average inductive AP from result pickles.")
    parser.add_argument("--dir", type=str, default="results", help="Directory containing .pkl files")
    parser.add_argument("--out", type=str, default="inductive_aps_mean.json", help="Output JSON file name")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory '{args.dir}' not found.")
        return

    # Store lists of run-level mean APs grouped by their base prefix
    grouped_metrics = {}

    for pkl_file in folder.glob("*.pkl"):
        prefix_key = pkl_file.stem
        base_prefix = get_base_prefix(prefix_key)
        
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            
            # Calculate the mean of validation APs across all epochs for this specific run
            if "new_nodes_val_aps" in data and len(data["new_nodes_val_aps"]) > 0:
                run_mean_ap = float(np.mean(data["new_nodes_val_aps"]))
                
                if base_prefix not in grouped_metrics:
                    grouped_metrics[base_prefix] = []
                grouped_metrics[base_prefix].append(run_mean_ap)
            else:
                print(f"Warning: 'new_nodes_val_aps' missing or empty in {pkl_file.name}")
            
        except Exception as e:
            print(f"Failed to process {pkl_file.name}: {e}")

    # Calculate the cross-run mean for each base prefix
    mean_metrics = {}
    for base_prefix, aps in grouped_metrics.items():
        mean_metrics[base_prefix] = float(np.mean(aps))

    with open(args.out, "w") as f:
        json.dump(mean_metrics, f, indent=4)
        
    total_files = sum(len(v) for v in grouped_metrics.values())
    print(f"Successfully aggregated {len(mean_metrics)} base configurations from {total_files} files.")
    print(f"Saved mean APs to {args.out}")

if __name__ == "__main__":
    main()