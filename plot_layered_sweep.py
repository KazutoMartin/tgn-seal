import json
import glob
import os
import matplotlib.pyplot as plt

def generate_combined_plots(directory="./sweep_results"):
    wide_files = [f for f in glob.glob(os.path.join(directory, "*_capacity_sweep.json"))]
    
    metrics = [
        ("iou", "Intersection over Union (IoU)"),
        ("f2", "F2 Score"),
        ("cache_hit_rate", "Cache Hit Rate"),
        ("recall", "Subgraph Edge Recall"),
        ("precision", "Subgraph Edge Precision"),
        ("latency_ms", "Extraction Latency (ms)")
    ]

    for wide_file in wide_files:
        narrow_file_1 = wide_file.replace(".json", "_1.json")
        narrow_file_2 = wide_file.replace(".json", "_2.json")
        
        if os.path.exists(narrow_file_1):
            narrow_file = narrow_file_1
        elif os.path.exists(narrow_file_2):
            narrow_file = narrow_file_2
        else:
            print(f"Matching narrow sweep file not found for {wide_file}")
            continue
            
        with open(wide_file, 'r') as f:
            wide_data = json.load(f)["sweep_records"]
        with open(narrow_file, 'r') as f:
            narrow_data = json.load(f)["sweep_records"]
            
        dataset_name = os.path.basename(wide_file).split('_capacity')[0]
        
        # Combine records to find the global optimum across both sweeps
        all_records = wide_data + narrow_data
        
        best_f2_record = max(all_records, key=lambda x: x["f2"])
        best_iou_record = max(all_records, key=lambda x: x["iou"])
        
        best_f2_c1 = best_f2_record['c1']
        best_f2_val = best_f2_record['f2']
        best_iou_c1 = best_iou_record['c1']
        best_iou_val = best_iou_record['iou']
        
        print(f"\n--- Dataset: {dataset_name} ---")
        print(f"Optimal Capacity for F2: c1={best_f2_c1}, c2={best_f2_record['c2']} (F2={best_f2_val:.4f})")
        print(f"Optimal Capacity for IoU: c1={best_iou_c1}, c2={best_iou_record['c2']} (IoU={best_iou_val:.4f})")
            
        wide_c1 = [r["c1"] for r in wide_data]
        narrow_c1 = [r["c1"] for r in narrow_data]
        
        def create_plot(is_zoomed):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for i, (metric_key, metric_title) in enumerate(metrics):
                ax = axes[i]
                
                wide_vals = [r[metric_key] for r in wide_data]
                narrow_vals = [r[metric_key] for r in narrow_data]
                
                all_c1_vals = wide_c1 + narrow_c1
                all_metric_vals = wide_vals + narrow_vals
                
                if metric_key == "latency_ms":
                    # Plot them as individual connected series, skipping the gray continuous line
                    ax.plot(wide_c1, wide_vals, 'o-', label='Wide Sweep', color='#1f77b4', linewidth=2, markersize=7, zorder=2)
                    ax.plot(narrow_c1, narrow_vals, 's--', label='Narrow Sweep', color='#ff7f0e', linewidth=2, markersize=7, zorder=3)
                else:
                    # Sort the combined data by c1 to draw a single continuous line
                    sorted_pairs = sorted(zip(all_c1_vals, all_metric_vals))
                    sorted_c1 = [p[0] for p in sorted_pairs]
                    sorted_vals = [p[1] for p in sorted_pairs]
                    
                    # 1. Draw the continuous connected line
                    ax.plot(sorted_c1, sorted_vals, '-', color='gray', alpha=0.6, linewidth=2, zorder=1)
                    
                    # 2. Plot the distinct markers for Wide and Narrow sweeps on top
                    ax.plot(wide_c1, wide_vals, 'o', label='Wide Sweep', color='#1f77b4', markersize=7, zorder=2)
                    ax.plot(narrow_c1, narrow_vals, 's', label='Narrow Sweep', color='#ff7f0e', markersize=7, zorder=3)
                
                # Draw vertical lines for the optimal capacities
                ax.axvline(x=best_f2_c1, color='#d62728', linestyle='--', alpha=0.7, zorder=0, label=f'Optimum F2 (C1={best_f2_c1})')
                ax.axvline(x=best_iou_c1, color='#1f77b4', linestyle='--', alpha=0.7, zorder=0, label=f'Optimum IoU (C1={best_iou_c1})')
                
                ax.set_title(metric_title, fontweight='bold')
                ax.set_xlabel("c1 (Hop-1 Edge Capacity)")
                ax.set_ylabel(metric_key)
                
                if not is_zoomed:
                    # Apply specific scaling on the unzoomed version
                    if metric_key == "latency_ms":
                        ax.set_ylim(0, 5)
                    else:
                        ax.set_ylim(0, 1)
                    
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.7)
                
            # Place optimal capacity, F2, and IoU directly in the main title
            title_text = (
                f"Hyperparameter Sweep: {dataset_name} ({'Zoomed' if is_zoomed else 'Unzoomed'})\n"
                f"Optimal Parameters -> Best F2: C1={best_f2_c1} (F2={best_f2_val:.4f}) | Best IoU: C1={best_iou_c1} (IoU={best_iou_val:.4f})"
            )
            plt.suptitle(title_text, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            suffix = "_combined_zoomed.png" if is_zoomed else "_combined.png"
            out_filename = wide_file.replace(".json", suffix)
            plt.savefig(out_filename, dpi=150)
            plt.close()
            print(f"Generated plot: {out_filename}")

        # Generate both unzoomed (0 to 1/5 range) and zoomed (auto-scaled) versions
        create_plot(is_zoomed=False)
        create_plot(is_zoomed=True)

if __name__ == "__main__":
    generate_combined_plots()