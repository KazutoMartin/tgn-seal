#!/usr/bin/env python3
"""
Fixed Configuration Benchmarking Script:
- Aggregator: Learned Attention (--aggregator attention)
- Pooling: Targeted Node Extraction (--pooling target)
- Cache: Layered Cache (--use_layered_cache)
- Link Predictor: Transformer
- Datasets: email-Eu-core-temporal-Dept1 through Dept4
- Execution: 2 runs per configuration
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

# Base departmental configurations (Email-Eu-core Dept 1 to 4)
EXPERIMENT_CONFIGS = [
    "-d email-Eu-core-temporal-Dept1 --use_memory --embedding_module identity --prefix tgn-seal-dept1-attn-target-layered --n_runs 2 --n_epoch 50",
    # "-d email-Eu-core-temporal-Dept2 --use_memory --embedding_module identity --prefix tgn-seal-dept2-attn-target-layered --n_runs 2 --n_epoch 50",
    # "-d email-Eu-core-temporal-Dept3 --use_memory --embedding_module identity --prefix tgn-seal-dept3-attn-target-layered --n_runs 2 --n_epoch 50",
    # "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --prefix tgn-seal-dept4-attn-target-layered --n_runs 2 --n_epoch 50",
]

POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv fixed_ablation_summary.csv",
]

def send_ntfy_notification(message: str, title: str = "Ablation Script"):
    """Send a push notification through ntfy.sh."""
    topic = "tgnsealimprovementeval"
    url = f"https://ntfy.sh/{topic}"
    data = message.encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Title": title},
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status < 200 or response.status >= 300:
                print(f"[NOTIFICATION] ntfy returned HTTP {response.status}")
            else:
                print("[NOTIFICATION] ntfy notification sent.")
    except Exception as e:
        # Fails silently to ensure script continuation
        print(f"[NOTIFICATION] Failed to send ntfy notification: {e}")

def extract_prefix(args_str: str) -> str:
    parts = args_str.split()
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        return parts[idx + 1]
    return f"ablation_exp_{int(time.time())}"

def run_command(command: str, log_filepath: Path, env_vars: dict) -> bool:
    with open(log_filepath, "a") as f_log:
        f_log.write(f"\n=== Executing: {command} ===\nTimestamp: {datetime.now().isoformat()}\n{'=' * 80}\n\n")
        f_log.flush()
        try:
            process = subprocess.Popen(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_vars
            )
            for line in process.stdout:
                f_log.write(line)
                f_log.flush()
            process.wait()
            if process.returncode != 0:
                f_log.write(f"\n[ERROR] Command failed with exit code: {process.returncode}\n")
                return False
            return True
        except Exception as e:
            f_log.write(f"\n[CRITICAL SCRIPT EXCEPTION] OCCURRED: {e}\n")
            return False

def create_result_zip(zip_filename: Path, logs_dir: Path):
    print(f"\nCompressing evaluation artifacts into: {zip_filename}")
    target_extensions = {".pkl", ".csv", ".json", ".log"}
    target_dirs = ["results", "saved_results", "saved_models", "val_results", str(logs_dir)]
    files_to_zip = set()

    for dir_name in target_dirs:
        p = Path(dir_name)
        if p.exists() and p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    files_to_zip.add(Path(root) / f)

    for item in Path(".").iterdir():
        if item.is_file() and item.suffix in target_extensions:
            files_to_zip.add(item)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if any(part.startswith(".") or part in ["venv", "__pycache__"] for part in file_path.parts):
                continue
            arcname = file_path.relative_to(Path(".")) if file_path.is_absolute() else file_path
            zipf.write(file_path, arcname=str(arcname))
    print(f"Archiving complete! Bundled {len(files_to_zip)} result files.")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU ID to run on.")
    parser.add_argument("--n_hops", type=int, default=2, help="Enclosing subgraph hop depth (default: 2).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-processing evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--logs-dir", type=str, default="fixed_experiment_logs")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"fixed_run_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    # Inject the requested fixed parameters into the base configs
    selected_tasks = []
    for base_cmd in EXPERIMENT_CONFIGS:
        fixed_args = " ".join([
            "--aggregator attention",
            "--pooling target",
            "--use_layered_cache",
            "--link_pred_module transformer",
            f"--n_hops {args.n_hops}"
        ])
        selected_tasks.append(f"{base_cmd} {fixed_args}")

    total_tasks = len(selected_tasks)
    success_count, failed_tasks = 0, []

    print(f"==================================================")
    print(f"Total Fixed Runs Scheduled: {total_tasks}")
    print(f"Params: Attention Aggregator, Target Pooling, Layered Cache")
    print(f"==================================================")

    for idx, cmd_args in enumerate(selected_tasks, 1):
        try:
            prefix = extract_prefix(cmd_args)
            log_file = logs_path / f"{prefix}.log"
            full_command = f"python train_self_supervised.py {cmd_args}"

            print(f"\n[{idx}/{total_tasks}] Prefix: {prefix}\nCommand: {full_command}")
            if args.dry_run:
                continue

            # Start timing
            start_time = time.time()
            
            # Execute Model
            success = run_command(full_command, log_file, env)
            
            # End timing strictly before network notification to exclude delay
            end_time = time.time()
            run_duration = (end_time - start_time) / 60.0

            if success:
                print(f"STATUS: Success (Completed in {run_duration:.2f} mins)")
                success_count += 1
                send_ntfy_notification(
                    f"Task {idx}/{total_tasks} completed successfully in {run_duration:.2f} mins.\nPrefix: {prefix}", 
                    title="Training Complete"
                )
            else:
                print(f"STATUS: FAILED - Check log: {log_file}")
                failed_tasks.append((prefix, full_command))
                send_ntfy_notification(
                    f"Task {idx}/{total_tasks} FAILED.\nPrefix: {prefix}", 
                    title="Training Failed"
                )
                
        except Exception as task_error:
            failed_tasks.append((f"Task_{idx}", str(task_error)))
            send_ntfy_notification(f"Task_{idx} triggered exception: {task_error}", title="Script Exception")

    if not args.skip_eval and not args.dry_run:
        post_proc_log = logs_path / "post_processing_evaluation.log"
        for eval_cmd in POST_PROCESSING_COMMANDS:
            try:
                run_command(eval_cmd, post_proc_log, env)
            except Exception:
                continue

    if not args.dry_run:
        try:
            create_result_zip(zip_output_name, logs_path)
        except Exception as zip_error:
            print(f"\n[ERROR] Failed to create zip: {zip_error}")

    print(f"\nFinished. Success: {success_count} | Failed: {len(failed_tasks)}")
    if failed_tasks:
        print("Failed tasks:")
        for prefix, cmd in failed_tasks:
            print(f"  - {prefix}: {cmd}")

    if not args.dry_run:
        if failed_tasks:
            send_ntfy_notification(
                f"Ablation script finished with {success_count} successful and {len(failed_tasks)} failed task(s).",
                title="Ablation Script - FAILED",
            )
        else:
            send_ntfy_notification(
                f"Ablation script finished successfully. All {success_count} task(s) completed.",
                title="Ablation Script - Complete",
            )

if __name__ == "__main__":
    main()