#!/usr/bin/env python3
"""
Custom Fault-Tolerant Cache Ablation Benchmarking Script
Runs flat and layered cache configurations based on optimal JSON parameters.
Target Executable: train_self_supervised.py
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

# Base configurations for the datasets
EXPERIMENT_CONFIGS = [
    "-d email-Eu-core-temporal-Dept1 --use_memory --embedding_module identity --n_runs 5 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept2 --use_memory --embedding_module identity --n_runs 5 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept3 --use_memory --embedding_module identity --n_runs 5 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --n_runs 5 --n_epoch 50",
]

# Base arguments for the target transformer architecture
BASE_FLAGS = "--link_pred_module transformer --pooling mean --loss bce --aggregator last"
SUFFIX_BASE = "trans-meanpool-bce-lastaggr"

# Cache parameters mapped from global_flat_summary_flat.json and global_layered_summary_layered.json
DATASET_CACHE_CONFIGS = {
    "email-Eu-core-temporal-Dept1": {
        "flat": "--use_cache --flat_cache_max 105 --ttl 74919.0",
        "layered": "--use_layered_cache --layered_cache_h1 25 --layered_cache_h2 96 --ttl 74919.0"
    },
    "email-Eu-core-temporal-Dept2": {
        "flat": "--use_cache --flat_cache_max 105 --ttl 71945.5",
        "layered": "--use_layered_cache --layered_cache_h1 45 --layered_cache_h2 90 --ttl 71945.5"
    },
    "email-Eu-core-temporal-Dept3": {
        "flat": "--use_cache --flat_cache_max 105 --ttl 166451.5",
        "layered": "--use_layered_cache --layered_cache_h1 45 --layered_cache_h2 90 --ttl 166451.5"
    },
    "email-Eu-core-temporal-Dept4": {
        "flat": "--use_cache --flat_cache_max 105 --ttl 38687.5",
        "layered": "--use_layered_cache --layered_cache_h1 35 --layered_cache_h2 99 --ttl 38687.5"
    }
}

POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv cache_ablation_summary.csv",
]

def send_ntfy_notification(topic: str, message: str, title: str = "tgntgntgntraining", tags: str = None):
    if not topic:
        return
    try:
        url = f"https://ntfy.sh/{topic}"
        headers = {"Title": title}
        if tags:
            headers["Tags"] = tags
        req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"\n[Warning] Failed to send ntfy notification: {e}")

def run_command(command: str, log_filepath: Path, env_vars: dict) -> tuple:
    error_tail = []
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
                error_tail.append(line.strip())
                if len(error_tail) > 60:
                    error_tail.pop(0)
            process.wait()
            if process.returncode != 0:
                f_log.write(f"\n[ERROR] Command failed with exit code: {process.returncode}\n")
                return False, "\n".join(error_tail)
            return True, ""
        except Exception as e:
            f_log.write(f"\n[CRITICAL SCRIPT EXCEPTION] OCCURRED: {e}\n")
            return False, str(e)

def create_result_zip(zip_filename: Path, logs_dir: Path):
    try:
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
    except Exception as e:
        print(f"[ERROR] Failed to create global result zip: {e}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU ID to run on.")
    parser.add_argument("--dataset_filter", type=str, default=None, help="String to filter which datasets to run (e.g., 'Dept1').")
    parser.add_argument("--n_hops", type=int, default=2, help="Enclosing subgraph hop depth.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-processing evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--logs-dir", type=str, default="ablation_cache_logs")
    parser.add_argument("--ntfy-topic", type=str, default="tgntgntgntraining", help="ntfy.sh topic to receive push notifications.")
    parser.add_argument("--test-ntfy", action="store_true", help="Send a test ntfy push and wait for user confirmation before executing.")
    args = parser.parse_args()

    if args.test_ntfy:
        print(f"Sending test ntfy notification to topic '{args.ntfy_topic}'...")
        send_ntfy_notification(
            topic=args.ntfy_topic, 
            message="🔔 This is a test notification from the Cache Ablation script. The webhook is working correctly.", 
            title="TGN Alert Test", 
            tags="bell"
        )
        response = input("Did you receive the notification? Continue running experiments? (y/n): ")
        if response.lower() != 'y':
            print("Exiting as requested.")
            sys.exit(0)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"ablation_cache_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    base_configs = EXPERIMENT_CONFIGS
    if args.dataset_filter is not None:
        base_configs = [cfg for cfg in base_configs if args.dataset_filter in cfg]

    selected_tasks = []
    for base_cmd in base_configs:
        dataset_name = base_cmd.split("-d ")[1].split(" ")[0]
        cache_opts = DATASET_CACHE_CONFIGS.get(dataset_name)
        
        if not cache_opts:
            print(f"[WARNING] No cache configurations found for dataset: {dataset_name}. Skipping.")
            continue

        # Flat Configuration Task
        prefix_flat = f"{dataset_name}-{SUFFIX_BASE}-flat"
        cmd_flat = f"{base_cmd} --prefix {prefix_flat} --n_hops {args.n_hops} {BASE_FLAGS} {cache_opts['flat']}"
        selected_tasks.append((prefix_flat, cmd_flat))

        # Layered Configuration Task
        prefix_layered = f"{dataset_name}-{SUFFIX_BASE}-layered"
        cmd_layered = f"{base_cmd} --prefix {prefix_layered} --n_hops {args.n_hops} {BASE_FLAGS} {cache_opts['layered']}"
        selected_tasks.append((prefix_layered, cmd_layered))

    total_tasks = len(selected_tasks)
    success_count, failed_tasks = 0, []

    print(f"==================================================")
    print(f"Total Cache Ablation Runs Scheduled: {total_tasks}")
    print(f"Targeting Executable: train_self_supervised.py")
    print(f"==================================================")

    for idx, (prefix, cmd_args) in enumerate(selected_tasks, 1):
        log_file = logs_path / f"{prefix}.log"
        full_command = f"python train_self_supervised.py {cmd_args}"

        print(f"\n[{idx}/{total_tasks}] Prefix: {prefix}\nCommand: {full_command}")
        if args.dry_run:
            continue

        start_time = time.time()
        success, error_tail = run_command(full_command, log_file, env)
        elapsed_mins = (time.time() - start_time) / 60.0

        short_err = error_tail[-250:] if len(error_tail) > 250 else error_tail

        if success:
            print(f"STATUS: Success (Completed in {elapsed_mins:.2f} mins)")
            success_count += 1
            send_ntfy_notification(
                topic=args.ntfy_topic,
                message=f"✅ Task Completed ({idx}/{total_tasks})\nPrefix: {prefix}\nTime: {elapsed_mins:.2f} mins",
                title="TGN Success",
                tags="white_check_mark"
            )
        else:
            print(f"STATUS: FAILED - Check log: {log_file}")
            failed_tasks.append((prefix, full_command))
            send_ntfy_notification(
                topic=args.ntfy_topic,
                message=f"❌ Task Failed ({idx}/{total_tasks})\nPrefix: {prefix}\nRuntime: {elapsed_mins:.2f} mins\n\nError:\n{short_err}",
                title="TGN Error",
                tags="x"
            )

    if not args.skip_eval and not args.dry_run:
        post_proc_log = logs_path / "post_processing_evaluation.log"
        for eval_cmd in POST_PROCESSING_COMMANDS:
            try:
                run_command(eval_cmd, post_proc_log, env)
            except Exception as e:
                print(f"[Warning] Post-processing command failed: {e}")

    if not args.dry_run:
        create_result_zip(zip_output_name, logs_path)

    print(f"\nFinished. Success: {success_count} | Failed: {len(failed_tasks)}")

if __name__ == "__main__":
    main()