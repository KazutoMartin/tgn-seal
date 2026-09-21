#!/usr/bin/env python3
"""
Custom Fault-Tolerant Ablation Benchmarking Script
Datasets: email-Eu-core-temporal-Dept1 through Dept4, calls, CollegeMsg
Hops: Fixed at 2 for all variations
Target Executable: train_self_supervised.py
"""

import argparse
import os
import subprocess
import sys
import time
import uuid
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

TELEGRAM_TARGET_CHAT_ID = "1088423022"

# Base configurations for all datasets
# CollegeMsg explicitly uses the attention embedding module, others use identity.
EXPERIMENT_CONFIGS = [
    "-d email-Eu-core-temporal-Dept1 --use_memory --embedding_module identity --n_runs 1 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept2 --use_memory --embedding_module identity --n_runs 1 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept3 --use_memory --embedding_module identity --n_runs 1 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --n_runs 1 --n_epoch 50",
    "-d calls --use_memory --embedding_module identity --n_runs 1 --n_epoch 50",
    "-d CollegeMsg --use_memory --embedding_module attention --n_runs 1 --n_epoch 50",
]

# Dataset-specific optimal capacities for the layered cache (Hop 1 and Hop 2)
LAYERED_CAPACITIES = {
    "calls": {"h1": 30, "h2": 117},
    "CollegeMsg": {"h1": 20, "h2": 88},
    "email-Eu-core-temporal-Dept1": {"h1": 100, "h2": 400},
    "email-Eu-core-temporal-Dept2": {"h1": 50, "h2": 114},
    "email-Eu-core-temporal-Dept3": {"h1": 120, "h2": 200},
    "email-Eu-core-temporal-Dept4": {"h1": 200, "h2": 688},
}

# Dataset-specific optimal capacities for the flat cache based on F2 scores
FLAT_CAPACITIES = {
    "calls": 105,
    "CollegeMsg": 105,
    "email-Eu-core-temporal-Dept1": 160,
    "email-Eu-core-temporal-Dept2": 120,
    "email-Eu-core-temporal-Dept3": 180,
    "email-Eu-core-temporal-Dept4": 180,
}

# 10 specific variations mapped to train_self_supervised.py flags
VARIATIONS = [
    {
        "suffix": "trans-nocache-last-bce-notemp-drnlfast",
        "flags": "--link_pred_module transformer --aggregator last --loss bce --drnl_version fast"
    },
    {
        "suffix": "trans-cache-last-bce-notemp-drnlfast",
        "flags": "--link_pred_module transformer --use_cache --aggregator last --loss bce --drnl_version fast"
    },
    {
        "suffix": "trans-layercache-last-bce-notemp-drnlfast",
        "flags": "--link_pred_module transformer --use_layered_cache --aggregator last --loss bce --drnl_version fast"
    },
    {
        "suffix": "dgcnn-nocache-last-bce-notemp-drnlorig",
        "flags": "--link_pred_module dgcnn --aggregator last --loss bce --drnl_version original"
    },
    {
        "suffix": "trans-nocache-last-focal-notemp-drnlfast",
        "flags": "--link_pred_module transformer --aggregator last --loss focal --drnl_version fast"
    },
    {
        "suffix": "trans-layercache-last-focal-notemp-drnlfast",
        "flags": "--link_pred_module transformer --use_layered_cache --aggregator last --loss focal --drnl_version fast"
    },
    {
        "suffix": "trans-nocache-last-bce-yestemp-drnlfast",
        "flags": "--link_pred_module transformer --aggregator last --loss bce --use_temporal_decay --drnl_version fast"
    },
    {
        "suffix": "trans-nocache-attn-bce-notemp-drnlfast",
        "flags": "--link_pred_module transformer --aggregator attention --loss bce --drnl_version fast"
    },
    {
        "suffix": "trans-layercache-attn-focal-notemp-drnlfast",
        "flags": "--link_pred_module transformer --use_layered_cache --aggregator attention --loss focal --drnl_version fast"
    },
    {
        "suffix": "trans-cache-last-focal-notemp-drnlfast",
        "flags": "--link_pred_module transformer --use_cache --aggregator last --loss focal --drnl_version fast"
    },
]

POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv detailed_ablation_summary.csv",
]

def send_ntfy_notification(topic: str, message: str, title: str = "TGNTrainingStatus", tags: str = None):
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

def send_telegram_message(bot_token: str, chat_id: str, text: str):
    """Sends a plain text message (e.g., error logs) to Telegram, catching all exceptions."""
    if not bot_token:
        return
    try:
        import urllib.parse
        # Telegram max message length is 4096 characters
        text = text[-4000:] 
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({'chat_id': chat_id, 'text': text}).encode('utf-8')
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            pass
    except Exception as e:
        print(f"[Warning] Failed to send telegram text message: {e}")

def send_telegram_document(bot_token: str, chat_id: str, file_path: Path, caption: str = ""):
    """Uploads a file to Telegram, catching all network and I/O exceptions."""
    if not bot_token or not file_path.exists():
        return

    # 1. Attempt upload via requests if available
    try:
        import requests
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": (file_path.name, f, "application/zip")},
                timeout=120
            )
        if resp.status_code != 200:
            print(f"[Warning] Telegram API returned error: {resp.text}")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] Failed to send telegram document via requests: {e}")
        return

    # 2. Pure standard library fallback using multipart/form-data
    try:
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{chat_id}\r\n'.encode())

        if caption:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption[:1024]}\r\n'.encode())

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="document"; filename="{file_path.name}"\r\n'.encode())
        body.extend(b"Content-Type: application/zip\r\n\r\n")

        with open(file_path, "rb") as f:
            body.extend(f.read())
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendDocument",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            pass
    except Exception as e:
        print(f"[Warning] Failed to send telegram document via urllib: {e}")

def zip_results_folder(zip_filename: Path, results_dir: str = "results") -> bool:
    """Zips the results folder safely, returning False on any error."""
    try:
        res_path = Path(results_dir)
        if not res_path.exists() or not res_path.is_dir():
            print(f"[Warning] Results directory '{results_dir}' not found.")
            return False
        
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(res_path):
                for f in files:
                    file_full = Path(root) / f
                    arcname = file_full.relative_to(res_path)
                    zipf.write(file_full, arcname=str(arcname))
        return True
    except Exception as e:
        print(f"[ERROR] Exception occurred during zipping: {e}")
        return False

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
                # Increased buffer to 60 lines for highly detailed Telegram crash reports
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
    parser.add_argument("--dataset_filter", type=str, default=None, help="String to filter which datasets to run (e.g., 'calls', 'Dept1', 'CollegeMsg').")
    parser.add_argument("--n_hops", type=int, default=2, help="Enclosing subgraph hop depth (Forced to 2 per spec).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-processing evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--logs-dir", type=str, default="ablation_experiment_logs")
    parser.add_argument("--ntfy-topic", type=str, default="TGNTrainingStatus", help="ntfy.sh topic to receive push notifications.")
    parser.add_argument("--telegram-token", type=str, default=os.environ.get("TELEGRAM_BOT_TOKEN", ""), help="Telegram Bot Token.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"ablation_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    base_configs = EXPERIMENT_CONFIGS
    if args.dataset_filter is not None:
        base_configs = [cfg for cfg in base_configs if args.dataset_filter in cfg]

    selected_tasks = []
    for base_cmd in base_configs:
        dataset_name = base_cmd.split("-d ")[1].split(" ")[0]
        
        c1 = LAYERED_CAPACITIES[dataset_name]["h1"]
        c2 = LAYERED_CAPACITIES[dataset_name]["h2"]
        c_flat = FLAT_CAPACITIES[dataset_name]
        
        for var in VARIATIONS:
            prefix = f"{dataset_name}-{var['suffix']}"
            cmd = f"{base_cmd} --prefix {prefix} --n_hops {args.n_hops} --flat_cache_max {c_flat} --layered_cache_h1 {c1} --layered_cache_h2 {c2} {var['flags']}"
            selected_tasks.append((prefix, cmd))

    total_tasks = len(selected_tasks)
    success_count, failed_tasks = 0, []

    print(f"==================================================")
    print(f"Total Ablation Runs Scheduled: {total_tasks}")
    print(f"Matrix: {len(VARIATIONS)} Variations x {len(base_configs)} Dataset(s)")
    print(f"Targeting Executable: train_self_supervised.py")
    if args.telegram_token:
        print(f"Telegram Delivery: Enabled (Target ID: {TELEGRAM_TARGET_CHAT_ID})")
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

        # Sub-process string for ntfy (max 250 chars)
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
            
            # Post the full 60-line crash trace to Telegram
            if args.telegram_token:
                tele_err_msg = f"❌ TGN Execution Failed ({idx}/{total_tasks})\nDataset/Prefix: {prefix}\nRuntime: {elapsed_mins:.2f} mins\n\nDetailed Trace:\n{error_tail}"
                send_telegram_message(bot_token=args.telegram_token, chat_id=TELEGRAM_TARGET_CHAT_ID, text=tele_err_msg)

        # Secure Post-Run Upload Routine
        if args.telegram_token:
            try:
                results_zip_path = Path(f"results_snapshot_{prefix}.zip")
                if zip_results_folder(results_zip_path, results_dir="results"):
                    status_str = "SUCCESS" if success else "FAILED"
                    caption = (
                        f"Run {idx}/{total_tasks} [{status_str}]\n"
                        f"Prefix: {prefix}\n"
                        f"Elapsed: {elapsed_mins:.2f} min"
                    )
                    print(f"Uploading results archive ({results_zip_path.name}) to Telegram ID {TELEGRAM_TARGET_CHAT_ID}...")
                    
                    send_telegram_document(
                        bot_token=args.telegram_token,
                        chat_id=TELEGRAM_TARGET_CHAT_ID,
                        file_path=results_zip_path,
                        caption=caption
                    )
                    
                    # Safely remove the snapshot zip after uploading
                    try:
                        results_zip_path.unlink()
                    except OSError as unlink_err:
                        print(f"[Warning] Could not delete temporary zip {results_zip_path}: {unlink_err}")
                else:
                    print(f"[Warning] Could not zip results folder for {prefix}")
                    send_ntfy_notification(args.ntfy_topic, f"⚠️ Failed to zip results directory for {prefix}.", title="TGN Zip Error", tags="warning")
            except Exception as e:
                print(f"[CRITICAL ERROR] Telegram Document Upload crashed: {e}")
                send_ntfy_notification(args.ntfy_topic, f"⚠️ Telegram document upload crashed for {prefix}: {str(e)[:150]}", title="TGN Sys Error", tags="warning")

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