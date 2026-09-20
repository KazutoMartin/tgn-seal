#!/usr/bin/env python3
"""
Ablation Benchmarking Script:
- Aggregator Mechanisms: last message vs. learned attention
- Fixed Options: Targeted node pooling, focal loss, no cache
- Datasets: email-Eu-core-temporal-Dept1 through Dept4 (2 runs each)
- Notification: Sends ntfy alerts upon job completion or failure
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
    "-d email-Eu-core-temporal-Dept1 --use_memory --embedding_module identity --prefix tgn-seal-dept1 --n_runs 2 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept2 --use_memory --embedding_module identity --prefix tgn-seal-dept2 --n_runs 2 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept3 --use_memory --embedding_module identity --prefix tgn-seal-dept3 --n_runs 2 --n_epoch 50",
    "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --prefix tgn-seal-dept4 --n_runs 2 --n_epoch 50",
]

# (suffix, flag value for --aggregator)
AGGREGATOR_VARIANTS = [
    ("last", "last"),
    ("attn", "attention"),
]

POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv ablation_summary.csv",
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
        # 10s timeout prevents stalled pipelines if ntfy is down
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"\n[Warning] Failed to send ntfy notification: {e}")


def build_ablation_variant(
    base_args: str,
    agg_suffix: str,
    agg_flag_val: str,
    n_hops: int = 2,
    link_pred: str = "transformer",
    n_runs_override: int = None,
    n_epoch_override: int = None,
) -> str:
    """Constructs the command line string with custom prefixes and fixed ablation parameters."""
    parts = base_args.split()

    # Create descriptive prefix
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        parts[idx + 1] = f"{parts[idx + 1]}-{agg_suffix}-target-focal-nocache"

    # Inject dynamic ablation flags
    parts.extend(["--aggregator", agg_flag_val])
    
    # Inject fixed parameters (target pooling, focal loss, default no-cache behavior)
    parts.extend(["--pooling", "target"])
    parts.extend(["--loss", "focal"])
    parts.extend(["--link_pred_module", link_pred])
    parts.extend(["--n_hops", str(n_hops)])

    # Handle smoke run overrides
    if n_runs_override is not None and "--n_runs" in parts:
        parts[parts.index("--n_runs") + 1] = str(n_runs_override)
    if n_epoch_override is not None and "--n_epoch" in parts:
        parts[parts.index("--n_epoch") + 1] = str(n_epoch_override)

    return " ".join(parts)


def extract_prefix(args_str: str) -> str:
    parts = args_str.split()
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        return parts[idx + 1]
    return f"ablation_exp_{int(time.time())}"


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
                # Maintain a rolling buffer of the last 10 lines for concise error reporting
                error_tail.append(line.strip())
                if len(error_tail) > 10:
                    error_tail.pop(0)
                    
            process.wait()
            if process.returncode != 0:
                f_log.write(f"\n[ERROR] Command failed with exit code: {process.returncode}\n")
                return False, "\n".join(error_tail[-6:])
            return True, ""
        except Exception as e:
            f_log.write(f"\n[CRITICAL SCRIPT EXCEPTION] OCCURRED: {e}\n")
            return False, str(e)


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
    parser.add_argument(
        "--aggregator-mode",
        type=str,
        default="all",
        choices=["all", "last", "attention"],
        help="Message aggregator to run. 'all' tests both last and attention.",
    )
    parser.add_argument(
        "--dept",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Run only a single department (e.g. --dept 1). Default runs all 4 departments.",
    )
    parser.add_argument("--n_hops", type=int, default=2, help="Enclosing subgraph hop depth (default: 2).")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-processing evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--logs-dir", type=str, default="ablation_experiment_logs")
    parser.add_argument("--smoke-n-runs", type=int, default=None, help="Override --n_runs for quick smoke testing.")
    parser.add_argument("--smoke-n-epoch", type=int, default=None, help="Override --n_epoch for quick smoke testing.")
    parser.add_argument("--ntfy-topic", type=str, default="TGNTrainingStatus", help="Your ntfy.sh topic to receive push notifications.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"ablation_bundle_focal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    # Filter aggregators
    active_aggregators = (
        AGGREGATOR_VARIANTS if args.aggregator_mode == "all" else [v for v in AGGREGATOR_VARIANTS if v[1] == args.aggregator_mode]
    )

    # Filter base configs by department
    base_configs = EXPERIMENT_CONFIGS
    if args.dept is not None:
        base_configs = [cfg for cfg in base_configs if f"Dept{args.dept}" in cfg]

    selected_tasks = []
    for base_cmd in base_configs:
        for agg_suffix, agg_flag in active_aggregators:
            cmd = build_ablation_variant(
                base_args=base_cmd,
                agg_suffix=agg_suffix,
                agg_flag_val=agg_flag,
                n_hops=args.n_hops,
                link_pred="transformer",
                n_runs_override=args.smoke_n_runs,
                n_epoch_override=args.smoke_n_epoch,
            )
            selected_tasks.append(cmd)

    total_tasks = len(selected_tasks)
    success_count, failed_tasks = 0, []

    print(f"==================================================")
    print(f"Total Ablation Runs Scheduled: {total_tasks}")
    print(f"Matrix: {len(active_aggregators)} Aggregator(s) x {len(base_configs)} Dataset(s)")
    print(f"Fixed: target pooling, focal loss, no cache")
    print(f"==================================================")

    for idx, cmd_args in enumerate(selected_tasks, 1):
        prefix = extract_prefix(cmd_args)
        log_file = logs_path / f"{prefix}.log"
        full_command = f"python train_self_supervised.py {cmd_args}"

        print(f"\n[{idx}/{total_tasks}] Prefix: {prefix}\nCommand: {full_command}")
        if args.dry_run:
            continue

        start_time = time.time()
        success, error_tail = run_command(full_command, log_file, env)
        
        # Stop timer explicitly before calling network request
        elapsed_mins = (time.time() - start_time) / 60.0

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
            
            # Clamp error string output to prevent monolithic push notifications
            short_err = error_tail[-250:] if len(error_tail) > 250 else error_tail
            send_ntfy_notification(
                topic=args.ntfy_topic,
                message=f"❌ Task Failed ({idx}/{total_tasks})\nPrefix: {prefix}\nRuntime: {elapsed_mins:.2f} mins\n\nError Tail:\n{short_err}",
                title="TGN Error",
                tags="x"
            )

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

if __name__ == "__main__":
    main()