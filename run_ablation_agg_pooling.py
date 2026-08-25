#!/usr/bin/env python3
"""
Ablation Benchmarking Script:
- Aggregator Mechanisms: last message vs. learned attention
- Transformer Pooling: mean pooling vs. targeted node extraction
- Caching: No cache (baseline)
- Datasets: email-Eu-core-temporal-Dept1 through Dept4
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile
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

# (suffix, flag value for --pooling)
POOLING_VARIANTS = [
    ("mean", "mean"),
    ("target", "target"),
]

POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv ablation_summary.csv",
]


def build_ablation_variant(
    base_args: str,
    agg_suffix: str,
    agg_flag_val: str,
    pool_suffix: str,
    pool_flag_val: str,
    n_hops: int = 2,
    link_pred: str = "transformer",
    n_runs_override: int = None,
    n_epoch_override: int = None,
) -> str:
    """Constructs the command line string with custom prefixes and ablation parameters."""
    parts = base_args.split()

    # Create descriptive prefix: e.g., tgn-seal-dept1-last-mean-nocache-2hop
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        parts[idx + 1] = f"{parts[idx + 1]}-{agg_suffix}-{pool_suffix}-nocache-{n_hops}hop"

    # Inject ablation flags
    parts.extend(["--aggregator", agg_flag_val])
    parts.extend(["--pooling", pool_flag_val])
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
    parser.add_argument(
        "--aggregator-mode",
        type=str,
        default="all",
        choices=["all", "last", "attention"],
        help="Message aggregator to run. 'all' tests both last and attention.",
    )
    parser.add_argument(
        "--pooling-mode",
        type=str,
        default="all",
        choices=["all", "mean", "target"],
        help="Transformer pooling strategy to run. 'all' tests both mean and target.",
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
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"ablation_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    # Filter aggregators and pooling variants
    active_aggregators = (
        AGGREGATOR_VARIANTS if args.aggregator_mode == "all" else [v for v in AGGREGATOR_VARIANTS if v[1] == args.aggregator_mode]
    )
    active_poolings = (
        POOLING_VARIANTS if args.pooling_mode == "all" else [v for v in POOLING_VARIANTS if v[1] == args.pooling_mode]
    )

    # Filter base configs by department
    base_configs = EXPERIMENT_CONFIGS
    if args.dept is not None:
        base_configs = [cfg for cfg in base_configs if f"Dept{args.dept}" in cfg]

    selected_tasks = []
    for base_cmd in base_configs:
        for agg_suffix, agg_flag in active_aggregators:
            for pool_suffix, pool_flag in active_poolings:
                cmd = build_ablation_variant(
                    base_args=base_cmd,
                    agg_suffix=agg_suffix,
                    agg_flag_val=agg_flag,
                    pool_suffix=pool_suffix,
                    pool_flag_val=pool_flag,
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
    print(f"Matrix: {len(active_aggregators)} Aggregator(s) x {len(active_poolings)} Pooling(s) x {len(base_configs)} Dataset(s)")
    print(f"==================================================")

    for idx, cmd_args in enumerate(selected_tasks, 1):
        try:
            prefix = extract_prefix(cmd_args)
            log_file = logs_path / f"{prefix}.log"
            full_command = f"python train_self_supervised.py {cmd_args}"

            print(f"\n[{idx}/{total_tasks}] Prefix: {prefix}\nCommand: {full_command}")
            if args.dry_run:
                continue

            start_time = time.time()
            if run_command(full_command, log_file, env):
                print(f"STATUS: Success (Completed in {(time.time() - start_time) / 60.0:.2f} mins)")
                success_count += 1
            else:
                print(f"STATUS: FAILED - Check log: {log_file}")
                failed_tasks.append((prefix, full_command))
        except Exception as task_error:
            failed_tasks.append((f"Task_{idx}", str(task_error)))

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