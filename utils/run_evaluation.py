#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# ==============================================================================
# EXPERIMENT CONFIGURATIONS
# ==============================================================================
EXPERIMENT_GROUPS = {
    "DYREP": [
        "-d email_dept2 --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-rnn-email2 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-rnn-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
    "JODIE": [
        "-d wiki-talk-2y --use_memory --memory_updater rnn --embedding_module time --prefix jodie-rnn-wiki-talk-2y --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --memory_updater rnn --embedding_module time --prefix jodie-rnn-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
    "TGN_NO_MEM": [
        "-d email_dept4 --prefix tgn-no-mem-email4 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --prefix tgn-no-mem-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
    "TGN_TIME": [
        "-d email_dept4 --use_memory --embedding_module time --prefix tgn-time-email4 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --embedding_module time --prefix tgn-time-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
    "TGN_ID": [
        "-d email_dept4 --use_memory --embedding_module identity --prefix tgn-id-email4 --n_runs 10 --n_epoch 50",
        "-d wiki-talk-2y --use_memory --embedding_module identity --prefix tgn-id-wiki-talk-2y --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --embedding_module identity --prefix tgn-id-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
    "TGN_SEAL": [
        "-d email_dept1 --use_memory --embedding_module identity --prefix tgn-seal-email1-3h --n_runs 10 --n_epoch 50",
        "-d email_dept2 --use_memory --embedding_module identity --prefix tgn-seal-email2-3h --n_runs 10 --n_epoch 50",
        "-d email_dept3 --use_memory --embedding_module identity --prefix tgn-seal-email3-3h --n_runs 10 --n_epoch 50",
        "-d email_dept4 --use_memory --embedding_module identity --prefix tgn-seal-email4-3h --n_runs 10 --n_epoch 50",
        "-d wiki-talk-2y --use_memory --embedding_module identity --prefix tgn-seal-wiki-talk-2y-3h --n_runs 10 --n_epoch 50",
        "-d sms --use_memory --embedding_module identity --prefix tgn-seal-sms-2h --n_runs 1 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --embedding_module identity --prefix tgn-seal-id-CollegeMsg-2m-2h --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --embedding_module time --prefix tgn-seal-time-CollegeMsg-2m-2h --n_runs 10 --n_epoch 50",
        "-d CollegeMsg-2m --use_memory --embedding_module graph_attention --prefix tgn-seal-attn-CollegeMsg-2m-2h --n_runs 10 --n_epoch 50",
    ],
    "TGAT": [
        "-d CollegeMsg-2m --embedding_module graph_attention --prefix tgat-CollegeMsg-2m --n_runs 10 --n_epoch 50",
    ],
}

# Full post-processing suite executed in sequential order
POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation",
    "python -m plots.plot_tgn_seal_result",
    "python -m plots.plot_results",
]


def apply_cache_flag(cmd_args: str, enable_cache: bool) -> str:
    """Appends --use_cache and adjusts prefix if caching is enabled."""
    if not enable_cache:
        return cmd_args

    parts = cmd_args.split()
    if "--use_cache" not in parts:
        parts.append("--use_cache")
        
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        parts[idx + 1] = f"{parts[idx + 1]}-cached"

    return " ".join(parts)


def extract_prefix(args_str: str) -> str:
    """Extracts the prefix parameter from an argument string for naming logs."""
    parts = args_str.split()
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        return parts[idx + 1]
    return f"exp_{int(time.time())}"


def run_command(command: str, log_filepath: Path, env_vars: dict) -> bool:
    """Runs a shell command and appends output live to a log file. Returns True if successful."""
    with open(log_filepath, "a") as f_log:
        f_log.write(f"\n=== Executing: {command} ===\n")
        f_log.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f_log.write("=" * 80 + "\n\n")
        f_log.flush()

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_vars,
            )

            # Stream output directly to file
            for line in process.stdout:
                f_log.write(line)
                f_log.flush()

            process.wait()
            
            # Non-zero return code means the training crashed (e.g., out of memory, dataset missing)
            if process.returncode != 0:
                f_log.write(f"\n[ERROR] Command failed with exit code: {process.returncode}\n")
                return False
                
            return True
            
        except Exception as e:
            f_log.write(f"\n[CRITICAL SCRIPT EXCEPTION] OCCURRED: {e}\n")
            return False


def create_result_zip(zip_filename: Path, logs_dir: Path):
    """Gathers all logs, pickle files (.pkl), generated images, and result folders."""
    print(f"\nCompressing all evaluation artifacts into: {zip_filename}")
    
    # Target file extensions to ensure all pickle and plot outputs are captured
    target_extensions = {".pkl", ".png", ".pdf", ".svg", ".csv", ".json", ".log"}
    target_dirs = ["results", "plots", "saved_results", "saved_models", "val_results", str(logs_dir)]

    files_to_zip = set()

    # 1. Gather files from designated directories
    for dir_name in target_dirs:
        p = Path(dir_name)
        if p.exists() and p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    files_to_zip.add(Path(root) / f)

    # 2. Gather any .pkl or image files generated directly in the root directory
    for item in Path(".").iterdir():
        if item.is_file() and item.suffix in target_extensions:
            files_to_zip.add(item)

    # 3. Create compressed package
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            # Avoid zipping virtualenv or hidden folders
            if any(part.startswith(".") or part in ["venv", "__pycache__"] for part in file_path.parts):
                continue
            arcname = file_path.relative_to(Path(".")) if file_path.is_absolute() else file_path
            zipf.write(file_path, arcname=str(arcname))

    print(f"Archiving complete! Total files bundled: {len(files_to_zip)}")


def main():
    parser = argparse.ArgumentParser(description="Lab Server Benchmark Runner for Dynamic Graph LP")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use (e.g., '0')")
    parser.add_argument(
        "--group",
        type=str,
        default="ALL",
        choices=["ALL"] + list(EXPERIMENT_GROUPS.keys()),
        help="Target experiment group to execute",
    )
    parser.add_argument("--use-cache", action="store_true", help="Enable subgraph caching (--use_cache)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-processing evaluation and plotting")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--logs-dir", type=str, default="experiment_logs", help="Directory to store console logs")
    args = parser.parse_args()

    # Set up environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."  # Ensures 'python -m evaluation...' resolves root modules

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_output_name = Path(f"evaluation_bundle_{timestamp}.zip")

    # Filter task list
    if args.group == "ALL":
        selected_tasks = [(group, cmd) for group, cmd_list in EXPERIMENT_GROUPS.items() for cmd in cmd_list]
    else:
        selected_tasks = [(args.group, cmd) for cmd in EXPERIMENT_GROUPS[args.group]]

    total_tasks = len(selected_tasks)
    print("=" * 80)
    print(f"Starting Dynamic Graph Benchmark Suite | Tasks: {total_tasks} | GPU: {args.gpu}")
    print(f"Caching: {args.use_cache} | Logs Directory: {logs_path.resolve()}")
    print("=" * 80)

    success_count = 0
    failed_tasks = []

    # --------------------------------------------------------------------------
    # STAGE 1: RUN MODEL TRAINING & EVALUATION
    # --------------------------------------------------------------------------
    for idx, (group_name, raw_cmd_args) in enumerate(selected_tasks, 1):
        try:
            cmd_args = apply_cache_flag(raw_cmd_args, args.use_cache)
            prefix = extract_prefix(cmd_args)
            log_file = logs_path / f"{prefix}.log"
            full_command = f"python train_self_supervised.py {cmd_args}"

            print(f"\n[{idx}/{total_tasks}] Running [{group_name}] -> Prefix: {prefix}")
            print(f"Command: {full_command}")

            if args.dry_run:
                print(f"[DRY-RUN] Would execute and log to: {log_file}")
                continue

            start_time = time.time()
            ok = run_command(full_command, log_file, env)
            elapsed_min = (time.time() - start_time) / 60.0

            if ok:
                print(f"STATUS: Success (Completed in {elapsed_min:.2f} mins)")
                success_count += 1
            else:
                print(f"STATUS: FAILED - Check log: {log_file}")
                failed_tasks.append((prefix, full_command))
                
        except Exception as task_error:
            # Absolute fallback if anything in the loop manipulation fails
            print(f"STATUS: CRITICAL FAILURE on task index {idx} - {task_error}")
            failed_tasks.append((f"Task_Index_{idx}", str(task_error)))
            continue # Move on to the next task regardless

    # --------------------------------------------------------------------------
    # STAGE 2: RUN STATISTICAL EVALUATION & PLOTTING SCRIPTS
    # --------------------------------------------------------------------------
    if not args.skip_eval and not args.dry_run:
        print("\n" + "=" * 80)
        print("STAGE 2: Running Statistical Evaluations & Plot Generation")
        print("=" * 80)
        
        post_proc_log = logs_path / "post_processing_evaluation.log"
        for eval_cmd in POST_PROCESSING_COMMANDS:
            try:
                print(f"Executing: {eval_cmd}")
                proc_ok = run_command(eval_cmd, post_proc_log, env)
                if proc_ok:
                    print("  └─ STATUS: Completed successfully.")
                else:
                    print(f"  └─ WARNING: Command failed. Check {post_proc_log}")
            except Exception as eval_error:
                print(f"  └─ CRITICAL FAILURE on {eval_cmd} - {eval_error}")
                continue # Ensure one failed plot script doesn't crash the next

    # --------------------------------------------------------------------------
    # STAGE 3: COMPRESS ALL PKL, LOGS, AND PLOTS
    # --------------------------------------------------------------------------
    if not args.dry_run:
        try:
            create_result_zip(zip_output_name, logs_path)
        except Exception as zip_error:
            print(f"\n[ERROR] Failed to create zip archive: {zip_error}")

    # --------------------------------------------------------------------------
    # FINAL SUMMARY FOR LAB OPERATOR
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("      BENCHMARK SUITE EXECUTION FINISHED")
    print("#" * 80)
    print(f"Total Tasks Attempted: {total_tasks}")
    print(f"Successful Runs:       {success_count}")
    print(f"Failed Runs:           {len(failed_tasks)}")

    if not args.dry_run:
        print("\n" + "=" * 80)
        print(" INSTRUCTIONS FOR LAB OPERATOR:")
        print(" Please send the following zip package back to the user:")
        print(f" -> {zip_output_name.resolve()}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()