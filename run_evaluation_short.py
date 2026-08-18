#!/usr/bin/env python3
"""
Short/focused benchmarking run.

Scope, fixed by design (not meant to be widened via flags -- use
run_evaluation.py directly for that):
    - datasets: email-Eu-core-temporal-Dept1, Dept2, Dept3, Dept4 only
      (every other dataset -- CollegeMsg, calls -- is excluded)
    - --n_runs 2 (overriding whatever EXPERIMENT_GROUPS specifies) 
    - --n_epoch 50

Still sweeps cache mode (nocache/cache/layered-cache) by default, same as
run_evaluation.py -- only hop depth is fixed here, to 2 hops, since that's
the axis being cut down for this short run (pass --hops 3hop or --hops all
to override).

Imports EXPERIMENT_GROUPS, CACHE_VARIANTS, HOP_VARIANTS, build_variant,
run_command, and create_result_zip directly from run_evaluation.py instead
of duplicating them, so the two scripts can never drift out of sync on
naming or behavior. Requires run_evaluation.py to be in the same directory.
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from run_evaluation import (
    EXPERIMENT_GROUPS,
    CACHE_VARIANTS,
    HOP_VARIANTS,
    build_variant,
    extract_prefix,
    run_command,
    create_result_zip,
)

N_RUNS = 2
N_EPOCH = 50
DATASET_FILTER = "Dept"  # matches email-Eu-core-temporal-Dept1..4 only


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--cache-mode", type=str, default="all", choices=["all", "nocache", "cache", "layered-cache"]
    )
    parser.add_argument("--hops", type=str, default="2hop", choices=["all", "2hop", "3hop"])
    parser.add_argument(
        "--link-pred", type=str, default="transformer",
        choices=["transformer", "dgcnn", "gin", "sage", "gcn", "merge"],
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--logs-dir", type=str, default="experiment_logs")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"evaluation_bundle_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    active_cache_variants = (
        CACHE_VARIANTS if args.cache_mode == "all" else [v for v in CACHE_VARIANTS if v[0] == args.cache_mode]
    )
    active_hop_variants = HOP_VARIANTS if args.hops == "all" else [v for v in HOP_VARIANTS if v[0] == args.hops]

    selected_tasks = []
    for group_name, cmd_list in EXPERIMENT_GROUPS.items():
        for base_cmd in cmd_list:
            if DATASET_FILTER not in base_cmd:
                continue
            for cache_suffix, cache_flag in active_cache_variants:
                for hop_suffix, n_hops in active_hop_variants:
                    cmd = build_variant(
                        base_cmd, cache_suffix, cache_flag, hop_suffix, n_hops, args.link_pred,
                        n_runs_override=N_RUNS, n_epoch_override=N_EPOCH,
                    )
                    selected_tasks.append((group_name, cmd))

    total_tasks, success_count, failed_tasks = len(selected_tasks), 0, []

    for idx, (group_name, cmd_args) in enumerate(selected_tasks, 1):
        try:
            prefix = extract_prefix(cmd_args)
            log_file = logs_path / f"{prefix}.log"
            full_command = f"python train_self_supervised.py {cmd_args}"

            print(f"\n[{idx}/{total_tasks}] Running [{group_name}] -> Prefix: {prefix}\nCommand: {full_command}")
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
        try:
            run_command(
                "python -m evaluation.statistical_evaluation --csv results_summary_short.csv", post_proc_log, env
            )
        except Exception:
            pass

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