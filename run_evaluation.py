#!/usr/bin/env python3
"""
Master benchmarking script -- graph-transformer decoder, cached vs. uncached,
2-hop vs. 3-hop enclosing subgraphs.

Scope: baseline runs with the OLD (legacy) link predictor are already
evaluated per mentor guidance and are NOT reproduced here. This script always
uses the new graph-transformer decoder (--link_pred_module transformer) and,
for each architecture group (DYREP, JODIE, TGN_NO_MEM, TGN_TIME, TGN_ID,
TGN_SEAL, TGAT), runs every combination of:

    cache:  no cache / --use_cache / --use_layered_cache
    hops:   --n_hops 2 / --n_hops 3

--n_hops used to be hardcoded to 2 inside tgn.py regardless of the --prefix
label (the old "-3h"/"-2h" suffixes were cosmetic and never actually changed
subgraph depth). It's now a real flag on train_self_supervised.py, and this
script tests both values by default -- 16 base experiments x 3 cache modes
x 2 hop depths = 96 runs.

Dataset names match what's actually in data/ (ml_<name>.csv): calls,
CollegeMsg, email-Eu-core-temporal-Dept1..4. wiki-talk-2y is not available
and has been removed from every group that referenced it. "calls" is its
own dataset (not a stand-in for the old "sms" slot). In TGN_SEAL, CollegeMsg
is only run with the graph_attention embedding module -- the old identity
and time CollegeMsg variants there have been dropped.

--link_pred_module stays switchable via --link-pred if a legacy decoder
(dgcnn/gin/sage/gcn/merge) needs to be substituted later.

Plotting has been removed -- statistical_evaluation still runs, but
plot_tgn_seal_result / plot_results do not. All .pkl result files are bundled
into the zip at the end so plots can be regenerated later from raw data.
"""
import argparse
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

# Base configs per architecture group (baseline/no-cache variants excluded --
# already evaluated). Each entry here is expanded into cache x hop variants
# by build_variant().
#
# Dataset names match the files actually present under data/ (ml_<name>.csv):
#   calls, CollegeMsg, email-Eu-core-temporal-Dept1..4
# wiki-talk-2y is not available and has been removed from every group that
# used it (JODIE, TGN_ID, TGN_SEAL). "calls" is its own dataset, confirmed
# NOT equivalent to the old "sms" slot.
EXPERIMENT_GROUPS = {
    "DYREP": [
        "-d email-Eu-core-temporal-Dept2 --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-rnn-dept2 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-rnn-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "JODIE": [
        # wiki-talk-2y removed (not available)
        "-d CollegeMsg --use_memory --memory_updater rnn --embedding_module time --prefix jodie-rnn-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "TGN_NO_MEM": [
        "-d email-Eu-core-temporal-Dept4 --prefix tgn-no-mem-dept4 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg --prefix tgn-no-mem-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "TGN_TIME": [
        "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module time --prefix tgn-time-dept4 --n_runs 10 --n_epoch 50",
        "-d CollegeMsg --use_memory --embedding_module time --prefix tgn-time-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "TGN_ID": [
        "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --prefix tgn-id-dept4 --n_runs 10 --n_epoch 50",
        # wiki-talk-2y removed (not available)
        "-d CollegeMsg --use_memory --embedding_module identity --prefix tgn-id-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "TGN_SEAL": [
        "-d email-Eu-core-temporal-Dept1 --use_memory --embedding_module identity --prefix tgn-seal-dept1 --n_runs 10 --n_epoch 50",
        "-d email-Eu-core-temporal-Dept2 --use_memory --embedding_module identity --prefix tgn-seal-dept2 --n_runs 10 --n_epoch 50",
        "-d email-Eu-core-temporal-Dept3 --use_memory --embedding_module identity --prefix tgn-seal-dept3 --n_runs 10 --n_epoch 50",
        "-d email-Eu-core-temporal-Dept4 --use_memory --embedding_module identity --prefix tgn-seal-dept4 --n_runs 10 --n_epoch 50",
        # wiki-talk-2y removed (not available)
        "-d calls --use_memory --embedding_module identity --prefix tgn-seal-calls --n_runs 1 --n_epoch 50",
        "-d CollegeMsg --use_memory --embedding_module graph_attention --prefix tgn-seal-attn-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
    "TGAT": [
        "-d CollegeMsg --embedding_module graph_attention --prefix tgat-CollegeMsg --n_runs 10 --n_epoch 50",
    ],
}

# (suffix appended to --prefix, flag injected into the training command --
# empty string means no cache flag at all, i.e. the transformer baseline)
CACHE_VARIANTS = [
    ("nocache", ""),
    ("cache", "--use_cache"),
    ("layered-cache", "--use_layered_cache"),
]

# Enclosing-subgraph hop depth passed to --n_hops. Previously hardcoded to 2
# in tgn.py regardless of what any --prefix suffix implied.
HOP_VARIANTS = [
    ("2hop", "2"),
    ("3hop", "3"),
]

# Plotting removed intentionally -- .pkl files are archived instead and can
# be re-plotted later. statistical_evaluation is kept since it just crunches
# numbers from the .pkl files rather than producing image output.
POST_PROCESSING_COMMANDS = [
    "python -m evaluation.statistical_evaluation --csv results_summary.csv",
]


def build_variant(base_args: str, cache_suffix: str, cache_flag: str, hop_suffix: str, n_hops: str, link_pred: str, n_runs_override=None, n_epoch_override=None) -> str:
    """Take a base experiment command and produce one (cache, hop) variant of
    it, with an isolated --prefix and the requested --link_pred_module.
    cache_flag == "" produces the no-cache transformer baseline."""
    parts = base_args.split()
    if "--prefix" in parts:
        idx = parts.index("--prefix")
        parts[idx + 1] = f"{parts[idx + 1]}-{cache_suffix}-{hop_suffix}"
    if cache_flag:
        parts.append(cache_flag)
    parts.extend(["--n_hops", n_hops])
    parts.extend(["--link_pred_module", link_pred])
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
    return f"exp_{int(time.time())}"


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
    print(f"\nCompressing all evaluation artifacts into: {zip_filename}")
    # No .png/.pdf/.svg -- plotting was removed, only raw results are archived.
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
    print(f"Archiving complete! Total files bundled: {len(files_to_zip)} (.pkl results only -- replot locally later)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--group", type=str, default="ALL", choices=["ALL"] + list(EXPERIMENT_GROUPS.keys()))
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="all",
        choices=["all", "nocache", "cache", "layered-cache"],
        help="Which variant(s) to run per experiment: 'nocache' is the transformer baseline (no caching), 'cache' is the flat push cache, 'layered-cache' is the hop-partitioned cache. Default 'all' runs all three so cached vs. uncached transformer performance is directly comparable.",
    )
    parser.add_argument(
        "--link-pred",
        type=str,
        default="transformer",
        choices=["transformer", "dgcnn", "gin", "sage", "gcn", "merge"],
        help="Decoder passed as --link_pred_module to every training run.",
    )
    parser.add_argument(
        "--hops",
        type=str,
        default="all",
        choices=["all", "2hop", "3hop"],
        help="Enclosing-subgraph hop depth(s) to test via --n_hops. Default 'all' runs both 2 and 3 hops so subgraph depth is directly comparable, same pattern as --cache-mode.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--logs-dir", type=str, default="experiment_logs")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring match against the full built command. Combine with --group/--cache-mode to run one specific dataset/variant as a sanity check, e.g. --group TGN_SEAL --cache-mode layered-cache --filter Dept4",
    )
    parser.add_argument(
        "--smoke-n-runs",
        type=int,
        default=None,
        help="Override --n_runs on every selected experiment (e.g. 1) for a fast local smoke test. Does not touch the production EXPERIMENT_GROUPS config.",
    )
    parser.add_argument(
        "--smoke-n-epoch",
        type=int,
        default=None,
        help="Override --n_epoch on every selected experiment (e.g. 1-2) for a fast local smoke test.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "."

    logs_path = Path(args.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    zip_output_name = Path(f"evaluation_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    active_cache_variants = (
        CACHE_VARIANTS if args.cache_mode == "all" else [v for v in CACHE_VARIANTS if v[0] == args.cache_mode]
    )
    active_hop_variants = (
        HOP_VARIANTS if args.hops == "all" else [v for v in HOP_VARIANTS if v[0] == args.hops]
    )

    groups_to_run = EXPERIMENT_GROUPS if args.group == "ALL" else {args.group: EXPERIMENT_GROUPS[args.group]}

    selected_tasks = []
    for group_name, cmd_list in groups_to_run.items():
        for base_cmd in cmd_list:
            for cache_suffix, cache_flag in active_cache_variants:
                for hop_suffix, n_hops in active_hop_variants:
                    cmd = build_variant(
                        base_cmd, cache_suffix, cache_flag, hop_suffix, n_hops, args.link_pred,
                        n_runs_override=args.smoke_n_runs, n_epoch_override=args.smoke_n_epoch,
                    )
                    selected_tasks.append((group_name, cmd))

    if args.filter:
        selected_tasks = [(g, c) for g, c in selected_tasks if args.filter in c]

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