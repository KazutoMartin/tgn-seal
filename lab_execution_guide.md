# Lab Server Execution Guide

Four stages: **check the environment → dry-run → small real test → full background run.**
Don't skip a stage — each one catches a different class of problem, and the failure modes look identical from the outside (a hung process) but have completely different causes.

---

## Stage 0 — Environment check (no GPU, no training, ~1 minute)

```bash
# 1. Confirm the four project files are the right versions
grep -n "n_hops" model/tgn.py                     # should show self.n_hops used, not hardcoded 2
grep -n "link_pred_module" train_self_supervised.py | head -3   # should show "transformer" as a choice/default
grep -n "HOP_VARIANTS" run_evaluation.py           # should exist
grep -n "def paths_for" evaluation/statistical_evaluation.py    # should take (exp, cache_variant, hop_variant)

# 2. Confirm the dataset files are actually there, under the right names
ls data/ | grep -E "ml_(calls|CollegeMsg|email-Eu-core-temporal-Dept[1-4])"
# Expect 18 files: 6 dataset names x 3 files each (.csv, .npy, _node.npy)

# 3. Confirm GPU visibility and pick which physical GPU to use
nvidia-smi
# note the index of a free GPU (0, 1, 2...) for --gpu below

# 4. Confirm Python deps import cleanly
python -c "import torch, torch_geometric, sklearn, scipy, pandas; print('deps OK, CUDA:', torch.cuda.is_available())"
```

If any of these fail, stop here — nothing downstream will work and the error will be much harder to diagnose once training is in the mix.

---

## Stage 1 — Dry run (no GPU, no training, ~5 seconds)

Prints every command the full sweep would run, without executing any of them. This is where you catch typos in dataset names, wrong group names, or a misbuilt matrix — for free.

```bash
python run_evaluation.py --group ALL --dry-run
```

You should see **96 lines** starting with `Command:`. Count them:
```bash
python run_evaluation.py --group ALL --dry-run 2>&1 | grep -c "^Command:"
```

Skim a few — check the dataset name, `--prefix`, cache flag, `--n_hops`, and `--link_pred_module transformer` all look right for a couple of different groups.

---

## Stage 2 — Small real test (uses GPU, ~1-5 minutes)

Now actually run something, but tiny — 1 run, 1 epoch, one narrow slice — to confirm the whole pipeline works end to end: data loads, GPU is used, cache activates, a checkpoint gets written, and the analysis script can read it back.

```bash
python run_evaluation.py --group TGN_TIME --cache-mode layered-cache --hops 3hop --filter Dept4 \
    --smoke-n-runs 1 --smoke-n-epoch 1
```

Then verify, in order:

```bash
# 1. Did it actually use the GPU? (not silently fall back to CPU)
grep "Using GPU\|CUDA NOT AVAILABLE" experiment_logs/tgn-time-dept4-layered-cache-3hop.log

# 2. Did a result file get written?
ls -la results/ | grep tgn-time-dept4-layered-cache-3hop

# 3. Did the analysis script pick it up correctly?
PYTHONPATH=. python -m evaluation.statistical_evaluation --csv /tmp/smoketest.csv
cat /tmp/smoketest.csv
```

If all three check out, the pipeline is sound — the only thing changing in the full run is scale, not correctness. If GPU logging is missing or the CSV comes up empty, stop and fix it now; running the full 96-experiment sweep on a broken pipeline just wastes a lab session finding out slowly.

Clean up the smoke-test artifacts before the real run so they don't get swept into your final `.pkl`/`.csv` outputs:
```bash
rm -f results/tgn-time-dept4-layered-cache-3hop*.pkl
rm -f experiment_logs/tgn-time-dept4-layered-cache-3hop.log
```

---

## Stage 3 — The full run, backgrounded with `nohup`

Training 96 experiments (most at `--n_runs 10 --n_epoch 50`) will run far longer than any SSH session should stay open. `nohup` (no hang up) keeps the process alive after you disconnect; `&` backgrounds it so your terminal is free; redirecting output to a file lets you check on it without staying attached.

```bash
nohup python run_evaluation.py --group ALL --gpu 0 > nohup_run.log 2>&1 &
```

Immediately after, note the process ID — you'll want it to check status or kill it later:
```bash
echo $! > run.pid
cat run.pid
```

You can now safely close the SSH session; the run keeps going.

### Checking on it later

```bash
# Is it still running?
ps -p $(cat run.pid)

# Tail the overall progress (which experiment number is running, success/fail counts)
tail -f nohup_run.log

# How many experiments have finished (successfully or not)?
grep -c "^STATUS:" nohup_run.log

# Any GPU fallbacks anywhere so far?
grep -L "Using GPU" experiment_logs/*.log

# Which experiments have failed so far?
grep "STATUS: FAILED" nohup_run.log
```

### Stopping it if you need to

```bash
kill $(cat run.pid)
```
This kills the orchestrator; if a `train_self_supervised.py` subprocess is mid-run it should exit too since `run_evaluation.py` runs it in the foreground of its own loop, not detached.

### When it's done

```bash
ls -la evaluation_bundle_*.zip     # the archive with everything you need to pull off the server
tail -20 nohup_run.log              # final success/fail counts
```
Pull it back to your machine:
```bash
scp <user>@<lab-server>:<path>/evaluation_bundle_*.zip .
```

### If you'd rather split the run across sessions instead of one 96-run block

Useful if the server has a job-time limit, or you just want incremental checkpoints of progress you can inspect between groups:
```bash
nohup python run_evaluation.py --group DYREP      --gpu 0 > nohup_dyrep.log      2>&1 &
# wait for it to finish, then:
nohup python run_evaluation.py --group JODIE       --gpu 0 > nohup_jodie.log      2>&1 &
# ...and so on for TGN_NO_MEM, TGN_TIME, TGN_ID, TGN_SEAL, TGAT
```
Each call still generates its own `evaluation_bundle_*.zip` — grab the last one (or run `statistical_evaluation.py` manually once everything's done, since it reads from `results/` regardless of which `run_evaluation.py` call produced each file).

---

## Full command reference

### `run_evaluation.py`

| Flag | Default | What it does |
|---|---|---|
| `--gpu` | `"0"` | Physical GPU index, sets `CUDA_VISIBLE_DEVICES` for the subprocess |
| `--group` | `ALL` | One of `DYREP`, `JODIE`, `TGN_NO_MEM`, `TGN_TIME`, `TGN_ID`, `TGN_SEAL`, `TGAT`, or `ALL` for all seven |
| `--cache-mode` | `all` | `all` \| `nocache` \| `cache` \| `layered-cache` — which cache variant(s) to generate |
| `--hops` | `all` | `all` \| `2hop` \| `3hop` — which subgraph hop depth(s) to generate (`--n_hops 2`/`3`) |
| `--link-pred` | `transformer` | `transformer` \| `dgcnn` \| `gin` \| `sage` \| `gcn` \| `merge` — decoder passed to every run |
| `--filter` | none | Substring match against the full built command — narrows to specific datasets/architectures, e.g. `--filter Dept4` |
| `--smoke-n-runs` | none | Override `--n_runs` on every selected experiment (fast local testing) |
| `--smoke-n-epoch` | none | Override `--n_epoch` on every selected experiment (fast local testing) |
| `--skip-eval` | off | Skip `statistical_evaluation` after training |
| `--dry-run` | off | Print commands only, run nothing |
| `--logs-dir` | `experiment_logs` | Where per-run logs go |

Every flag composes — `--group TGN_SEAL --cache-mode layered-cache --hops 3hop --filter CollegeMsg --smoke-n-runs 1` all stack together to narrow the exact slice you want.

### `statistical_evaluation.py`

| Flag | Default | What it does |
|---|---|---|
| `--csv` | none | Write a flat summary table (experiment × hop × cache) to the given path, e.g. `--csv results_summary.csv` |

Run standalone any time after training (reads whatever `.pkl` files exist in `results/`, partial results are fine):
```bash
PYTHONPATH=. python -m evaluation.statistical_evaluation --csv results_summary.csv
```

### `train_self_supervised.py` (called automatically by `run_evaluation.py` — you shouldn't need to invoke this directly, but useful to know what it accepts)

| Flag | Default | What it does |
|---|---|---|
| `--link_pred_module` | `transformer` | `transformer` \| `dgcnn` \| `gin` \| `sage` \| `gcn` \| `merge` |
| `--use_cache` | off | Flat incremental push cache |
| `--use_layered_cache` | off | Hop-partitioned multi-layer cache (implies caching is on) |
| `--n_hops` | `2` | Enclosing-subgraph depth passed to the link predictor |