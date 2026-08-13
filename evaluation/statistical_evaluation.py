import pickle

import numpy as np
from scipy.stats import wilcoxon

from utils.build_result_paths import build_models_result_paths

# =====================================================
# Configuration
# =====================================================

NUM_RUNS = 1
AVAILABLE_DATASET = ["calls", "CollegeMsg-2m", "email1", "email2", "email3", "email4"]
DATASET = AVAILABLE_DATASET[0]

AVAILABLE_KEY = ["val_aps", "new_nodes_val_aps"]
AP_KEY = AVAILABLE_KEY[0]
MODE = "mean"
ALPHA = 0.01

models = {
    "DyRep": ("dyrep", "rnn", None),
    "JODIE": ("jodie", "rnn", None),
    "TGAT": ("tgat", None, None),
    "TGN_NoMem": ("tgn", "no-mem", None),
    "TGN_Id": ("tgn", "id", None),
    "TGN_Time": ("tgn", "time", None),
    "TGN_SEAL": ("tgn", "seal", "2h"),
    # "TGN_SEAL": ("tgn", "seal-attn", "2h"),
    # "TGN_SEAL_3": ("tgn", "seal", "3h"),
}

results = build_models_result_paths(DATASET, NUM_RUNS, models)


# =====================================================
# Helper Functions
# =====================================================

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_ap(res, key=AP_KEY, mode="best"):
    aps = np.asarray(res[key])

    if mode == "best":
        return aps.max()
    elif mode == "mean":
        return aps.mean()
    else:
        raise ValueError("mode must be 'best' or 'mean'")


def collect_model_aps(file_list, mode):
    return np.array([
        extract_ap(read_pickle(p), mode=mode)
        for p in file_list
    ])


def extract_training_stats(res):
    epoch_times = np.asarray(res["epoch_times"], dtype=float)

    num_epochs = len(epoch_times)
    mean_epoch_time = epoch_times.mean()
    total_training_time = epoch_times.sum()

    return mean_epoch_time, total_training_time, num_epochs


# =====================================================
# Main Evaluation
# =====================================================

def main():
    print("=" * 60)
    print(f"AP evaluation mode: {MODE.upper()}")
    print(f"AP source key: {AP_KEY}")
    print("=" * 60)

    # Collect APs
    model_aps = {
        name: collect_model_aps(paths, MODE)
        for name, paths in results.items()
    }

    # Print mean ± std
    print(f"\nMean ± Std AP over {NUM_RUNS} runs:\n")
    for name, aps in model_aps.items():
        print(f"{name:12s}: {aps.mean():.4f} ± {aps.std():.4f}")

    # Statistical comparison
    print("\n" + "=" * 60)
    print("Paired Wilcoxon Signed-Rank Test")
    print("H1: TGN-SEAL > Baseline")
    print("=" * 60 + "\n")

    tgn_seal_ap = model_aps["TGN_SEAL"]

    for baseline in ["DyRep", "JODIE", "TGAT", "TGN_NoMem", "TGN_Id", "TGN_Time"]:
        baseline_ap = model_aps[baseline]

        stat, p_value = wilcoxon(
            tgn_seal_ap,
            baseline_ap,
            alternative="greater"
        )

        significant = "✅ YES" if p_value < ALPHA else "❌ NO"

        print(f"TGN-SEAL vs {baseline}")
        print(f"  p-value      : {p_value:.4e}")
        print(f"  significant? : {significant}")
        print("-" * 45)

    training_stats = {}

    for name, paths in results.items():
        stats = np.array([
            extract_training_stats(read_pickle(p))
            for p in paths
        ])

        training_stats[name] = {
            "epoch_time": stats[:, 0],
            "total_time": stats[:, 1],
            "num_epochs": stats[:, 2],
        }

    print("\n" + "=" * 75)
    print("Training Time Statistics")
    print("=" * 75)

    for name, stats in training_stats.items():
        epoch_time = stats["epoch_time"]
        total_time = stats["total_time"]
        num_epochs = stats["num_epochs"]

        print(f"\n{name}")
        print(f"  Mean epoch time  : {epoch_time.mean():.4f} ± {epoch_time.std():.4f} sec")
        print(f"  Mean total time  : {total_time.mean():.2f} ± {total_time.std():.2f} sec")
        print(f"  Mean # epochs    : {num_epochs.mean():.2f} ± {num_epochs.std():.2f}")


if __name__ == "__main__":
    main()
