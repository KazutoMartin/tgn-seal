import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def preprocess(data_path):
    # Read CSV directly
    df_csv = pd.read_csv(data_path)

    # Only keep rows with known destination
    df_csv = df_csv[df_csv["dst"].notna()]

    u_list = df_csv["src"].astype(int).tolist()
    i_list = df_csv["dst"].astype(int).tolist()
    ts_list = df_csv["ts"].astype(float).tolist()
    label_list = [1] * len(df_csv)
    idx_list = list(range(len(df_csv)))

    # No edge features → use dummy vector
    feat_l = [np.zeros(1) for _ in range(len(df_csv))]

    return pd.DataFrame(
        {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
    ), np.array(feat_l)


def rename(df):
    ui_list = [*df.u.tolist(), *df.i.tolist()]
    unique_list = np.unique(np.array(ui_list))

    for i, row in df.iterrows():
        df.at[i, "u"] = np.where(unique_list == row["u"])[0][0]
        df.at[i, "i"] = np.where(unique_list == row["i"])[0][0]


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        rename(new_df)
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = "./data/{}.csv".format(data_name)
    OUT_DF = "./data/ml_{}.csv".format(data_name)
    OUT_FEAT = "./data/ml_{}.npy".format(data_name)
    OUT_NODE_FEAT = "./data/ml_{}_node.npy".format(data_name)

    df, feat = preprocess(PATH)
    df = df.sort_values(by="ts").reset_index(drop=True)
    new_df = reindex(df, bipartite)

    # Add a dummy first edge feature
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))  # node features

    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser("Interface for TGN data preprocessing")
parser.add_argument(
    "--data",
    type=str,
    help="Dataset name (eg. wikipedia or reddit)",
    default="wikipedia",
)
parser.add_argument(
    "--bipartite", action="store_true", help="Whether the graph is bipartite"
)

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
