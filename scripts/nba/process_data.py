import argparse
import logging
import os

import numpy as np
import pandas as pd
import rootutils
from einops import rearrange
from joblib import Parallel, delayed
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/social_vae_data/nba/score/train",
)
parser.add_argument(
    "--outdir",
    type=str,
    default="data/social_vae_data/nba_processed/score/train",
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=1,
)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


def process_file(name: str, data_path: str, output_filename: str) -> None:
    logging.info(f"Processing {name}")
    input_filename = f"{data_path}/{name}"
    output_filename = f"{output_filename}/{name.replace('.txt', '.npz')}"
    df = pd.read_csv(
        input_filename, sep=" ", header=None, names=["frame", "agent_id", "x", "y", "group"]
    )
    df.sort_values(by=["frame", "agent_id"])

    df["team"] = -1

    player_data = df[df["group"] == "PLAYER"]
    player_ranks = player_data.groupby("frame").cumcount()
    df.loc[player_data.index, "team"] = (player_ranks >= 5).astype(int)
    df[["agent_id", "team"]] = df[["agent_id", "team"]] + 1
    df["group"] = df["group"].map({"PLAYER": 0, "BALL": 1})

    # Make unique agent ids from 0 to N-1
    agent_rank = {agent_id: rank for rank, agent_id in enumerate(df["agent_id"].unique())}
    df["agent_id"] = df["agent_id"].map(agent_rank)

    T = df["frame"].nunique()
    data = rearrange(df.values, "(T A) D -> T A D", T=T)
    data_dict = {
        "frame_id": data[..., 0],
        "agent_id": data[..., 1],
        "pos": data[..., 2:4],
        "group": data[..., 4],
        "team": data[..., 5],
    }
    logging.info(f"Processed {name} with shape {data.shape}")
    np.savez(output_filename, **data_dict)


def main():
    logging.info(f"Processing {args.data_dir} with {args.n_jobs} jobs")
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(f, args.data_dir, args.outdir)
        for f in tqdm(os.listdir(args.data_dir))
    )


if __name__ == "__main__":
    main()
