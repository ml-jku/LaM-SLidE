# Based on https://github.com/bjing2016/mdgen/blob/master/scripts/prep_sims.py
# but adapted to our codebase.
import argparse
import os

import numpy as np
import pandas as pd
import rootutils
import tqdm
from joblib import Parallel, delayed

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.traj_utils import load_traj

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="splits/atlas.csv")
parser.add_argument("--sim_dir", type=str, default="/data/cb/scratch/datasets/atlas")
parser.add_argument("--outdir", type=str, default="./data_atlas")
parser.add_argument("--n_jobs", type=int, default=1)
parser.add_argument("--stride", type=int, default=100)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col="name")
names = df.index


def do_job(name):
    traj_path = f"{args.sim_dir}/{name}/{name}.xtc"
    top_path = f"{args.sim_dir}/{name}/{name}.pdb"

    traj = load_traj(trajfile=traj_path, top=top_path)
    traj = traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != "H"], True)
    traj = traj.superpose(traj)
    traj = traj[:: args.stride]
    arr = traj.xyz

    np.savez(f"{args.outdir}/{name}-traj-arrays.npz", positions=arr)
    traj[0].save(f"{args.outdir}/{name}-traj-state0.pdb")


def main():
    jobs = [name for name in names if not os.path.exists(f"{args.outdir}/{name}-traj-arrays.npz")]

    Parallel(n_jobs=args.n_jobs)(
        delayed(do_job)(name) for name in tqdm.tqdm(jobs, total=len(jobs))
    )


if __name__ == "__main__":
    main()
