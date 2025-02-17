# Based on https://github.com/bjing2016/mdgen/blob/master/scripts/prep_sims.py
# but adapted to our codebase.
import argparse
import os
from multiprocessing import Pool

import mdtraj as md
import numpy as np
import pandas as pd
import tqdm

from src.utils.constants import ATOM_ENCODING

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="splits/atlas.csv")
parser.add_argument("--sim_dir", type=str, default="/data/cb/scratch/datasets/atlas")
parser.add_argument("--outdir", type=str, default="./data_atlas")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--atlas", action="store_true")
parser.add_argument("--stride", type=int, default=1)
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col="name")
names = df.index


def main():
    jobs = []
    for name in names:
        if os.path.exists(f"{args.outdir}/{name}{args.suffix}.npy"):
            continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)


if args.atlas:

    def do_job(name):
        for i in [1, 2, 3]:
            traj: md.Trajectory = md.load(
                f"{args.sim_dir}/{name}/{name}_prod_R{i}_fit.xtc",
                top=f"{args.sim_dir}/{name}/{name}.pdb",
            )
            traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != "H"], True)
            traj.center_coordinates()
            traj.superpose(traj)
            atoms_enc = np.array([ATOM_ENCODING[atom.element.symbol] for atom in traj.top.atoms])
            np.save(f"{args.outdir}/{name}_R{i}{args.suffix}_traj.npy", traj.xyz[:: args.stride])
            np.save(f"{args.outdir}/{name}_R{i}{args.suffix}_atoms.npy", atoms_enc)

else:

    def do_job(name):
        traj: md.Trajectory = md.load(
            f"{args.sim_dir}/{name}/{name}.xtc", top=f"{args.sim_dir}/{name}/{name}.pdb"
        )
        traj.center_coordinates()
        traj.superpose(traj)
        np.save(f"{args.outdir}/{name}{args.suffix}.npy", traj.xyz[:: args.stride])


if __name__ == "__main__":
    main()
