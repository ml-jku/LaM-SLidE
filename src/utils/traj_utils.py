import os
import pickle  # nosec B403

import MDAnalysis as mda
import mdtraj as md
import numpy as np
from MDAnalysis.coordinates.XTC import XTCWriter

from src.utils import RankedLogger
from src.utils import residue_constants as rc
from src.utils.backbone_utils import (
    compute_contact_matrix,
    compute_joint_js_distance,
    compute_js_distance,
    compute_pairwise_distances,
    compute_phi_psi,
    compute_radius_of_gyration,
    compute_validity,
)
from src.utils.tica_utils import run_tica, tica_features

log = RankedLogger(__name__, rank_zero_only=True)


def load_traj(trajfile, top):
    _, ext = os.path.splitext(trajfile)
    if ext in [".pdb"]:
        traj = md.load(trajfile)
    elif ext in [".xtc", ".dcd"]:
        traj = md.load(trajfile, top=top)
    elif ext in [".npz"]:
        positions = np.load(trajfile)["positions"]
        traj = md.Trajectory(positions, md.load(top).topology)
    elif ext in [".npy"]:
        positions = np.load(trajfile)
        if positions.ndim == 4:
            positions = positions[0]
        traj = md.Trajectory(positions, md.load(top).topology)
    else:
        raise NotImplementedError
    return traj


def create_trajectory(positions: np.ndarray, top) -> md.Trajectory:
    """Create and process an MD trajectory from positions.

    Args:
        positions: Array of atomic positions
        top: Topology of the trajectory

    Returns:
        md.Trajectory: Processed trajectory centered and superposed
    """
    traj_model = md.Trajectory(
        positions,
        top,
    )
    traj_model = traj_model.center_coordinates()
    traj_model = traj_model.superpose(traj_model, 0)
    return traj_model


def traj_analysis(traj_model, traj_ref, lagtime: int = 1000):
    features = tica_features(traj_ref)
    feat_model = tica_features(traj_model)
    tica_model = run_tica(traj_ref, lagtime=lagtime)
    tics_ref = tica_model.transform(features)
    tics_model = tica_model.transform(feat_model)

    phi_ref, psi_ref = compute_phi_psi(traj_ref)
    phi_model, psi_model = compute_phi_psi(traj_model)
    ramachandran_js = compute_joint_js_distance(phi_ref, psi_ref, phi_model, psi_model)

    try:
        pwd_ref = compute_pairwise_distances(traj_ref)
        pwd_model = compute_pairwise_distances(traj_model)

        rg_ref = compute_radius_of_gyration(traj_ref)
        rg_model = compute_radius_of_gyration(traj_model)

        pwd_js = compute_js_distance(pwd_ref, pwd_model)
        rg_js = compute_js_distance(rg_ref, rg_model)
    except BaseException:
        pwd_js = 0
        rg_js = 0

    tic_js = compute_js_distance(tics_ref[:, :2], tics_model[:, :2])
    tic2d_js = compute_joint_js_distance(
        tics_ref[:, 0], tics_ref[:, 1], tics_model[:, 0], tics_model[:, 1]
    )

    val_ca = compute_validity(traj_model)
    try:
        contact_ref = compute_contact_matrix(traj_ref)
        contact_model = compute_contact_matrix(traj_model)
        n_residues = contact_ref.shape[0]
        rmse_contact = np.sqrt(
            2 / (n_residues * (n_residues - 1)) * np.sum((contact_ref - contact_model) ** 2)
        )
    except BaseException:
        rmse_contact = 0

    return ramachandran_js, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact


def write_trajectory(positions, elements, output_pdb="structure.pdb", output_xtc="trajectory.xtc"):
    """Convert positions and elements to PDB + XTC format.

    Parameters:
    -----------
    positions : np.ndarray
        Shape (n_frames, n_atoms, 3) array of positions in Angstroms
    elements : list
        List of atomic symbols
    """
    n_frames, n_atoms, _ = positions.shape
    with open(output_pdb, "w") as f:
        f.write("TITLE     Structure\n")
        f.write("MODEL     1\n")
        for i, (element, pos) in enumerate(zip(elements, positions[0])):
            x, y, z = pos
            f.write(
                f"ATOM  {i+1:5d}  {element:<3s} MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
            )
        f.write("TER\nENDMDL\n")

    u = mda.Universe(output_pdb)
    with XTCWriter(output_xtc, n_atoms) as w:
        for frame in range(n_frames):
            u.atoms.positions = positions[frame]
            w.write(u.atoms)


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                log.warning(f"{resi.name} {at.name} not found")
                continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:, i, j] = traj.xyz[:, at.index]
    return arr
