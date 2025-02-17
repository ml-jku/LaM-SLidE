# https://github.com/yaledeus/FBM

import mdtraj as md
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


def compute_pairwise_distances(traj, offset=3):
    """Compute pairwise distances excluding adjacent residues within an offset."""
    ca_indices = traj.topology.select("name CA")
    n_atoms = len(ca_indices)
    distances = []

    for i in range(n_atoms):
        for j in range(i + offset + 1, n_atoms):
            distance = np.linalg.norm(
                traj.xyz[:, ca_indices[i], :] - traj.xyz[:, ca_indices[j], :], axis=-1
            )
            distances.append(distance)

    return np.stack(distances).T


def compute_radius_of_gyration(traj):
    """Compute the radius of gyration."""
    ca_indices = traj.topology.select("name CA")
    rg = md.compute_rg(traj.atom_slice(ca_indices))
    return rg[:, np.newaxis]


def discretize_features(features, bins, pseudo_count=1e-6):
    """Discretize feature values and return histogram with pseudo counts."""
    hist, _ = np.histogram(features, bins=bins, density=True)
    hist += pseudo_count
    return hist


def discretize_features2d(features, bins, pseudo_count=1e-6):
    """Discretize features and return histogram2d with pseudo counts."""
    hist, _, _ = np.histogram2d(features[:, 0], features[:, 1], bins=bins, density=True)
    hist += pseudo_count
    return hist


def compute_phi_psi(traj):
    """Compute phi and psi (especially for alanine-dipeptide)."""
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    phi, psi = phi[:, 0], psi[:, 0]
    return phi, psi


def ramachandran_kld(phi_model, psi_model, phi_ref, psi_ref, bins=50, eps_ram=1e-10):
    # Ramachandran plot KLDs
    hist_ram_ref = np.histogram2d(
        phi_ref, psi_ref, bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    hist_ram_gen = np.histogram2d(
        phi_model, psi_model, bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    kld_ram_test = (
        np.sum(hist_ram_ref * np.log((hist_ram_ref + eps_ram) / (hist_ram_gen + eps_ram)))
        * (2 * np.pi / bins) ** 2
    )

    return kld_ram_test


def compute_js_distance(feat_ref, feat_model, bins=50):
    js_dists = []
    ndim = feat_ref.shape[1]
    for d in range(ndim):
        feat_ref_d, feat_model_d = feat_ref[:, d], feat_model[:, d]
        feat_bins = np.linspace(feat_ref_d.min(), feat_ref_d.max(), bins)
        feat_ref_d_hist = discretize_features(feat_ref_d, feat_bins)
        feat_model_d_hist = discretize_features(feat_model_d, feat_bins)
        js_dists.append(jensenshannon(feat_ref_d_hist, feat_model_d_hist))
    js = np.array(js_dists).mean()
    return js


def compute_joint_js_distance(feat0_ref, feat1_ref, feat0_model, feat1_model, bins=50):
    # Stack TIC vectors to form 2D projections
    feat2d_ref = np.vstack((feat0_ref, feat1_ref)).T
    feat2d_model = np.vstack((feat0_model, feat1_model)).T

    feat_bins = (
        np.linspace(feat0_ref.min(), feat0_ref.max(), bins),
        np.linspace(feat1_ref.min(), feat1_ref.max(), bins),
    )

    # Compute histograms for the joint distributions
    hist_ref = discretize_features2d(feat2d_ref, feat_bins)
    hist_model = discretize_features2d(feat2d_model, feat_bins)

    # Flatten histograms and compute JS distance
    hist_ref_flat = hist_ref.flatten()
    hist_model_flat = hist_model.flatten()

    # Compute the Jensen-Shannon distance
    js = jensenshannon(hist_ref_flat, hist_model_flat)

    return js


def compute_contact_matrix(traj, contact_threshold=1.0):
    alpha_carbons = traj.topology.select("name CA")
    n_residues = len(alpha_carbons)
    contact_matrix = np.zeros((n_residues, n_residues))

    distances = md.compute_distances(
        traj, np.array([(i, j) for i in alpha_carbons for j in alpha_carbons])
    )
    distances = distances.reshape((traj.n_frames, n_residues, n_residues))

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            contact_rate = np.mean(distances[:, i, j] < contact_threshold)
            contact_matrix[i, j] = contact_rate
    return contact_matrix


def compute_validity(traj, clash_threshold=0.3, bond_break_threshold=0.419):
    alpha_carbons = traj.topology.select("name CA")
    alpha_carbons_xyz = traj.xyz[:, alpha_carbons, :]
    num_atoms = alpha_carbons_xyz.shape[1]
    alpha_carbons_xyz_torch = torch.from_numpy(alpha_carbons_xyz)
    distances = torch.cdist(alpha_carbons_xyz_torch, alpha_carbons_xyz_torch).numpy()
    # judge clash
    has_clash = np.sum(distances < clash_threshold, axis=(1, 2)) - num_atoms > 0
    # judge bond break
    adjacent_distances = distances[:, np.arange(num_atoms - 1), np.arange(1, num_atoms)]
    has_bond_break = np.sum(adjacent_distances > bond_break_threshold, axis=1) > 0
    valid_conformations = ~(has_clash | has_bond_break)
    val_ca = np.mean(valid_conformations)
    return val_ca
