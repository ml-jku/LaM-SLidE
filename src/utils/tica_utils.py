# https://github.com/yaledeus/FBM/blob/main/utils/tica_utils.py

import deeptime as dt
import emcee
import mdtraj as md
import numpy as np
from matplotlib.colors import LogNorm

SELECTION = "symbol == C or symbol == N or symbol == S"


def distances(xyz):
    distance_matrix_ca = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def wrap(array):
    return np.sin(array), np.cos(array)


def tica_features(trajectory, use_dihedrals=True, use_distances=True, selection=SELECTION):
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    # n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    else:
        return []


def run_tica(trajectory, lagtime=500, dim=40):
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def ramachandran_kld(phi_gen, psi_gen, phi_md, psi_md):
    # Ramachandran plot
    # Compute KLDs
    nbins_ram = 64
    eps_ram = 1e-10
    hist_ram_md = np.histogram2d(
        phi_md, psi_md, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    hist_ram_gen = np.histogram2d(
        phi_gen, psi_gen, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    kld_ram_test = (
        np.sum(hist_ram_md * np.log((hist_ram_md + eps_ram) / (hist_ram_gen + eps_ram)))
        * (2 * np.pi / nbins_ram) ** 2
    )

    return kld_ram_test


def get_vamp2(traj, lag):
    feats = tica_features(traj)
    vamp = dt.decomposition.VAMP(lag).fit_fetch(feats)
    vamp2_score = vamp.score(2)

    return vamp2_score


def ESS(TIC, axis=0):
    """
    :param TIC: TIC of T sampling steps, (T, tic_dim)
    :return: effective sample size
    """
    T = TIC.shape[0]
    tau = emcee.autocorr.integrated_time(TIC[:, axis], quiet=True)[0]

    return T / tau


def plot_tic01(ax, tics, name, tics_lims, cmap="viridis"):
    # Calculate bin edges explicitly using the reference limits
    bins_x = np.linspace(tics_lims[:, 0].min(), tics_lims[:, 0].max(), 100)
    bins_y = np.linspace(tics_lims[:, 1].min(), tics_lims[:, 1].max(), 100)

    # Use these same bin edges for both plots
    _ = ax.hist2d(tics[:, 0], tics[:, 1], bins=[bins_x, bins_y], norm=LogNorm(), cmap=cmap)

    ax.set_xlabel("tic0")
    ax.set_ylabel("tic1")
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
    ax.set_title(f"{name}")


def plot_free_energy(ax, tics, xlabel, title, tics_ref, label=None, axis=0):
    min_val = tics_ref[:, axis].min()
    max_val = tics_ref[:, axis].max()
    edges = np.linspace(min_val, max_val, 101)

    hist, _ = np.histogram(tics[:, axis], bins=edges, density=True)
    free_energy = -np.log(hist / hist.max())
    centers = 0.5 * (edges[1:] + edges[:-1])

    ax.plot(centers, free_energy, linewidth=5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(min_val, max_val)
