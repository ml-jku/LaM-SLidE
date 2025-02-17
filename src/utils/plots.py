import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import gaussian_kde


def plot_tic2d(
    tic0_ref,
    tic1_ref,
    tic0_model,
    tic1_model,
    save_path,
    xlabel="TIC 0",
    ylabel="TIC 1",
    name="FBM",
):
    kde = gaussian_kde(np.vstack([tic0_ref, tic1_ref]))
    tic0_min, tic0_max = tic0_ref.min(), tic0_ref.max()
    tic1_min, tic1_max = tic1_ref.min(), tic1_ref.max()
    X, Y = np.meshgrid(np.linspace(tic0_min, tic0_max, 200), np.linspace(tic1_min, tic1_max, 200))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    thresh = 0.013
    Z[Z < thresh] = np.nan
    sigma = 1.0  # Gaussian filter sigma
    Z_smooth = gaussian_filter(Z, sigma=sigma)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.contour(X, Y, Z_smooth, levels=15, cmap="viridis", linewidths=2.0, alpha=0.8)

    local_max = maximum_filter(Z_smooth, size=20) == Z_smooth
    maxima = np.argwhere(local_max & ~np.isnan(Z_smooth))
    # print(f"local maxima: {maxima}")

    collections = ax.collections
    paths = [path for collection in collections for path in collection.get_paths()]
    vertices = np.concatenate([path.vertices for path in paths])
    contour_x, contour_y = vertices[:, 0], vertices[:, 1]

    tic0_min, tic0_max = contour_x.min(), contour_x.max()
    tic1_min, tic1_max = contour_y.min(), contour_y.max()

    i = 1
    for y, x in maxima:
        center_x = X[0, x]
        center_y = Y[y, 0]
        if tic0_min <= center_x <= tic0_max and tic1_min <= center_y <= tic1_max:
            #######################
            # dist_ref = np.sqrt((tic0_ref - center_x) ** 2 + (tic1_ref - center_y) ** 2)
            # dist_model = np.sqrt((tic0_model - center_x) ** 2 + (tic1_model - center_y) ** 2)
            # print(f"cluster {i}, MD index: {np.argmin(dist_ref)}, model index: {np.argmin(dist_model)}")
            #######################

            plt.text(
                center_x,
                center_y,
                str(i),
                ha="center",
                va="center",
                fontsize=40,
                color="black",
                fontweight="bold",
            )
            i += 1

    model_points = (
        (tic0_model >= tic0_min)
        & (tic0_model <= tic0_max)
        & (tic1_model >= tic1_min)
        & (tic1_model <= tic1_max)
    )

    plt.scatter(tic0_ref[0], tic1_ref[0], color="red", s=50)
    plt.scatter(
        tic0_model[model_points], tic1_model[model_points], color="orange", alpha=0.6, s=15
    )

    plt.text(
        0.05,
        0.05,
        name,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=40,
    )

    plt.xlim(tic0_min, tic0_max)
    plt.ylim(tic1_min, tic1_max)

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.xlabel(xlabel, fontsize=50)
    plt.ylabel(ylabel, fontsize=50)

    plt.savefig(save_path, format="pdf")


def plot_free_energy(torsion_ref, torsion_model, save_path, xlabel="TIC 0", name="FBM"):
    plt.figure(figsize=(10, 8))

    feat_bins = np.linspace(torsion_ref.min(), torsion_ref.max(), 100)
    hist_ref, edges_ref = np.histogram(torsion_ref, bins=feat_bins, density=True)
    free_energy_ref = -np.log(hist_ref / hist_ref.max())
    centers_ref = 0.5 * (edges_ref[1:] + edges_ref[:-1])
    plt.plot(centers_ref, free_energy_ref, linewidth=4, label="MD", linestyle="-")

    hist_model, edges_model = np.histogram(torsion_model, bins=feat_bins, density=True)
    free_energy_model = -np.log(hist_model / hist_model.max())
    centers_model = 0.5 * (edges_model[1:] + edges_model[:-1])
    plt.plot(centers_model, free_energy_model, linewidth=4, label=name, linestyle="--")

    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel(xlabel, fontsize=50)
    plt.ylabel("Free energy/$k_B$T", fontsize=50)

    plt.text(
        0.96,
        0.96,
        name,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=40,
    )

    plt.savefig(save_path, format="pdf")


def plot_ramachandran(
    ax,
    torsions,
    title,
    show_initial=False,
    bins=100,
):
    """Helper function to plot a single Ramachandran plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    torsions : tuple of arrays
        Tuple containing (phi, psi) angles in radians
    title : str
        Plot title
    show_initial : bool, optional
        Whether to show the initial structure point
    bins : int, optional
        Number of bins for the histogram
    """
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)

    ax.hist2d(
        torsions[0].flatten(),
        torsions[1].flatten(),
        bins=[bin_edges, bin_edges],
        norm=mpl.colors.LogNorm(),
        density=True,
    )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_title(title)

    if show_initial:
        ax.scatter(torsions[0][0], torsions[1][0], marker="x", color="red", s=50)
