import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.subplots import make_subplots

from src.utils.constants import ATOM_COLOR_MAP, ATOM_ENCODING


class ColorMap:
    @classmethod
    def atom_enc_to_atom_type(cls, atom_enc):
        swap_encoding = {v: k for k, v in ATOM_ENCODING.items()}
        return swap_encoding[atom_enc]

    @classmethod
    def atom_enc_to_color(cls, atom_enc):
        return ATOM_COLOR_MAP[ColorMap.atom_enc_to_atom_type(atom_enc)]


def plot_3d_comparison(
    df_predictions, df_ground_truth=None, width=800, height=400, ax_range=[-1, 1]
):
    df_predictions["atom_type"] = df_predictions["atom_type"].apply(ColorMap.atom_enc_to_color)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df_predictions["x"],
            y=df_predictions["y"],
            z=df_predictions["z"],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=5,
                color=df_predictions["atom_type"],
            ),
            name="Predictions",
        )
    )

    if df_ground_truth is not None:
        df_ground_truth["atom_type"] = df_ground_truth["atom_type"].apply(
            ColorMap.atom_enc_to_color
        )
        fig.add_trace(
            go.Scatter3d(
                x=df_ground_truth["x"],
                y=df_ground_truth["y"],
                z=df_ground_truth["z"],
                mode="markers",
                marker=dict(
                    symbol="diamond-open",
                    size=5,
                    color=df_ground_truth["atom_type"],
                ),
                name="Ground Truth",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=ax_range, autorange=False),
            yaxis=dict(range=ax_range, autorange=False),
            zaxis=dict(range=ax_range, autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=width,
        height=height,
    )

    return fig


def plot_occ_pointcloud(
    df_occs: DataFrame,
    color_column: str,
    title: str,
    df_atoms: DataFrame = None,
    marker_size: float = 5,
):
    fig = px.scatter_3d(
        df_occs,
        x="X",
        y="Y",
        z="Z",
        color=color_column,
        title=title,
        color_discrete_map=ATOM_COLOR_MAP,
        category_orders={color_column: list(ATOM_COLOR_MAP.keys())},
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1], autorange=False),
            yaxis=dict(range=[0, 1], autorange=False),
            zaxis=dict(range=[0, 1], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(title="Occ/Atom", x=0, y=1, traceorder="normal"),
    )
    fig.update_traces(marker=dict(size=marker_size))

    if df_atoms is not None:
        # Add additional points using go.Scatter3d
        for atom_type in df_atoms["atom_type"].unique():
            atom_df = df_atoms[df_atoms["atom_type"] == atom_type]
            atom_trace = go.Scatter3d(
                x=atom_df["X"],
                y=atom_df["Y"],
                z=atom_df["Z"],
                mode="markers",
                marker=dict(
                    size=10,
                    symbol="diamond-open",
                    color=ATOM_COLOR_MAP[atom_type],
                ),
                text=atom_df["atom_type"],
                name=f"{atom_type} Atom",
            )
            fig.add_trace(atom_trace)

    return fig


def plot_density_point_cloud(
    df_densities: DataFrame,
    color_column: str,
    title: str,
    df_atoms: DataFrame = None,
    marker_size: float = 5,
):
    fig = px.scatter_3d(
        df_densities,
        x="X",
        y="Y",
        z="Z",
        color=color_column,
        color_continuous_scale="Viridis",
        title=title,
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1], autorange=False),
            yaxis=dict(range=[0, 1], autorange=False),
            zaxis=dict(range=[0, 1], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(title="Dens", x=0, y=1, traceorder="normal"),
    )
    fig.update_traces(marker=dict(size=marker_size))

    if df_atoms is not None:
        # Add additional points using go.Scatter3d
        for atom_type in df_atoms["atom_type"].unique():
            atom_df = df_atoms[df_atoms["atom_type"] == atom_type]
            atom_trace = go.Scatter3d(
                x=atom_df["X"],
                y=atom_df["Y"],
                z=atom_df["Z"],
                mode="markers",
                marker=dict(
                    size=10,
                    symbol="diamond-open",
                    color=ATOM_COLOR_MAP[atom_type],
                ),
                text=atom_df["atom_type"],
                name=f"{atom_type} Atom",
            )
            fig.add_trace(atom_trace)

    return fig


def plot_density_channels(df_densities, atom_encoding, dens_threshold=0.01, ax_range=[0, 1]):
    def generate_specs(atom_encoding):
        specs = []
        for i in range(0, len(atom_encoding), 2):
            if i + 1 < len(atom_encoding):
                # Add a full row with two scene subplots
                specs.append([{"type": "scene"}, {"type": "scene"}])
            else:
                # If it's the last item and odd, add a row with one scene and None
                specs.append([{"type": "scene"}, None])
        return specs

    specs = generate_specs(atom_encoding)

    fig = make_subplots(
        rows=len(atom_encoding) // 2 + len(atom_encoding) % 2,
        cols=2,
        specs=specs,
        subplot_titles=[f"Channel {k}" for k, _ in atom_encoding.items()],
    )

    for i, (a, _) in enumerate(atom_encoding.items()):
        df_atom_density = df_densities.loc[lambda x: x[a] >= dens_threshold, ["x", "y", "z", a]]

        fig.add_trace(
            go.Scatter3d(
                x=df_atom_density["x"],
                y=df_atom_density["y"],
                z=df_atom_density["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df_atom_density[a],
                    colorscale="Viridis",
                    opacity=0.8,
                    # colorbar=dict(title=f"Density {i}"),
                    showscale=False,
                ),
                text=[f"Density: {d:.8f}" for d in df_atom_density[a]],
            ),
            row=i // 2 + 1,
            col=1 if i % 2 == 0 else 2,
        )

    for i in range(1, len(atom_encoding) + 1):
        fig.update_layout(
            {
                f"scene{i}": dict(
                    xaxis=dict(range=ax_range, autorange=False),
                    yaxis=dict(range=ax_range, autorange=False),
                    zaxis=dict(range=ax_range, autorange=False),
                    aspectmode="cube",
                )
            }
        )

    fig.update_layout(
        height=1500,  # Increased height to accommodate 5 plots
        width=1000,
        title_text=f"3D Density Scatter Plots for 5 Channels (Density > {dens_threshold})",
        showlegend=False,
    )

    return fig


def plot_density_channels_comparison(
    df_pred, df_truth, atom_encoding, dens_threshold=0.01, ax_range=[0, 1]
):
    def generate_specs(atom_encoding):
        return [[{"type": "scene"}, {"type": "scene"}] for _ in atom_encoding]

    specs = generate_specs(atom_encoding)

    # Generate correct subplot titles
    subplot_titles = []
    for k in atom_encoding.keys():
        subplot_titles.extend([f"Channel {k} (Pred)", f"Channel {k} (Truth)"])

    fig = make_subplots(
        rows=len(atom_encoding),
        cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    for i, (a, _) in enumerate(atom_encoding.items()):
        # Plot predictions
        df_atom_density_pred = df_pred.loc[lambda x: x[a] >= dens_threshold, ["x", "y", "z", a]]
        fig.add_trace(
            go.Scatter3d(
                x=df_atom_density_pred["x"],
                y=df_atom_density_pred["y"],
                z=df_atom_density_pred["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df_atom_density_pred[a],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=False,
                ),
                text=[f"Density: {d:.8f}" for d in df_atom_density_pred[a]],
            ),
            row=i + 1,
            col=1,
        )

        # Plot ground truth
        df_atom_density_truth = df_truth.loc[lambda x: x[a] >= dens_threshold, ["x", "y", "z", a]]
        fig.add_trace(
            go.Scatter3d(
                x=df_atom_density_truth["x"],
                y=df_atom_density_truth["y"],
                z=df_atom_density_truth["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df_atom_density_truth[a],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=False,
                ),
                text=[f"Density: {d:.8f}" for d in df_atom_density_truth[a]],
            ),
            row=i + 1,
            col=2,
        )

    for i in range(len(atom_encoding)):
        fig.update_layout(
            {
                f"scene{2*i+1}": dict(
                    xaxis=dict(range=ax_range, autorange=False),
                    yaxis=dict(range=ax_range, autorange=False),
                    zaxis=dict(range=ax_range, autorange=False),
                    aspectmode="cube",
                ),
                f"scene{2*i+2}": dict(
                    xaxis=dict(range=ax_range, autorange=False),
                    yaxis=dict(range=ax_range, autorange=False),
                    zaxis=dict(range=ax_range, autorange=False),
                    aspectmode="cube",
                ),
            }
        )

    fig.update_layout(
        height=500 * len(atom_encoding),
        width=1200,
        title_text=f"3D Density Scatter Plots Comparison (Density > {dens_threshold})",
        showlegend=False,
    )

    return fig


def plot_ramachandran(
    torsions, name, title, output_dir, step_width, protein, show_initial_state=False, bins=100
):
    if torsions[0].shape[-1] == 1:
        plt.figure(figsize=(10, 10))
        # plt.title(f"{name.replace('_', ' ')}: Ramachandran plot - {title}")
        plt.title("MD")
        plt.hist2d(
            torsions[0].flatten(), torsions[1].flatten(), bins=bins, norm=mpl.colors.LogNorm()
        )
        if show_initial_state:
            plt.scatter(torsions[0][0], torsions[1][0], marker="x", color="red", s=50)
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
    elif torsions[0].shape[-1] == 3:
        fig, axs = plt.subplots(1, 3, figsize=(35, 10))
        for i in range(3):
            axs[i].hist2d(
                torsions[0][:, i], torsions[1][:, i], bins=bins, norm=mpl.colors.LogNorm()
            )
            axs[i].scatter(torsions[0][0, i], torsions[1][0, i], marker="x", color="red", s=50)

            axs[i].set_xlim(-np.pi, np.pi)
            axs[i].set_ylim(-np.pi, np.pi)
            axs[i].set_xlabel("Phi")
            axs[i].set_ylabel("Psi")

        # fig.suptitle(f"{name.replace('_', ' ')}: Ramachandran plot - {title}")
    else:
        raise NotImplementedError(
            "Ramachandran plot only implemented for one or three angle pairs."
        )
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_ramachandran_{title}_step_width_{step_width}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_dual_ramachandran(
    output_dir=".",
    show_initial_state=False,
    bins=100,
    title_font_size=24,
    ax_font_size=14,
    font_weight="bold",
    *,
    torsions1,
    torsions2,
    name,
    title1,
    title2,
    protein,
):
    fig, axs = plt.subplots(
        1, 2, figsize=(18, 9), gridspec_kw={"wspace": 0}
    )  # No horizontal space between plots

    # Plot for first set of torsions (torsions1)
    axs[0].hist2d(
        torsions1[0].flatten(),
        torsions1[1].flatten(),
        bins=bins,
        norm=mpl.colors.LogNorm(),
        density=True,
    )
    axs[0].set_xlim(-np.pi, np.pi)
    axs[0].set_ylim(-np.pi, np.pi)
    axs[0].set_xlabel("Phi", fontsize=ax_font_size)
    axs[0].set_ylabel("Psi", fontsize=ax_font_size)
    axs[0].set_title(
        f"{title1}", fontsize=title_font_size, fontweight=font_weight
    )  # Bold and larger title

    if show_initial_state:
        axs[0].scatter(torsions1[0][0], torsions1[1][0], marker="x", color="red", s=50)

    # Plot for second set of torsions (torsions2)
    axs[1].hist2d(
        torsions2[0].flatten(),
        torsions2[1].flatten(),
        bins=bins,
        norm=mpl.colors.LogNorm(),
        density=True,
    )
    axs[1].set_xlim(-np.pi, np.pi)
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_xlabel("Phi", fontsize=ax_font_size)
    axs[1].set_ylabel("Psi", fontsize=ax_font_size)
    axs[1].set_title(
        f"{title2}", fontsize=title_font_size, fontweight=font_weight
    )  # Bold and larger title

    if show_initial_state:
        axs[1].scatter(torsions2[0][0], torsions2[1][0], marker="x", color="red", s=50)

    # Adjust layout to remove spacing between subplots
    for ax in axs:
        ax.label_outer()  # Hide inner labels

    # Hide the ticks between the subplots
    axs[1].tick_params(left=False)
    axs[1].yaxis.set_ticklabels([])

    # Save the figure with both plots
    plt.savefig(
        os.path.join(
            output_dir,
            f"{protein}_{name}_dual_ramachandran_{title1}_{title2}.pdf",
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_density_channels_single_plot(
    df_densities,
    atom_encoding,
    atom_colors,
    dens_threshold=0.01,
    ax_range=[0, 1],
):
    import plotly.graph_objects as go

    layout = go.Layout(plot_bgcolor="rgba(242,242,242,1)")

    fig = go.Figure(layout=layout)

    for atom_type in atom_encoding.keys():
        df_atom_density = df_densities.loc[
            df_densities[atom_type] >= dens_threshold, ["x", "y", "z", atom_type]
        ]

        fig.add_trace(
            go.Scatter3d(
                x=df_atom_density["x"],
                y=df_atom_density["y"],
                z=df_atom_density["z"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=atom_colors[atom_type],
                    opacity=0.8,
                ),
                name=f"Atom Type: {atom_type}",
                text=[f"Density: {d:.8f}" for d in df_atom_density[atom_type]],
                showlegend=True,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=ax_range,
                autorange=False,
                backgroundcolor="rgba(242,242,242,1)",
                showgrid=False,
            ),
            yaxis=dict(
                range=ax_range,
                autorange=False,
                backgroundcolor="rgba(242,242,242,1)",
                showgrid=False,
            ),
            zaxis=dict(
                range=ax_range,
                autorange=False,
                backgroundcolor="rgba(242,242,242,1)",
                showgrid=False,
            ),
            aspectmode="cube",
        ),
        title_text=f"3D Density Scatter Plot for All Channels (Density > {dens_threshold})",
    )

    return fig


def plot_pedestrian_trajectory(
    pos, x_min=None, x_max=None, y_min=None, y_max=None, padding=0.1, title=None
):
    # Calculate the min and max values across all time steps using numpy
    calc_x_min = np.min(pos[:, :, 0])
    calc_x_max = np.max(pos[:, :, 0])
    calc_y_min = np.min(pos[:, :, 1])
    calc_y_max = np.max(pos[:, :, 1])

    # Use calculated values only if limits are not provided
    if x_min is None:
        x_min = calc_x_min
        x_padding = padding * (calc_x_max - calc_x_min)
        x_min -= x_padding

    if x_max is None:
        x_max = calc_x_max
        x_padding = padding * (calc_x_max - calc_x_min)
        x_max += x_padding

    if y_min is None:
        y_min = calc_y_min
        y_padding = padding * (calc_y_max - calc_y_min)
        y_min -= y_padding

    if y_max is None:
        y_max = calc_y_max
        y_padding = padding * (calc_y_max - calc_y_min)
        y_max += y_padding

    # Rest of the function remains the same...
    fig = go.Figure()

    particle_colors = list(range(pos.shape[1]))
    frames = []
    for t in range(pos.shape[0]):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pos[t, :, 0],
                        y=pos[t, :, 1],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=particle_colors,
                            colorscale="Viridis",
                            opacity=0.7,
                        ),
                        name=f"Time step {t}",
                    )
                ],
                name=str(t),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=pos[0, :, 0],
            y=pos[0, :, 1],
            mode="markers",
            marker=dict(size=8, color=particle_colors, colorscale="Viridis", opacity=0.7),
            name="Trajectory",
        )
    )

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                    ),
                ],
                x=0.1,
                y=0,
                xanchor="right",
            )
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time step: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {
                        "args": [
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k in range(len(frames))
                ],
            }
        ],
        xaxis=dict(
            range=[x_min, x_max],
            autorange=False,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            autorange=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        title=title,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True,
    )
    return fig
