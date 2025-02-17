# Code based on https://github.com/bjing2016/mdgen/tree/master adopted for hydra.
import os
import pickle
import time
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import jensenshannon
from statsmodels.tsa.stattools import acovf
from tqdm.auto import tqdm

import wandb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import src.modules.analysis as analysis
from modules.sampling import SIAtom14SamplingWrapper
from src.utils import RankedLogger, task_wrapper
from src.utils.logging_utils import load_run_config_from_wb
from src.utils.traj_utils import load_traj
from src.utils.utils import load_ckpt_path, load_class

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Python 3.11 is not supported by pyemma, so we need to monkey patch it.
# isort: skip_file
import src.utils.monkey_patch  # noqa: F401
import pyemma

log = RankedLogger(__name__, rank_zero_only=True)


def create_output_dirs(cfg: DictConfig):
    plot_dir = os.path.join(cfg.paths.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    traj_dir = os.path.join(cfg.paths.output_dir, "trajectory")
    os.makedirs(traj_dir, exist_ok=True)
    return traj_dir, plot_dir


def setup_model(cfg: DictConfig):
    cfg_model = load_run_config_from_wb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        run_id=cfg.wandb_run_id,
    )
    ckpt_path = load_ckpt_path(
        ckpt_dir=cfg_model.callbacks.model_checkpoint.dirpath, last=cfg.ckpt_last
    )
    log.info(f"Loading checkpoint from {ckpt_path}")
    model: L.LightningModule = load_class(
        class_string=cfg_model.model._target_
    ).load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location="cpu",
        sampling_method=cfg.sampling_method,
        sampling_kwargs=cfg.sampling_kwargs,
    )
    model.load_ema_weights()
    model.freeze()
    model.eval()
    model.to(cfg.device)
    if cfg.full_precision:
        model.float()
    return model


def analyze_trajectory(name, cfg):
    """
    https://github.com/bjing2016/mdgen/tree/master adopted for hydra.
    """
    out = {}
    np.random.seed(137)
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    # BACKBONE torsion marginals PLOT ONLY
    if cfg.plot:
        feats, traj = analysis.get_featurized_traj(
            f"{cfg.trajs_output_dir}/{name}", sidechains=False, cossin=False
        )
        if cfg.truncate:
            traj = traj[: cfg.truncate]
        feats, ref = analysis.get_featurized_traj(
            f"{cfg.mddir}/{name}/{name}", sidechains=False, cossin=False
        )
        pyemma.plots.plot_feature_histograms(
            ref, feature_labels=feats, ax=axs[0, 0], color=colors[0]
        )
        pyemma.plots.plot_feature_histograms(traj, ax=axs[0, 0], color=colors[1])
        axs[0, 0].set_title("BB torsions")

    # JENSEN SHANNON DISTANCES ON ALL TORSIONS
    feats, traj = analysis.get_featurized_traj(
        f"{cfg.pdbdir}/{name}", sidechains=True, cossin=False
    )
    if cfg.truncate:
        traj = traj[: cfg.truncate]
    feats, ref = analysis.get_featurized_traj(
        f"{cfg.mddir}/{name}/{name}", sidechains=True, cossin=False
    )

    out["features"] = feats.describe()

    out["JSD"] = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:, i], range=(-np.pi, np.pi), bins=100)[0]
        out["JSD"][feat] = jensenshannon(ref_p, traj_p)

    for i in [1, 3]:
        ref_p = np.histogram2d(
            *ref[:, i : i + 2].T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50
        )[0]
        traj_p = np.histogram2d(
            *traj[:, i : i + 2].T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50
        )[0]
        out["JSD"]["|".join(feats.describe()[i : i + 2])] = jensenshannon(
            ref_p.flatten(), traj_p.flatten()
        )

    # Torsion decorrelations
    if cfg.no_decorr:
        pass
    else:
        out["md_decorrelation"] = {}
        for i, feat in enumerate(feats.describe()):

            autocorr = acovf(np.sin(ref[:, i]), demean=False, adjusted=True, nlag=100000) + acovf(
                np.cos(ref[:, i]), demean=False, adjusted=True, nlag=100000
            )
            baseline = np.sin(ref[:, i]).mean() ** 2 + np.cos(ref[:, i]).mean() ** 2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if "PHI" in feat or "PSI" in feat:
                axs[0, 1].plot(
                    lags, (autocorr - baseline) / (1 - baseline), color=colors[i % len(colors)]
                )
            else:
                axs[0, 2].plot(
                    lags, (autocorr - baseline) / (1 - baseline), color=colors[i % len(colors)]
                )

            out["md_decorrelation"][feat] = (autocorr.astype(np.float16) - baseline) / (
                1 - baseline
            )

        axs[0, 1].set_title("Backbone decorrelation")
        axs[0, 2].set_title("Sidechain decorrelation")
        axs[0, 1].set_xscale("log")
        axs[0, 2].set_xscale("log")

        out["our_decorrelation"] = {}
        for i, feat in enumerate(feats.describe()):

            autocorr = acovf(
                np.sin(traj[:, i]), demean=False, adjusted=True, nlag=1 if cfg.ito else 1000
            ) + acovf(np.cos(traj[:, i]), demean=False, adjusted=True, nlag=1 if cfg.ito else 1000)
            baseline = np.sin(traj[:, i]).mean() ** 2 + np.cos(traj[:, i]).mean() ** 2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if "PHI" in feat or "PSI" in feat:
                axs[1, 1].plot(
                    lags, (autocorr - baseline) / (1 - baseline), color=colors[i % len(colors)]
                )
            else:
                axs[1, 2].plot(
                    lags, (autocorr - baseline) / (1 - baseline), color=colors[i % len(colors)]
                )

            out["our_decorrelation"][feat] = (autocorr.astype(np.float16) - baseline) / (
                1 - baseline
            )

        axs[1, 1].set_title("Backbone decorrelation")
        axs[1, 2].set_title("Sidechain decorrelation")
        axs[1, 1].set_xscale("log")
        axs[1, 2].set_xscale("log")

    # TICA
    feats, traj = analysis.get_featurized_traj(
        f"{cfg.pdbdir}/{name}", sidechains=True, cossin=True
    )
    if cfg.truncate:
        traj = traj[: cfg.truncate]
    feats, ref = analysis.get_featurized_traj(
        f"{cfg.mddir}/{name}/{name}", sidechains=True, cossin=True
    )

    tica, _ = analysis.get_tica(ref)
    ref_tica = tica.transform(ref)
    traj_tica = tica.transform(traj)

    tica_0_min = min(ref_tica[:, 0].min(), traj_tica[:, 0].min())
    tica_0_max = max(ref_tica[:, 0].max(), traj_tica[:, 0].max())

    tica_1_min = min(ref_tica[:, 1].min(), traj_tica[:, 1].min())
    tica_1_max = max(ref_tica[:, 1].max(), traj_tica[:, 1].max())

    ref_p = np.histogram(ref_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    out["JSD"]["TICA-0"] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(
        *ref_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50
    )[0]
    traj_p = np.histogram2d(
        *traj_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50
    )[0]
    out["JSD"]["TICA-0,1"] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    # 1,0, 1,1 TICA FES
    if cfg.plot:
        pyemma.plots.plot_free_energy(*ref_tica[::100, :2].T, ax=axs[2, 0], cbar=False)
        pyemma.plots.plot_free_energy(*traj_tica[:, :2].T, ax=axs[2, 1], cbar=False)
        axs[2, 0].set_title("TICA FES (MD)")
        axs[2, 1].set_title("TICA FES (ours)")

    # TICA decorrelation
    if cfg.no_decorr:
        pass
    else:
        # x, adjusted=False, demean=True, fft=True, missing='none', nlag=None
        autocorr = acovf(ref_tica[:, 0], nlag=100000, adjusted=True, demean=False)
        out["md_decorrelation"]["tica"] = autocorr.astype(np.float16)
        if cfg.plot:
            axs[0, 3].plot(autocorr)
            axs[0, 3].set_title("MD TICA")

        autocorr = acovf(traj_tica[:, 0], nlag=1 if cfg.ito else 1000, adjusted=True, demean=False)
        out["our_decorrelation"]["tica"] = autocorr.astype(np.float16)
        if cfg.plot:
            axs[1, 3].plot(autocorr)
            axs[1, 3].set_title("Traj TICA")

    # Markov state model stuff
    if not cfg.no_msm:
        kmeans, ref_kmeans = analysis.get_kmeans(tica.transform(ref))
        try:
            msm, pcca, cmsm = analysis.get_msm(ref_kmeans, nstates=10)

            out["kmeans"] = kmeans
            out["msm"] = msm
            out["pcca"] = pcca
            out["cmsm"] = cmsm

            traj_discrete = analysis.discretize(tica.transform(traj), kmeans, msm)
            ref_discrete = analysis.discretize(tica.transform(ref), kmeans, msm)
            out["traj_metastable_probs"] = (traj_discrete == np.arange(10)[:, None]).mean(1)
            out["ref_metastable_probs"] = (ref_discrete == np.arange(10)[:, None]).mean(1)

            msm_transition_matrix = np.eye(10)
            for a, i in enumerate(cmsm.active_set):
                for b, j in enumerate(cmsm.active_set):
                    msm_transition_matrix[i, j] = cmsm.transition_matrix[a, b]

            out["msm_transition_matrix"] = msm_transition_matrix
            out["pcca_pi"] = pcca._pi_coarse

            msm_pi = np.zeros(10)
            msm_pi[cmsm.active_set] = cmsm.pi
            out["msm_pi"] = msm_pi

            if cfg.no_traj_msm:
                pass
            else:

                traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=cfg.msm_lag)
                out["traj_msm"] = traj_msm

                traj_transition_matrix = np.eye(10)
                for a, i in enumerate(traj_msm.active_set):
                    for b, j in enumerate(traj_msm.active_set):
                        traj_transition_matrix[i, j] = traj_msm.transition_matrix[a, b]
                out["traj_transition_matrix"] = traj_transition_matrix

                traj_pi = np.zeros(10)
                traj_pi[traj_msm.active_set] = traj_msm.pi
                out["traj_pi"] = traj_pi

        except Exception as e:
            print("ERROR", e, name, flush=True)

    if cfg.plot:
        fig.savefig(f"{cfg.pdbdir}/{name}.pdf")

    return name, out


def analyze_trajectories(cfg: DictConfig):
    if cfg.pdb_ids:
        pdb_ids = cfg.pdb_ids
    else:
        pdb_ids = [
            nam.split(".")[0]
            for nam in os.listdir(cfg.pdbdir)
            if ".pdb" in nam and "_traj" not in nam
        ]
        pdb_ids = [nam for nam in pdb_ids if os.path.exists(f"{cfg.pdbdir}/{nam}.xtc")]

    log.info(f"Number of trajectories: {len(pdb_ids)}")
    analyze_with_cfg = partial(analyze_trajectory, cfg=cfg)
    if cfg.num_workers > 1:
        p = Pool(cfg.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    out = dict(tqdm(__map__(analyze_with_cfg, pdb_ids), total=len(pdb_ids)))
    if cfg.num_workers > 1:
        p.__exit__(None, None, None)

    if cfg.save:
        with open(f"{cfg.paths.output_dir}/{cfg.save_name}", "wb") as f:
            f.write(pickle.dumps(out))

    return out


def sample_trajectory(
    name: str,
    wrapper: SIAtom14SamplingWrapper,
    cfg: DictConfig,
):
    traj_file = f"{cfg.data_dir}/{name}-traj-arrays.npz"
    top_file = f"{cfg.data_dir}/{name}-traj-state0.pdb"
    ref_traj = load_traj(trajfile=traj_file, top=top_file)

    log.info(f"Sampling trajectory {name}")
    start = time.time()
    traj_pred = wrapper.sample_traj(
        traj=ref_traj,
        num_rollouts=cfg.num_rollouts,
        cond_frame_idx=cfg.cond_frame_idx,
    )
    log.info(f"Time taken for {name}: {time.time() - start}")

    traj_pred = traj_pred.superpose(traj_pred)
    traj_pred.save(f"{cfg.trajs_output_dir}/{name}.xtc")
    traj_pred[0].save_pdb(f"{cfg.trajs_output_dir}/{name}.pdb")


def sample_trajectories(cfg: DictConfig):
    log.info(f"Loading split file {cfg.split}")
    df = pd.read_csv(cfg.split, index_col="name")

    log.info("Instantiating model")
    model = setup_model(cfg)
    wrapper = SIAtom14SamplingWrapper(model=model)

    for name in tqdm(df.index, desc="Sampling trajectories"):
        if cfg.pdb_ids and name not in cfg.pdb_ids:
            continue
        try:
            log.info(f"Sampling trajectory: {name}")
            sample_trajectory(name=name, wrapper=wrapper, cfg=cfg)
        except Exception as e:
            log.error(f"Error sampling trajectory {name}: {e}")


def calc_summary_metrics(out: dict, cfg: DictConfig):
    all_bb_anlges = []
    all_sc_anlges = []
    all_angles = []
    all_tica0 = []
    all_tica01 = []
    all_msms_jsd = []

    for peptide, metrics in out.items():
        jsd = metrics["JSD"]
        all_bb_anlges.extend(
            [v for k, v in jsd.items() if (("PHI" in k) or ("PSI" in k)) and ("|" not in k)]
        )  # exclude coupled phi/psi
        all_sc_anlges.extend([v for k, v in jsd.items() if ("CHI" in k)])
        all_angles.extend(
            [
                v
                for k, v in jsd.items()
                if (("PHI" in k) or ("PSI" in k) or ("CHI" in k)) and ("|" not in k)
            ]
        )
        all_tica0.append(jsd["TICA-0"])
        all_tica01.append(jsd["TICA-0,1"])
        if "ref_metastable_probs" in metrics and "traj_metastable_probs" in metrics:
            all_msms_jsd.append(
                jensenshannon(metrics["ref_metastable_probs"], metrics["traj_metastable_probs"])
            )

    summary_metrics = {
        "BB": np.mean(all_bb_anlges),
        "SC": np.mean(all_sc_anlges),
        "ALL": np.mean(all_angles),
        "TICA-0": np.mean(all_tica0),
        "TICA-0,1": np.mean(all_tica01),
    }
    if len(all_msms_jsd) > 0:
        summary_metrics["MSMS"] = np.mean(all_msms_jsd)

    return summary_metrics


@task_wrapper
def sample(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    L.seed_everything(cfg.seed)

    # Initialize W&B run
    run = wandb.init(
        project=f"{cfg.wandb_project}_eval",
        name=cfg.wandb_run_id,
        config=dict(cfg),
        tags=["evaluation"],
    )

    log.info(f"Evaluating run {cfg.wandb_run_id}.")

    log.info(f"Creating output directories at {cfg.trajs_output_dir}.")
    os.makedirs(cfg.trajs_output_dir, exist_ok=True)

    log.info("Sampling trajectories")
    sample_trajectories(cfg)

    log.info("Analyzing trajectories")
    out = analyze_trajectories(cfg)
    summary_metrics = calc_summary_metrics(out, cfg)

    wandb.log(summary_metrics)
    run.finish()

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_peptide.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for sampling.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    cfg = edict(OmegaConf.to_container(cfg, resolve=True))
    sample(cfg)


if __name__ == "__main__":
    main()
