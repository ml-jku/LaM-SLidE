from typing import List

import lightning as L
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb
from src.modules.sampling import SIAtom14SamplingWrapper
from src.utils import RankedLogger
from src.utils.backbone_utils import compute_phi_psi
from src.utils.plots import plot_ramachandran
from src.utils.tica_utils import plot_free_energy, plot_tic01, run_tica, tica_features
from src.utils.traj_utils import load_traj, traj_analysis

log = RankedLogger(__name__, rank_zero_only=True)


def log_metrics(traj_ref, traj_sampled, protein, tica_lagtime, pl_module, trainer):
    # Plot TICs
    feats_ref = tica_features(traj_ref)
    feats_model = tica_features(traj_sampled)
    tica_model = run_tica(traj_ref, lagtime=tica_lagtime)
    tics_ref = tica_model.transform(feats_ref)
    tics_model = tica_model.transform(feats_model)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_tic01(ax1, tics_ref, "Ref", tics_ref)
    plot_tic01(ax2, tics_model, "Model", tics_ref)
    plt.tight_layout()
    fig.suptitle(f"{protein} - TICA", y=1.05, fontsize=14)
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/tics": wandb.Image(
                fig, caption=f"TICA of {protein} - Epoch {trainer.current_epoch}"
            )
        },
    )
    plt.close(fig)
    # Plot ramachandran
    torsions_ref = compute_phi_psi(traj_ref)
    torsions_model = compute_phi_psi(traj_sampled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_ramachandran(ax1, torsions_ref, "Ref", show_initial=True)
    plot_ramachandran(ax2, torsions_model, "Model", show_initial=True)
    plt.tight_layout()
    fig.suptitle(f"{protein} - Ramachandran", y=1.05, fontsize=14)
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/ramachandran": wandb.Image(
                fig,
                caption=f"Ramachandran plot of {protein} - Epoch {trainer.current_epoch}",
            )
        },
    )
    plt.close(fig)
    # Plot free energy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    axis = 1
    plot_free_energy(
        ax1,
        tics=tics_ref,
        title="TIC 0",
        tics_ref=tics_ref,
        xlabel="TIC 0",
        label="Ref",
        axis=axis,
    )
    plot_free_energy(
        ax1,
        tics=tics_model,
        title="TIC 0",
        tics_ref=tics_ref,
        xlabel="TIC 0",
        label="Model",
        axis=axis,
    )
    ax1.legend()

    axis = 2
    plot_free_energy(
        ax2,
        tics=tics_ref,
        title="TIC 1",
        tics_ref=tics_ref,
        xlabel="TIC 1",
        label="Ref",
        axis=axis,
    )
    plot_free_energy(
        ax2,
        tics=tics_model,
        title="TIC 1",
        tics_ref=tics_ref,
        xlabel="TIC 1",
        label="Model",
        axis=axis,
    )
    ax2.legend()

    plt.tight_layout()
    fig.suptitle(f"{protein} - Free energy", y=1.05, fontsize=14)
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/free_energy": wandb.Image(
                fig,
                caption=f"Free energy of {protein} - Epoch {trainer.current_epoch}",
            )
        },
    )
    plt.close(fig)

    # Log metrics
    ramachandran_js, pwd_js, rg_js, tic_js, tic2d_js, val_ca, rmse_contact = traj_analysis(
        traj_sampled, traj_ref[: len(traj_sampled)], lagtime=tica_lagtime
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/ramachandran_js": ramachandran_js,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/pwd_js": pwd_js,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/rg_js": rg_js,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/tic_js": tic_js,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/tic2d_js": tic2d_js,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/val_ca": val_ca,
            "epoch": trainer.current_epoch,
        },
    )
    pl_module.logger.experiment.log(
        {
            f"val/{protein}/rmse_contact": rmse_contact,
            "epoch": trainer.current_epoch,
        },
    )
    return {
        "ramachandran_js": ramachandran_js,
        "pwd_js": pwd_js,
        "rg_js": rg_js,
        "tic_js": tic_js,
        "tic2d_js": tic2d_js,
        "val_ca": val_ca,
        "rmse_contact": rmse_contact,
    }


class SIAtom14SampleCallback(L.Callback):

    def __init__(
        self,
        data_dir: str,
        proteins: List[str],
        interval: int = 5000,
        num_rollouts: int = 1,
        tica_lagtime: int = 1000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.proteins = proteins
        self.interval = interval
        self.num_rollouts = num_rollouts
        self.tica_lagtime = tica_lagtime

    def sample_and_log(
        self,
        pl_module: L.LightningModule,
        trainer: L.Trainer,
        protein: str,
        sampler: SIAtom14SamplingWrapper,
    ):
        top_file = f"{self.data_dir}/{protein}-traj-state0.pdb"
        traj_file = f"{self.data_dir}/{protein}-traj-arrays.npz"
        traj_ref = load_traj(traj_file, top_file)
        traj_ref.superpose(traj_ref)
        traj_ref.center_coordinates()

        traj_sampled = sampler.sample_traj(
            traj_ref,
            num_rollouts=self.num_rollouts,
        )
        traj_sampled.superpose(traj_sampled)

        metrics = log_metrics(
            traj_ref=traj_ref,
            traj_sampled=traj_sampled,
            protein=protein,
            tica_lagtime=self.tica_lagtime,
            pl_module=pl_module,
            trainer=trainer,
        )

        return metrics

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if (trainer.current_epoch + 1) % self.interval != 0:
            return

        if trainer.is_global_zero and pl_module.logger is not None:
            sampler = SIAtom14SamplingWrapper(pl_module)
            log.info(f"Sampling {len(self.proteins)} trajectories")
            all_metrics = []
            for protein in tqdm(self.proteins):
                try:
                    metrics = self.sample_and_log(
                        pl_module=pl_module,
                        trainer=trainer,
                        protein=protein,
                        sampler=sampler,
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    log.error(f"Error sampling {protein}: {e}")

            if all_metrics:
                mean_metrics = {}
                for key in all_metrics[0].keys():
                    values = [m[key] for m in all_metrics]
                    mean_metrics[key] = sum(values) / len(values)

                # Log mean metrics
                for key, value in mean_metrics.items():
                    pl_module.logger.experiment.log(
                        {
                            f"val/avg_{key}": value,
                            "epoch": trainer.current_epoch,
                        },
                    )
