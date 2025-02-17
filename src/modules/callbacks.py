import lightning as L
import pandas as pd

from utils.plotting import plot_3d_comparison


class PointCloudVisualizationCallback(L.Callback):
    def __init__(self, ax_range=[-1, 1]):
        super().__init__()
        self.ax_range = ax_range

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero and pl_module.logger is not None:
            mask = pl_module.save_random_vis_batch["mask"][0].cpu().numpy()
            sample_preds = (
                pl_module.save_random_vis_batch["preds_pos"][0][mask].float().cpu().numpy()
            )
            sample_targets = (
                pl_module.save_random_vis_batch["targets"][0][mask].float().cpu().numpy()
            )
            sample_atom_types = (
                pl_module.save_random_vis_batch["atom_types"][0][mask].float().cpu().numpy()
            )

            df_preds = pd.DataFrame(
                {
                    "x": sample_preds[:, 0],
                    "y": sample_preds[:, 1],
                    "z": sample_preds[:, 2],
                    "atom_type": sample_atom_types,
                }
            )
            df_targets = pd.DataFrame(
                {
                    "x": sample_targets[:, 0],
                    "y": sample_targets[:, 1],
                    "z": sample_targets[:, 2],
                    "atom_type": sample_atom_types,
                }
            )
            fig = plot_3d_comparison(
                df_ground_truth=df_targets, df_predictions=df_preds, ax_range=self.ax_range
            )
            pl_module.logger.experiment.log({"val/vis/sample": fig})
