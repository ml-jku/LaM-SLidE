import argparse

import lightning as pl
import rootutils
import torch
from hydra.utils import instantiate

torch.set_float32_matmul_precision("high")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.logging_utils import load_run_config_from_wb
from src.utils.utils import load_ckpt_path, load_class

parser = argparse.ArgumentParser()

parser.add_argument("--wandb_project", type=str, default="nba-second-stage")
parser.add_argument("--wandb_entity", type=str, default="technobase.fm")
parser.add_argument("--wandb_run_id", type=str, default="ja24gmlx")
parser.add_argument("--device", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1500)

args = parser.parse_args()


def main():
    cfg = load_run_config_from_wb(
        entity=args.wandb_entity, project=args.wandb_project, run_id=args.wandb_run_id
    )
    cfg.data.batch_size = args.batch_size

    data_module = instantiate(cfg.data)

    checkpoint = load_ckpt_path(cfg.callbacks.model_checkpoint.dirpath, last=False)
    model = load_class(cfg.model._target_)
    model = model.load_from_checkpoint(
        checkpoint, map_location="cpu", K=20, num_runs=20, post_process=False
    )
    model.freeze()
    model.eval()

    data_module = instantiate(cfg.data)
    trainer = pl.Trainer(devices=[args.device])
    pl.seed_everything(43)
    trainer.test(model, datamodule=data_module, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
