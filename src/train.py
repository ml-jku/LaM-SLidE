import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import torch._dynamo
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# lt.monkey_patch()

OmegaConf.register_new_resolver("eval", eval)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# This is hacky but the problem is when using hydra with ddp and multirun
# that an argument error happens.
if os.environ.get("LOCAL_RANK", "0") != "0":
    filtered_args = []
    for arg in sys.argv:
        if not arg.startswith("hydra."):
            filtered_args.append(arg)
    sys.argv = filtered_args

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.utils import load_class, update_cfg_from_first_stage

log = RankedLogger(__name__, rank_zero_only=True)


torch._dynamo.config.cache_size_limit = 512


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if cfg.get("ckpt_path") and cfg.get("resume"):
        model: LightningModule = load_class(cfg.model._target_).load_from_checkpoint(
            cfg.get("ckpt_path"),
            map_location="cpu",
        )
    else:
        model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if cfg.get("log_grads") and len(logger) > 0:
        logger[0].watch(model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "matmul_precision": cfg.matmul_precision,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        if cfg.get("ckpt_path") and cfg.get("resume"):
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        else:
            trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Instantiating trainer for testing (higher precision), and set to single GPU.")
        if cfg.get("test_ckpt") == "last":
            ckpt_path = trainer.checkpoint_callback.last_model_path
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        if isinstance(cfg.trainer.devices, int):
            cfg.trainer.devices = 1
        elif isinstance(cfg.trainer.devices, list):
            cfg.trainer.devices = cfg.trainer.devices[0]
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger, precision="32-true"
        )
        log.info("Seeding for testing...")
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)
        log.info("Starting testing!")

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    if not cfg.first_stage:
        cfg = update_cfg_from_first_stage(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
