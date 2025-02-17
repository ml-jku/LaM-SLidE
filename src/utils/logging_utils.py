from typing import Any, Dict, Union

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

import wandb
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    if "first_stage_settings" in cfg:
        entity = cfg.get("first_stage_settings").get("entity")
        project = cfg.get("first_stage_settings").get("project")
        run_id = cfg.get("first_stage_settings").get("run_id")
        hparams["first_stage_run"] = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        hparams["first_stage_settings"] = cfg.get("first_stage_settings")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def load_run_config_from_wb(
    entity: str, project: str, run_id: str
) -> Union[DictConfig, ListConfig]:
    """Retrieve the run configuration from a wandb run.

    :param entity: The entity to retrieve the run from.
    :type entity: str
    :param project: The project name to retrieve the run from.
    :type project: str
    :param run_id: The run id to retrieve the run from.
    :type run_id: str
    :return: The run configuration.
    :rtype: Union[DictConfig, ListConfig]
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    config = run.config
    return OmegaConf.create(config)
