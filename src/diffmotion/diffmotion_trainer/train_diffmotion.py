from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
# import pytorch_lightning.profilers
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb
from rich import get_console
# from lightning.pytorch.loggers.wandb import WandbLogger

# from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
import lightning.pytorch.profilers as Profiler
# import pytorch_lightning.profilers as Profiler
import os

# from rich.console import Console

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    log.info(f'CPU Count = {os.cpu_count()}')
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")

    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    if cfg.get("logger") is not None:
        if 'wandb' in cfg.get("logger"):
            # wandb.init(settings=wandb.Settings(_service_wait=300))
             wandb.login(key='4ed0ca0cb4b019d34e2d593ea78ba7c74946aa53')    # if you want to use wandb cloud, you need to login and copy the API key to this parameter.


    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    profiler: Profiler = hydra.utils.instantiate(cfg.get("profiler"))

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger,  profiler=profiler)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)
    # model.summary()
    console = get_console()
    console.width = 768
    use_auto_scale_batch_size = cfg.use_auto_scale_batch_size
    if use_auto_scale_batch_size:
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(trainer)
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(model, datamodule, mode="binsearch")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = cfg.get("ckpt_path")
            # ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path,weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    # print(trainer.profiler.summary())
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../../configs_diffmotion", config_name="train_diffmotion_v2.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # os.environ['NUMEXPR_MAX_THREADS'] = '16'
    # os.environ['NUMEXPR_NUM_THREADS'] = '16'
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    # return optimized metric
    return metric_value


if __name__ == "__main__":
    # console = Console(width=1024)
    # print(console)
    main()
