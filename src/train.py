from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from omegaconf import DictConfig, OmegaConf

from callback import ModelCheckpointWithLogging


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    run_dir = Path(HydraConfig.get().run.dir)

    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=f"{cfg.run_prefix}_{str(run_dir).replace("/", "_")}",
        tags=[cfg.run_prefix],
    )
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    model = hydra.utils.instantiate(cfg.litmodule)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    callbacks = []
    if "callbacks" in cfg.trainer:
        callbacks = hydra.utils.instantiate(cfg.trainer.callbacks)
    callbacks.append(
        ModelCheckpointWithLogging(
            run_dir / "ckpt", save_last=True, save_on_train_epoch_end=True
        )
    )
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    train()
