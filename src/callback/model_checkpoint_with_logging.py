import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelCheckpointWithLogging(ModelCheckpoint):
    def _save_last_checkpoint(
        self, trainer: pl.Trainer, monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        super()._save_last_checkpoint(trainer, monitor_candidates)
        self._log_ckpt_path(trainer)

    def _log_ckpt_path(self, trainer: pl.Trainer) -> None:
        trainer.logger.experiment.summary["ckpt"] = self.last_model_path
