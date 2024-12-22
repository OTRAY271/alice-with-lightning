import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim

import wandb
from model import ALI


class LitALI(L.LightningModule):
    def __init__(self, ali: ALI, lr: float = 1e-4):
        super().__init__()

        self.ali = ali
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        self.z_val = torch.randn(64, self.ali.latent_dim, 1, 1)

        self.automatic_optimization = False

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        gen_opt, dis_opt = self.optimizers()

        x, _ = batch

        batch_size = x.size(0)
        z = torch.randn(batch_size, self.ali.latent_dim, 1, 1, device=self.device)

        z_fake = self.ali.enc(x)
        x_fake = self.ali.dec(z)

        label_real = torch.ones(batch_size, 1, device=self.device)
        label_fake = torch.zeros(batch_size, 1, device=self.device)

        pred_real = self.ali.dis(x, z_fake.detach())
        pred_fake = self.ali.dis(x_fake.detach(), z)

        loss_dis_real = self.criterion(pred_real, label_real)
        loss_dis_fake = self.criterion(pred_fake, label_fake)
        loss_dis = loss_dis_real + loss_dis_fake

        self.ali.dis.zero_grad()
        self.manual_backward(loss_dis)
        dis_opt.step()

        pred_real = self.ali.dis(x, z_fake)
        pred_fake = self.ali.dis(x_fake, z)

        loss_gen_real = self.criterion(pred_fake, label_real)
        loss_gen_fake = self.criterion(pred_real, label_fake)
        loss_gen = loss_gen_real + loss_gen_fake

        self.ali.enc.zero_grad()
        self.ali.dec.zero_grad()
        self.manual_backward(loss_gen)
        gen_opt.step()

        self.log_dict(
            {
                "train/gen_loss": loss_gen,
                "train/dis_loss": loss_dis,
                "train/recon_loss": F.mse_loss(self.ali.dec(z_fake), x),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, _ = batch

        recon_x = self.ali.dec(self.ali.enc(x))

        self.log_dict(
            {
                "val/recon_loss": F.mse_loss(recon_x, x),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx == 0:
            samples = torchvision.utils.make_grid(
                self.ali.dec(self.z_val.to(self.device))
            )
            reconstructions = torchvision.utils.make_grid(
                self._alternate_images(x[:32], recon_x[:32])
            )
            self.logger.experiment.log(
                {
                    "val/samples": wandb.Image(samples),
                    "val/reconstructions": wandb.Image(reconstructions),
                },
                step=self.global_step,
            )

    def configure_optimizers(
        self,
    ) -> tuple[optim.Optimizer, optim.Optimizer]:
        gen_opt = optim.Adam(
            list(self.ali.enc.parameters()) + list(self.ali.dec.parameters()),
            lr=self.lr,
            betas=(0.5, 1 - 1e-3),
        )
        dis_opt = optim.Adam(
            self.ali.dis.parameters(), lr=self.lr, betas=(0.5, 1 - 1e-3)
        )
        return gen_opt, dis_opt

    def _alternate_images(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> list[torch.Tensor]:
        return [x for pair in zip(x1, x2) for x in pair]
