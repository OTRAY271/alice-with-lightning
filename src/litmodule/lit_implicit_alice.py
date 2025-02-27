import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim

import wandb
from model import ALICE


class LitImplicitALICE(L.LightningModule):
    def __init__(self, alice: ALICE, lr: float = 1e-4, cc_loss_weight: float = 1.0):
        super().__init__()

        self.alice = alice
        self.lr = lr
        self.cc_loss_weight = cc_loss_weight
        self.criterion = nn.BCEWithLogitsLoss()

        self.z_val = torch.randn(64, self.alice.latent_dim, 1, 1)

        self.automatic_optimization = False

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        gen_opt, dis_opt = self.optimizers()

        x, _ = batch

        batch_size = x.size(0)
        z = torch.randn(batch_size, self.alice.latent_dim, 1, 1, device=self.device)

        z_fake = self.alice.enc(x)
        x_fake = self.alice.dec(z)
        x_recon = self.alice.dec(z_fake)
        z_recon = self.alice.enc(x_fake)

        label_real = torch.ones(batch_size, 1, device=self.device)
        label_fake = torch.zeros(batch_size, 1, device=self.device)

        pred_real = self.alice.dis(x, z_fake.detach())
        pred_fake = self.alice.dis(x_fake.detach(), z)
        pred_cc_real_x = self.alice.ccdis_x(x, x)
        pred_cc_fake_x = self.alice.ccdis_x(x, x_recon.detach())
        pred_cc_real_z = self.alice.ccdis_z(z, z)
        pred_cc_fake_z = self.alice.ccdis_z(z, z_recon.detach())

        loss_dis_real = self.criterion(pred_real, label_real) + self.cc_loss_weight * (
            self.criterion(pred_cc_real_x, label_real) * 0.5
            + self.criterion(pred_cc_real_z, label_real) * 0.5
        )
        loss_dis_fake = self.criterion(pred_fake, label_fake) + self.cc_loss_weight * (
            self.criterion(pred_cc_fake_x, label_fake) * 0.5
            + self.criterion(pred_cc_fake_z, label_fake) * 0.5
        )
        loss_dis = loss_dis_real + loss_dis_fake

        self.alice.dis.zero_grad()
        self.alice.ccdis_x.zero_grad()
        self.alice.ccdis_z.zero_grad()
        self.manual_backward(loss_dis)
        dis_opt.step()

        pred_real = self.alice.dis(x, z_fake)
        pred_fake = self.alice.dis(x_fake, z)
        pred_cc_real_x = self.alice.ccdis_x(x, x)
        pred_cc_fake_x = self.alice.ccdis_x(x, x_recon.detach())
        pred_cc_real_z = self.alice.ccdis_z(z, z)
        pred_cc_fake_z = self.alice.ccdis_z(z, z_recon.detach())

        loss_gen_real = self.criterion(pred_fake, label_real) + self.cc_loss_weight * (
            self.criterion(pred_cc_fake_x, label_real) * 0.5
            + self.criterion(pred_cc_fake_z, label_real) * 0.5
        )
        loss_gen_fake = self.criterion(pred_real, label_fake) + self.cc_loss_weight * (
            self.criterion(pred_cc_real_x, label_fake) * 0.5
            + self.criterion(pred_cc_real_z, label_fake) * 0.5
        )
        loss_gen = loss_gen_real + loss_gen_fake

        self.alice.enc.zero_grad()
        self.alice.dec.zero_grad()
        self.manual_backward(loss_gen)
        gen_opt.step()

        self.log_dict(
            {
                "train/gen_loss": loss_gen,
                "train/dis_loss": loss_dis,
                "train/recon_loss": F.mse_loss(x_recon, x),
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

        recon_x = self.alice.dec(self.alice.enc(x))

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
                self.alice.dec(self.z_val.to(self.device))
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
            list(self.alice.enc.parameters()) + list(self.alice.dec.parameters()),
            lr=self.lr,
            betas=(0.5, 1 - 1e-3),
        )
        dis_opt = optim.Adam(
            list(self.alice.dis.parameters())
            + list(self.alice.ccdis_x.parameters())
            + list(self.alice.ccdis_z.parameters()),
            lr=self.lr,
            betas=(0.5, 1 - 1e-3),
        )
        return gen_opt, dis_opt

    def _alternate_images(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> list[torch.Tensor]:
        return [x for pair in zip(x1, x2) for x in pair]
