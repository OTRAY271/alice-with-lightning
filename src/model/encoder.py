import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self, layers: nn.Module, latent_dim: int):
        super().__init__()

        self.layers = layers
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layers(x)
        if self.training:
            return z[:, : self.latent_dim] * torch.exp(z[:, self.latent_dim :])
        return z[:, : self.latent_dim]

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        lrelu_slope: float,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(lrelu_slope),
        )


class CIFAR10Encoder(BaseEncoder):
    def __init__(self, latent_dim: int = 64):
        layers = nn.Sequential(
            self._conv_block(3, 32, kernel_size=5, stride=1),
            self._conv_block(32, 64, kernel_size=4, stride=2),
            self._conv_block(64, 128, kernel_size=4, stride=1),
            self._conv_block(128, 256, kernel_size=4, stride=2),
            self._conv_block(256, 512, kernel_size=4, stride=1),
            self._conv_block(512, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, latent_dim * 2, kernel_size=1, stride=1),
        )

        super().__init__(layers, latent_dim)

    def _conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._conv_block(in_channels, out_channels, kernel_size, stride, 0.1)


class MNISTEncoder(BaseEncoder):
    def __init__(self, latent_dim: int = 64):
        layers = nn.Sequential(
            self._conv_block(1, 32, kernel_size=4, stride=2),
            self._conv_block(32, 64, kernel_size=4, stride=1),
            self._conv_block(64, 128, kernel_size=4, stride=2),
            self._conv_block(128, 256, kernel_size=4, stride=1),
            self._conv_block(256, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, latent_dim * 2, kernel_size=1, stride=1),
        )

        super().__init__(layers, latent_dim)

    def _conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._conv_block(in_channels, out_channels, kernel_size, stride, 0.1)


class CelebAEncoder(BaseEncoder):
    def __init__(self, latent_dim: int = 256):
        layers = nn.Sequential(
            self._conv_block(3, 64, kernel_size=2, stride=1),
            self._conv_block(64, 128, kernel_size=7, stride=2),
            self._conv_block(128, 256, kernel_size=5, stride=2),
            self._conv_block(256, 256, kernel_size=7, stride=2),
            self._conv_block(256, 512, kernel_size=4, stride=1),
            nn.Conv2d(512, latent_dim * 2, kernel_size=1, stride=1),
        )

        super().__init__(layers, latent_dim)

    def _conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._conv_block(in_channels, out_channels, kernel_size, stride, 0.02)
