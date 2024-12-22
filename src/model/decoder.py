import torch
from torch import nn


class BaseDecoder(nn.Module):
    def __init__(self, layers: nn.Module, latent_dim: int = 64):
        super().__init__()

        self.layers = layers
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)

    def _convt_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        lrelu_slope: float,
    ) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(lrelu_slope),
        )


class CIFAR10Decoder(BaseDecoder):
    def __init__(self, latent_dim: int = 64):
        layers = nn.Sequential(
            self._convt_block(latent_dim, 256, kernel_size=4, stride=1),
            self._convt_block(256, 128, kernel_size=4, stride=2),
            self._convt_block(128, 64, kernel_size=4, stride=1),
            self._convt_block(64, 32, kernel_size=4, stride=2),
            self._convt_block(32, 32, kernel_size=5, stride=1),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        super().__init__(layers, latent_dim)

    def _convt_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._convt_block(in_channels, out_channels, kernel_size, stride, 0.1)


class MNISTDecoder(BaseDecoder):
    def __init__(self, latent_dim: int = 64):
        layers = nn.Sequential(
            self._convt_block(latent_dim, 256, kernel_size=4, stride=1),
            self._convt_block(256, 128, kernel_size=4, stride=2),
            self._convt_block(128, 64, kernel_size=4, stride=1),
            self._convt_block(64, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        super().__init__(layers, latent_dim)

    def _convt_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._convt_block(in_channels, out_channels, kernel_size, stride, 0.1)


class CelebADecoder(BaseDecoder):
    def __init__(self, latent_dim: int = 256):
        layers = nn.Sequential(
            self._convt_block(latent_dim, 512, kernel_size=4, stride=1),
            self._convt_block(512, 256, kernel_size=7, stride=2),
            self._convt_block(256, 256, kernel_size=5, stride=2),
            self._convt_block(256, 128, kernel_size=7, stride=2),
            self._convt_block(128, 64, kernel_size=2, stride=1),
            nn.Conv2d(64, 3, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        super().__init__(layers, latent_dim)

    def _convt_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Module:
        return super()._convt_block(
            in_channels, out_channels, kernel_size, stride, 0.02
        )
