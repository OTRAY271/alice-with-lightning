import torch
from torch import nn


class BaseDiscriminator(nn.Module):
    def __init__(
        self,
        x_layers: nn.Module,
        z_layers: nn.Module,
        xz_layers: nn.Module,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.x_layers, self.z_layers, self.xz_layers = x_layers, z_layers, xz_layers
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h_x = self.x_layers(x)
        h_z = self.z_layers(z)
        h_xz = torch.cat([h_x, h_z], dim=1)
        return self.xz_layers(h_xz).view(-1, 1)

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
        lrelu_slope: float,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.LeakyReLU(lrelu_slope),
            nn.Dropout(dropout),
        )


class CIFAR10Discriminator(BaseDiscriminator):
    def __init__(self, latent_dim: int = 64):
        x_layers = nn.Sequential(
            self._conv_block(3, 32, kernel_size=5, stride=1, dropout=0.2),
            self._conv_block(32, 64, kernel_size=4, stride=2),
            self._conv_block(64, 128, kernel_size=4, stride=1),
            self._conv_block(128, 256, kernel_size=4, stride=2),
            self._conv_block(256, 512, kernel_size=4, stride=1),
        )

        z_layers = nn.Sequential(
            self._conv_block(latent_dim, 512, kernel_size=1, stride=1, dropout=0.2),
            self._conv_block(512, 512, kernel_size=1, stride=1),
        )

        xz_layers = nn.Sequential(
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1),
            nn.Dropout(0.5),
        )

        super().__init__(x_layers, z_layers, xz_layers, latent_dim)

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float = 0.5,
    ) -> nn.Module:
        return super()._conv_block(
            in_channels, out_channels, kernel_size, stride, dropout, 0.1
        )  # LeakyReLU is used instead of Maxout


class MNISTDiscriminator(BaseDiscriminator):
    def __init__(self, latent_dim: int = 64):
        x_layers = nn.Sequential(
            self._conv_block(1, 32, kernel_size=4, stride=2, dropout=0.2),
            self._conv_block(32, 64, kernel_size=4, stride=1),
            self._conv_block(64, 128, kernel_size=4, stride=2),
            self._conv_block(128, 256, kernel_size=4, stride=1),
            self._conv_block(256, 512, kernel_size=1, stride=1),
        )

        z_layers = nn.Sequential(
            self._conv_block(latent_dim, 512, kernel_size=1, stride=1, dropout=0.2),
            self._conv_block(512, 512, kernel_size=1, stride=1),
        )

        xz_layers = nn.Sequential(
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1),
            nn.Dropout(0.5),
        )

        super().__init__(x_layers, z_layers, xz_layers, latent_dim)

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float = 0.5,
    ) -> nn.Module:
        return super()._conv_block(
            in_channels, out_channels, kernel_size, stride, dropout, 0.1
        )


class CelebADiscriminator(BaseDiscriminator):
    def __init__(self, latent_dim: int = 256):
        x_layers = nn.Sequential(
            self._conv_block(3, 64, kernel_size=2, stride=1),  # no batchnorm
            self._conv_block(64, 128, kernel_size=7, stride=2),
            self._conv_block(128, 256, kernel_size=5, stride=2),
            self._conv_block(256, 256, kernel_size=7, stride=2),
            self._conv_block(256, 512, kernel_size=4, stride=1),
        )

        z_layers = nn.Sequential(
            self._conv_block(latent_dim, 1024, kernel_size=1, stride=1),
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
        )

        xz_layers = nn.Sequential(
            self._conv_block(1536, 2048, kernel_size=1, stride=1),
            self._conv_block(2048, 2048, kernel_size=1, stride=1),
            nn.Conv2d(2048, 1, kernel_size=1, stride=1),
            nn.Dropout(0.2),
        )

        super().__init__(x_layers, z_layers, xz_layers, latent_dim)

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> nn.Module:
        return super()._conv_block(
            in_channels, out_channels, kernel_size, stride, 0.2, 0.02
        )
