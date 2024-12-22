import torch
from torch import nn


class BaseCCDiscriminator(nn.Module):
    def __init__(
        self,
        x_layers: nn.Module,
        xx_layers: nn.Module,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.x_layers, self.xx_layers = x_layers, xx_layers
        self.latent_dim = latent_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h_x1 = self.x_layers(x1)
        h_x2 = self.x_layers(x2)
        h_xx = torch.cat([h_x1, h_x2], dim=1)
        return self.xx_layers(h_xx).view(-1, 1)

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


class MNISTCCDiscriminator(BaseCCDiscriminator):
    def __init__(self, latent_dim: int = 64):
        x_layers = nn.Sequential(
            self._conv_block(1, 32, kernel_size=4, stride=2, dropout=0.2),
            self._conv_block(32, 64, kernel_size=4, stride=1),
            self._conv_block(64, 128, kernel_size=4, stride=2),
            self._conv_block(128, 256, kernel_size=4, stride=1),
            self._conv_block(256, 512, kernel_size=1, stride=1),
        )

        xx_layers = nn.Sequential(
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            self._conv_block(1024, 1024, kernel_size=1, stride=1),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1),
            nn.Dropout(0.5),
        )

        super().__init__(x_layers, xx_layers, latent_dim)

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
