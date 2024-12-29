import torch
from torch import nn


class BaseCCDiscriminator(nn.Module):
    def __init__(self, layers: nn.Module):
        super().__init__()

        self.layers = layers

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.layers(x).view(-1, 1)

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


class MNISTCCDiscriminatorX(BaseCCDiscriminator):
    def __init__(self):
        layers = nn.Sequential(
            self._double_conv_block(2, 16),
            self._double_conv_block(16, 32, second_padding=2),
            self._double_conv_block(32, 64),
            self._conv_block(64, 50, kernel_size=4, stride=1),
            nn.Conv2d(50, 1, kernel_size=1, stride=1),
        )

        super().__init__(layers)

    def _double_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        first_padding: int = 1,
        second_padding: int = 1,
        lrelu_slope: float = 0.1,
        dropout: float = 0.5,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=first_padding,
            ),
            nn.LeakyReLU(lrelu_slope),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=second_padding,
            ),
            nn.LeakyReLU(lrelu_slope),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.LeakyReLU(lrelu_slope),
            nn.Dropout(dropout),
        )

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


class MNISTCCDiscriminatorZ(BaseCCDiscriminator):
    def __init__(self, latent_dim: int = 64):
        layers = nn.Sequential(
            self._conv_block(latent_dim * 2, 512, kernel_size=1, stride=1, dropout=0.2),
            self._conv_block(512, 512, kernel_size=1, stride=1),
            self._conv_block(512, 512, kernel_size=1, stride=1),
            self._conv_block(512, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 1, kernel_size=1, stride=1),
        )

        super().__init__(layers)

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
