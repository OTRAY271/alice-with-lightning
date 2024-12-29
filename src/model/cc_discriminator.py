import torch
from torch import nn


class BaseCCDiscriminator(nn.Module):
    def __init__(
        self,
        x_layers: nn.Module,
        xx_layers: nn.Module,
    ):
        super().__init__()

        self.x_layers, self.xx_layers = x_layers, xx_layers

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
    def __init__(self):
        x_layers = nn.Sequential(
            self._conv_block(1, 16, kernel_size=4, stride=2, dropout=0.2),
            self._conv_block(16, 32, kernel_size=4, stride=1),
            self._conv_block(32, 64, kernel_size=4, stride=2),
            self._conv_block(64, 128, kernel_size=4, stride=1),
            self._conv_block(128, 256, kernel_size=1, stride=1),
        )

        xx_layers = nn.Sequential(
            self._conv_block(512, 512, kernel_size=1, stride=1),
            self._conv_block(512, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 1, kernel_size=1, stride=1),
        )

        super().__init__(x_layers, xx_layers)

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


class MNISTCCDiscriminator2(BaseCCDiscriminator):
    def __init__(self):
        super().__init__(None, None)

        self.layers = nn.Sequential(
            self._double_conv_block(2, 16),
            self._double_conv_block(16, 32, second_padding=2),
            self._double_conv_block(32, 64),
            self._conv_block(64, 50, kernel_size=4, stride=1),
            nn.Conv2d(50, 1, kernel_size=1, stride=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.layers(x).view(-1, 1)

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
