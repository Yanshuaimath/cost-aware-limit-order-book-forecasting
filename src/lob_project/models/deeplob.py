from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SamePadConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh = math.ceil(ih / sh)
        ow = math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


def conv_activation(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    return nn.Sequential(
        SamePadConv2d(in_channels, out_channels, kernel_size, stride=stride),
        nn.LeakyReLU(negative_slope=0.01),
    )


class InceptionLOB(nn.Module):
    def __init__(self, in_channels: int = 16, branch_channels: int = 32) -> None:
        super().__init__()
        self.branch_1 = conv_activation(in_channels, branch_channels, (1, 1))
        self.branch_3 = nn.Sequential(
            conv_activation(in_channels, branch_channels, (1, 1)),
            conv_activation(branch_channels, branch_channels, (3, 1)),
        )
        self.branch_5 = nn.Sequential(
            conv_activation(in_channels, branch_channels, (1, 1)),
            conv_activation(branch_channels, branch_channels, (5, 1)),
        )
        self.branch_pool_conv = conv_activation(in_channels, branch_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(x, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        return torch.cat(
            [self.branch_1(x), self.branch_3(x), self.branch_5(x), self.branch_pool_conv(pooled)],
            dim=1,
        )


class DeepLOB(nn.Module):
    """CNN-Inception-LSTM architecture following the DeepLOB paper."""

    def __init__(self, n_classes: int = 3, lstm_hidden: int = 64) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            conv_activation(1, 16, (1, 2), stride=(1, 2)),
            conv_activation(16, 16, (4, 1)),
            conv_activation(16, 16, (4, 1)),
        )
        self.conv_block_2 = nn.Sequential(
            conv_activation(16, 16, (1, 2), stride=(1, 2)),
            conv_activation(16, 16, (4, 1)),
            conv_activation(16, 16, (4, 1)),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            conv_activation(16, 16, (4, 1)),
            conv_activation(16, 16, (4, 1)),
        )
        self.inception = InceptionLOB(in_channels=16, branch_channels=32)
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 40:
            raise ValueError(f"DeepLOB expects [batch, time, 40], received {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        if x.shape[-1] != 1:
            raise RuntimeError(f"Feature-width collapse failed; received width={x.shape[-1]}")
        x = self.inception(x).squeeze(-1).transpose(1, 2)
        sequence, _ = self.lstm(x)
        return self.classifier(sequence[:, -1])
