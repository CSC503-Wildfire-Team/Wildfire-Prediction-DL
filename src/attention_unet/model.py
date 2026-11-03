from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two 3x3 convolutions, each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate for a U-Net skip connection.

    The decoder feature provides the gating signal.
    The resulting one-channel attention map is applied
    to the corresponding encoder feature map.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        # W_x F_l
        self.skip_projection = nn.Conv2d(
            skip_channels,
            inter_channels,
            kernel_size=1,
            bias=True,
        )

        # W_g G_l
        self.gate_projection = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

        # Project combined features to one spatial attention score.
        self.attention_projection = nn.Conv2d(
            inter_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        skip: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:

        if skip.shape[-2:] != gate.shape[-2:]:
            raise ValueError(
                "AttentionGate spatial mismatch: "
                f"skip={skip.shape[-2:]}, "
                f"gate={gate.shape[-2:]}"
            )

        scores = (
            self.skip_projection(skip)
            + self.gate_projection(gate)
        )

        scores = self.relu(scores)

        alpha = self.sigmoid(
            self.attention_projection(scores)
        )

        # alpha has one channel and broadcasts across
        # all channels of the encoder feature.
        return skip * alpha


class DecoderBlock(nn.Module):
    """
    Upsample decoder features, apply attention to the
    encoder skip, concatenate both, then DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gate_channels=out_channels,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        gated_skip = self.attention(
            skip=skip,
            gate=x,
        )

        x = torch.cat(
            [gated_skip, x],
            dim=1,
        )

        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Four-level Attention U-Net.

    Default architecture:
        Encoder:
            64 -> 128 -> 256 -> 512

        Bridge:
            1024

        Decoder:
            512 -> 256 -> 128 -> 64

        Output:
            one-channel probability map

    n_in_channels is intentionally configurable because the
    final wildfire dataset / AlphaEarth feature count may change.
    """

    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int = 1,
        base_channels: int = 64,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()

        if n_in_channels <= 0:
            raise ValueError(
                "n_in_channels must be positive"
            )

        if n_out_channels <= 0:
            raise ValueError(
                "n_out_channels must be positive"
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        bridge_channels = base_channels * 16

        self.apply_sigmoid = apply_sigmoid

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Encoder
        self.enc1 = DoubleConv(
            n_in_channels,
            c1,
        )

        self.enc2 = DoubleConv(
            c1,
            c2,
        )

        self.enc3 = DoubleConv(
            c2,
            c3,
        )

        self.enc4 = DoubleConv(
            c3,
            c4,
        )

        # Bridge
        self.bridge = DoubleConv(
            c4,
            bridge_channels,
        )

        # Decoder
        self.dec4 = DecoderBlock(
            bridge_channels,
            c4,
            c4,
        )

        self.dec3 = DecoderBlock(
            c4,
            c3,
            c3,
        )

        self.dec2 = DecoderBlock(
            c3,
            c2,
            c2,
        )

        self.dec1 = DecoderBlock(
            c2,
            c1,
            c1,
        )

        # Pixel-wise binary prediction.
        self.output_conv = nn.Conv2d(
            c1,
            n_out_channels,
            kernel_size=1,
        )

        self.output_activation = nn.Sigmoid()

    @staticmethod
    def _validate_input(
        x: torch.Tensor,
    ) -> None:

        if x.ndim != 4:
            raise ValueError(
                "Expected input [B, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        height, width = x.shape[-2:]

        # Four max-pooling operations divide each
        # spatial dimension by 16.
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "H and W must be divisible by 16 "
                "for four encoder pooling levels; "
                f"got H={height}, W={width}"
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_input(x)

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bridge
        bridge = self.bridge(
            self.pool(e4)
        )

        # Attention-gated decoder
        d4 = self.dec4(
            bridge,
            e4,
        )

        d3 = self.dec3(
            d4,
            e3,
        )

        d2 = self.dec2(
            d3,
            e2,
        )

        d1 = self.dec1(
            d2,
            e1,
        )

        logits = self.output_conv(d1)

        if self.apply_sigmoid:
            return self.output_activation(logits)

        return logits