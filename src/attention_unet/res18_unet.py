from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Res18UNetBaseline(nn.Module):
    """
    WildfireSpreadTS ResNet-18 U-Net baseline (adapted from SMPModel.py).
    Takes [B, 40, H, W] and returns raw logits [B, 1, H, W].
    """

    def __init__(self, n_in_channels: int = 40) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=n_in_channels,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      if x.ndim == 5:
        x = x.flatten(start_dim=1, end_dim=2)
      return torch.sigmoid(self.model(x))