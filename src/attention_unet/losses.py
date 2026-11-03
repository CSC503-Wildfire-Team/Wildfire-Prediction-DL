from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_target_and_mask(
    target: torch.Tensor,
    prediction: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    # Allow target [B, H, W].
    if target.ndim == prediction.ndim - 1:
        target = target.unsqueeze(1)

    if target.shape != prediction.shape:
        raise ValueError(
            "Target/prediction shape mismatch: "
            f"target={tuple(target.shape)}, "
            f"prediction={tuple(prediction.shape)}"
        )

    target = target.to(
        dtype=prediction.dtype,
        device=prediction.device,
    )

    if valid_mask is None:
        # Supports either ordinary 0/1 targets or
        # datasets that mark invalid pixels using -1 / NaN.
        valid_mask = (
            torch.isfinite(target)
            & (target >= 0.0)
            & (target <= 1.0)
        )

    else:
        if valid_mask.ndim == prediction.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)

        if valid_mask.shape != prediction.shape:
            raise ValueError(
                "Mask/prediction shape mismatch: "
                f"mask={tuple(valid_mask.shape)}, "
                f"prediction={tuple(prediction.shape)}"
            )

        valid_mask = valid_mask.to(
            device=prediction.device,
            dtype=torch.bool,
        )

        valid_mask = (
            valid_mask
            & torch.isfinite(target)
        )

    # Invalid labels must not enter BCE.
    safe_target = torch.where(
        valid_mask,
        target,
        torch.zeros_like(target),
    )

    return safe_target, valid_mask


class HybridBCEDiceLoss(nn.Module):
    """
    L = lambda * BCE + (1 - lambda) * DiceLoss.

    bce_weight corresponds to lambda.

    Keep this configurable because the current methodology
    defines lambda but does not yet specify its final value.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        eps: float = 1e-7,
        from_logits: bool = False,
    ) -> None:
        super().__init__()

        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(
                "bce_weight must be in [0, 1]"
            )

        self.bce_weight = bce_weight
        self.eps = eps
        self.from_logits = from_logits

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        target, valid_mask = (
            _prepare_target_and_mask(
                target,
                prediction,
                valid_mask,
            )
        )

        n_valid = valid_mask.sum()

        if n_valid == 0:
            raise ValueError(
                "Loss received a batch "
                "with no valid target pixels"
            )

        if self.from_logits:
            probs = torch.sigmoid(
                prediction
            )

            bce_per_pixel = (
                F.binary_cross_entropy_with_logits(
                    prediction,
                    target,
                    reduction="none",
                )
            )

        else:
            probs = prediction.clamp(
                self.eps,
                1.0 - self.eps,
            )

            bce_per_pixel = (
                F.binary_cross_entropy(
                    probs,
                    target,
                    reduction="none",
                )
            )

        mask = valid_mask.to(
            dtype=prediction.dtype
        )

        bce = (
            (bce_per_pixel * mask).sum()
            / n_valid.to(
                dtype=prediction.dtype
            )
        )

        intersection = (
            probs
            * target
            * mask
        ).sum()

        denominator = (
            (probs * mask).sum()
            + (target * mask).sum()
        )

        dice_loss = 1.0 - (
            (
                2.0 * intersection
                + self.eps
            )
            /
            (
                denominator
                + self.eps
            )
        )

        return (
            self.bce_weight * bce
            + (1.0 - self.bce_weight)
            * dice_loss
        )