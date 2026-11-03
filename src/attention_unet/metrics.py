from __future__ import annotations

import torch


def _prepare_binary_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    valid_mask: torch.Tensor | None = None,
    from_logits: bool = False,
):

    if target.ndim == prediction.ndim - 1:
        target = target.unsqueeze(1)

    if target.shape != prediction.shape:
        raise ValueError(
            "Target/prediction shape mismatch: "
            f"target={tuple(target.shape)}, "
            f"prediction={tuple(prediction.shape)}"
        )

    target = target.to(
        device=prediction.device
    )

    if from_logits:
        probabilities = torch.sigmoid(
            prediction
        )
    else:
        probabilities = prediction

    if valid_mask is None:
        valid_mask = (
            torch.isfinite(target)
            & (target >= 0)
            & (target <= 1)
        )

    else:
        if valid_mask.ndim == prediction.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)

        valid_mask = valid_mask.to(
            device=prediction.device,
            dtype=torch.bool,
        )

        valid_mask = (
            valid_mask
            & torch.isfinite(target)
        )

    predicted_fire = (
        probabilities >= threshold
    )

    true_fire = (
        target >= 0.5
    )

    return (
        predicted_fire,
        true_fire,
        valid_mask,
    )


def confusion_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
    from_logits: bool = False,
):

    predicted_fire, true_fire, mask = (
        _prepare_binary_inputs(
            prediction,
            target,
            threshold,
            valid_mask,
            from_logits,
        )
    )

    tp = (
        predicted_fire
        & true_fire
        & mask
    ).sum().float()

    fp = (
        predicted_fire
        & ~true_fire
        & mask
    ).sum().float()

    fn = (
        ~predicted_fire
        & true_fire
        & mask
    ).sum().float()

    return tp, fp, fn


def segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
    from_logits: bool = False,
    eps: float = 1e-7,
) -> dict[str, float]:

    tp, fp, fn = confusion_counts(
        prediction,
        target,
        threshold,
        valid_mask,
        from_logits,
    )

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    precision = (
        tp
        / (tp + fp + eps)
    )

    recall = (
        tp
        / (tp + fn + eps)
    )

    dice = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
            + eps
        )
    )

    return {
        "iou": float(iou.item()),
        "precision": float(
            precision.item()
        ),
        "recall": float(
            recall.item()
        ),
        "dice": float(
            dice.item()
        ),
        "f1": float(
            f1.item()
        ),
    }