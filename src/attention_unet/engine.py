from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .metrics import confusion_counts


@dataclass(frozen=True)
class EpochResult:
    loss: float
    iou: float
    precision: float
    recall: float
    dice: float
    f1: float


def prepare_batch(
    batch: Any,
    use_aef: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Convert a DataLoader batch into model input and target.

    Supported forms:

        (x, y)

        (x, e, y)

        (x, e, y, valid_mask)

    or a dictionary containing:
        x, y
        optional e
        optional valid_mask

    Parameters
    ----------
    use_aef:
        If True, concatenate WSTS and AlphaEarth channels.
        If False, use WSTS features only.

    Returns
    -------
    inputs:
        [B, C, H, W]

    target:
        [B, H, W] or [B, 1, H, W]

    valid_mask:
        Optional validity mask.
    """

    e = None
    valid_mask = None

    if isinstance(batch, dict):
        x = batch["x"]
        y = batch["y"]

        e = batch.get("e")
        valid_mask = batch.get("valid_mask")

    elif isinstance(batch, (tuple, list)):

        if len(batch) == 2:
            x, y = batch

        elif len(batch) == 3:
            x, e, y = batch

        elif len(batch) == 4:
            x, e, y, valid_mask = batch

        else:
            raise ValueError(
                "Expected batch with 2, 3, or 4 elements; "
                f"got {len(batch)}"
            )

    else:
        raise TypeError(
            "Batch must be a tuple, list, or dictionary"
        )

    if x.ndim != 4:
        raise ValueError(
            "Expected WSTS tensor x with shape "
            f"[B, C, H, W], got {tuple(x.shape)}"
        )

    if use_aef:

        if e is None:
            raise ValueError(
                "use_aef=True requires AlphaEarth "
                "embeddings in the batch"
            )

        if e.ndim != 4:
            raise ValueError(
                "Expected AEF tensor e with shape "
                f"[B, C, H, W], got {tuple(e.shape)}"
            )

        if x.shape[0] != e.shape[0]:
            raise ValueError(
                "WSTS and AEF batch sizes do not match"
            )

        if x.shape[-2:] != e.shape[-2:]:
            raise ValueError(
                "WSTS and AEF spatial dimensions "
                f"do not match: {x.shape[-2:]} "
                f"vs {e.shape[-2:]}"
            )

        inputs = torch.cat(
            [x, e],
            dim=1,
        )

    else:
        inputs = x

    return inputs, y, valid_mask


def _metrics_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:

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


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    use_aef: bool,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """
    Run one training or validation epoch.

    Passing an optimizer enables training.
    optimizer=None performs validation/inference.

    No dataset splitting or hyperparameter selection is
    performed here; those remain external to the engine.
    """

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    total_tp = torch.tensor(
        0.0,
        device=device,
    )

    total_fp = torch.tensor(
        0.0,
        device=device,
    )

    total_fn = torch.tensor(
        0.0,
        device=device,
    )

    for batch in loader:

        inputs, target, valid_mask = (
            prepare_batch(
                batch,
                use_aef=use_aef,
            )
        )

        inputs = inputs.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        target = target.to(
            device=device,
            non_blocking=True,
        )

        if valid_mask is not None:
            valid_mask = valid_mask.to(
                device=device,
                non_blocking=True,
            )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            prediction = model(inputs)

            loss = loss_fn(
                prediction,
                target,
                valid_mask,
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.shape[0]

        total_loss += (
            float(loss.detach().item())
            * batch_size
        )

        total_samples += batch_size

        tp, fp, fn = confusion_counts(
            prediction.detach(),
            target,
            threshold=threshold,
            valid_mask=valid_mask,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_samples == 0:
        raise ValueError(
            "DataLoader produced no samples"
        )

    metrics = _metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
    )

    return EpochResult(
        loss=total_loss / total_samples,
        **metrics,
    )