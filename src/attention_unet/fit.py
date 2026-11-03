from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from .engine import EpochResult, run_epoch


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train: EpochResult
    val: EpochResult


@dataclass(frozen=True)
class FitResult:
    history: list[EpochRecord]
    best_epoch: int
    best_val_iou: float
    checkpoint_path: str | None


def fit(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
    use_aef: bool,
    threshold: float,
    checkpoint_path: str | Path | None = None,
    early_stopping_patience: int | None = None,
) -> FitResult:
    """
    Train and validate an Attention U-Net.

    The function is intentionally dataset- and experiment-agnostic:
    train/validation splitting, batch size, learning rate, loss
    weighting, threshold selection, and compute configuration are
    supplied externally.

    The checkpoint with the best validation IoU is retained.

    Parameters
    ----------
    epochs:
        Maximum number of epochs.

    use_aef:
        If True, run the embedding-augmented model.
        If False, run the WSTS-only baseline.

    threshold:
        Probability threshold used for segmentation metrics.

    checkpoint_path:
        Optional path for saving the best validation-IoU checkpoint.

    early_stopping_patience:
        Optional number of consecutive non-improving epochs before
        stopping. None disables early stopping.
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    if (
        early_stopping_patience is not None
        and early_stopping_patience < 1
    ):
        raise ValueError(
            "early_stopping_patience must be "
            "at least 1 or None"
        )

    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else None
    )

    if checkpoint is not None:
        checkpoint.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    history: list[EpochRecord] = []

    best_val_iou = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            use_aef=use_aef,
            threshold=threshold,
            optimizer=optimizer,
        )

        val_result = run_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_aef=use_aef,
            threshold=threshold,
            optimizer=None,
        )

        record = EpochRecord(
            epoch=epoch,
            train=train_result,
            val=val_result,
        )

        history.append(record)

        if val_result.iou > best_val_iou:

            best_val_iou = val_result.iou
            best_epoch = epoch
            epochs_without_improvement = 0

            if checkpoint is not None:

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "val_iou":
                            val_result.iou,
                        "threshold":
                            threshold,
                        "use_aef":
                            use_aef,
                        "train_metrics":
                            asdict(train_result),
                        "val_metrics":
                            asdict(val_result),
                    },
                    checkpoint,
                )

        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement
            >= early_stopping_patience
        ):
            break

    return FitResult(
        history=history,
        best_epoch=best_epoch,
        best_val_iou=best_val_iou,
        checkpoint_path=(
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
    )