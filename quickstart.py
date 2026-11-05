from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.attention_unet import (
    AttentionUNet,
    HybridBCEDiceLoss,
    fit,
    set_seed,
)


WSTS_CHANNELS = 40
AEF_CHANNELS = 64


class SyntheticWildfireDataset(Dataset):
    """
    Lightweight dataset that reproduces the model-side interface of the
    WildfireSpreadTS + AlphaEarth pipeline without requiring the full dataset.

    Each sample contains:
        x: [40, H, W] WSTS-like predictor channels
        e: [64, H, W] AlphaEarth-like embedding channels
        y: [H, W] binary next-day active-fire mask

    The feature values are synthetic. Their purpose is to exercise the
    model/training pipeline, not to provide meaningful wildfire predictions.
    """

    def __init__(
        self,
        num_samples: int = 4,
        height: int = 32,
        width: int = 32,
    ) -> None:
        super().__init__()

        if num_samples < 1:
            raise ValueError("num_samples must be positive")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "height and width must be divisible by 16 "
                "for the four-level Attention U-Net"
            )

        self.x = torch.randn(
            num_samples,
            WSTS_CHANNELS,
            height,
            width,
        )

        self.e = torch.randn(
            num_samples,
            AEF_CHANNELS,
            height,
            width,
        )

        # Sparse binary masks roughly mimic the strong foreground/background
        # imbalance of wildfire segmentation without pretending to be real data.
        self.y = torch.zeros(
            num_samples,
            height,
            width,
            dtype=torch.float32,
        )

        for index in range(num_samples):
            top = 4 + (index * 3) % max(height - 8, 1)
            left = 4 + (index * 5) % max(width - 8, 1)

            self.y[
                index,
                top : top + 4,
                left : left + 4,
            ] = 1.0

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return (
            self.x[index],
            self.e[index],
            self.y[index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight end-to-end smoke test of the "
            "wildfire Attention U-Net training pipeline."
        )
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Use the 40-channel WSTS-only baseline instead of "
            "the 104-channel WSTS + AEF configuration."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of quickstart training epochs (default: 1).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size (default: 2).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device (default: cpu).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Run without '--device cuda' to use CPU."
        )

    set_seed(args.seed)

    device = torch.device(args.device)

    use_aef = not args.baseline
    n_in_channels = (
        WSTS_CHANNELS + AEF_CHANNELS
        if use_aef
        else WSTS_CHANNELS
    )

    # Small synthetic datasets keep the quickstart fast and independent
    # of the full WildfireSpreadTS / AlphaEarth download.
    train_dataset = SyntheticWildfireDataset(
        num_samples=4,
        height=32,
        width=32,
    )

    val_dataset = SyntheticWildfireDataset(
        num_samples=2,
        height=32,
        width=32,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # base_channels=4 is intentionally lightweight for the quickstart.
    # The production architecture remains configurable independently.
    model = AttentionUNet(
        n_in_channels=n_in_channels,
        base_channels=4,
        use_se=True,
    ).to(device)

    loss_fn = HybridBCEDiceLoss(
        bce_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    mode_name = (
        "WSTS + AEF"
        if use_aef
        else "WSTS-only baseline"
    )

    sample_x, sample_e, sample_y = next(
        iter(train_loader)
    )

    print("=" * 60)
    print("Wildfire Prediction Quickstart")
    print("=" * 60)

    print(f"Mode:        {mode_name}")
    print(f"Device:      {device}")
    print(f"Input chans: {n_in_channels}")
    print(f"WSTS batch:  {tuple(sample_x.shape)}")

    if use_aef:
        print(f"AEF batch:   {tuple(sample_e.shape)}")

    print(f"Target:      {tuple(sample_y.shape)}")
    print()

    # Use a temporary directory so the quickstart verifies checkpointing
    # without leaving generated model files in the repository.
    with tempfile.TemporaryDirectory(
        prefix="wildfire_quickstart_"
    ) as temporary_directory:

        checkpoint_path = (
            Path(temporary_directory)
            / "best_model.pt"
        )

        result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            use_aef=use_aef,
            threshold=0.5,
            checkpoint_path=checkpoint_path,
        )

        if not checkpoint_path.exists():
            raise RuntimeError(
                "Quickstart training completed but no "
                "checkpoint was created."
            )

        final_record = result.history[-1]

        print("Training complete")
        print("-" * 60)
        print(
            f"Final train loss: "
            f"{final_record.train.loss:.4f}"
        )
        print(
            f"Final val loss:   "
            f"{final_record.val.loss:.4f}"
        )
        print(
            f"Final val IoU:    "
            f"{final_record.val.iou:.4f}"
        )
        print(
            f"Final val Dice:   "
            f"{final_record.val.dice:.4f}"
        )
        print(
            f"Best epoch:       "
            f"{result.best_epoch}"
        )
        print(
            f"Best val IoU:     "
            f"{result.best_val_iou:.4f}"
        )
        print(
            "Checkpoint:       verified "
            "(temporary quickstart output)"
        )

    print()
    print("Quickstart completed successfully.")
    print(
        "Note: synthetic inputs validate the software pipeline; "
        "they are not wildfire benchmark results."
    )


if __name__ == "__main__":
    main()