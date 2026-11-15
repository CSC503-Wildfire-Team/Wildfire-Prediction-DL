from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wsts_aef_dataset import (
    WSTSAEFDataset,
    build_datasets,
    build_loaders,
)
from src.attention_unet.model import AttentionUNet
from src.attention_unet.losses import HybridBCEDiceLoss
from src.attention_unet.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shared real-data smoke test for M4 RQ1."
    )

    parser.add_argument(
        "--mode",
        choices=("wsts-control", "proposed"),
        required=True,
        help=(
            "wsts-control: real 40-channel WSTS only; "
            "proposed: real 40-channel WSTS + real 64-channel AEF."
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root of HDF5 WSTS dataset.",
    )

    parser.add_argument(
        "--aef-dir",
        default=None,
        help="Directory containing <year>__<fire>.tif AEF files.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="M4 fold number (default: 0).",
    )

    # Only required for proposed mode.
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Fire year for proposed single-fire smoke test.",
    )

    parser.add_argument(
        "--fire",
        default=None,
        help="Fire name for proposed single-fire smoke test.",
    )

    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "m4_rq1.json"),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
        help="Small model width for smoke testing only.",
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(name):
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    return torch.device(name)


def find_first_index_for_fire(dataset, year, fire):
    """
    Return the first dataset index belonging to the requested fire.
    """

    offset = 0

    for fire_year, fires in dataset.datapoints_per_fire.items():
        for fire_name, count in fires.items():

            if fire_year == year and fire_name == fire:
                if count <= 0:
                    raise RuntimeError(
                        f"{year}/{fire} has no usable next-day samples."
                    )

                return offset

            offset += count

    raise RuntimeError(
        f"Could not find fire {year}/{fire} in DATA_DIR."
    )


def load_wsts_control_batch(args, config):
    """
    Real HDF5 test using the actual M4 fold mapping.
    No AEF files are required.
    """

    fold_cfg = config["m4_folds"][str(args.fold)]
    loader_fold = fold_cfg["loader_fold"]

    print("Building real WSTS datasets...")
    print(f"M4 fold:      {args.fold}")
    print(f"Loader fold:  {loader_fold}")
    print(f"Train years:  {fold_cfg['train_years']}")
    print(f"Val year:     {fold_cfg['val_year']}")
    print(f"Test year:    {fold_cfg['test_year']}")

    # aef_dir must be syntactically present because the dataset
    # constructor accepts it, but it is never read when use_aef=False.
    unused_aef_dir = str(ROOT / "_unused_aef")

    train_ds, val_ds, test_ds = build_datasets(
        data_dir=args.data_dir,
        aef_dir=unused_aef_dir,
        fold=loader_fold,
        crop_side_length=config["crop_side_length"],
        load_from_hdf5=True,
        use_aef=False,
    )

    print()
    print("Dataset sizes:")
    print("train:", len(train_ds))
    print("val:  ", len(val_ds))
    print("test: ", len(test_ds))

    if min(len(train_ds), len(val_ds), len(test_ds)) <= 0:
        raise RuntimeError("One or more M4 datasets are empty.")

    train_loader, _, _ = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    if len(batch) != 2:
        raise RuntimeError(
            f"Expected WSTS-only batch (x, y), got {len(batch)} items."
        )

    x, y = batch

    return x, None, y


def load_proposed_batch(args, config):
    """
    Full real-data proposed-model smoke test using ONE known
    WSTS fire and its matching real AEF file.

    This avoids requiring complete AEF coverage merely to run
    an integration smoke test.
    """

    if args.aef_dir is None:
        raise ValueError("--aef-dir is required for proposed mode.")

    if args.year is None or args.fire is None:
        raise ValueError(
            "--year and --fire are required for proposed mode."
        )

    expected_aef = (
        Path(args.aef_dir)
        / f"{args.year}__{args.fire}.tif"
    )

    if not expected_aef.is_file():
        raise FileNotFoundError(
            f"AEF file not found: {expected_aef}"
        )

    fold_cfg = config["m4_folds"][str(args.fold)]

    print("Building real single-fire WSTS + AEF dataset...")
    print(f"Fire:         {args.year}/{args.fire}")
    print(f"AEF:          {expected_aef}")
    print(f"Stats years:  {fold_cfg['train_years']}")

    dataset = WSTSAEFDataset(
        data_dir=args.data_dir,
        aef_dir=args.aef_dir,
        use_aef=True,
        included_fire_years=[args.year],
        n_leading_observations=1,
        crop_side_length=config["crop_side_length"],
        load_from_hdf5=True,
        remove_duplicate_features=False,
        stats_years=fold_cfg["train_years"],
        is_train=True,
    )

    index = find_first_index_for_fire(
        dataset,
        args.year,
        args.fire,
    )

    x, e, y = dataset[index]

    # Add batch dimension.
    x = x.unsqueeze(0)
    e = e.unsqueeze(0)
    y = y.unsqueeze(0)

    return x, e, y


def validate_batch(x, e, y, expected_crop):
    print()
    print("Batch:")
    print("WSTS:   ", tuple(x.shape))

    if e is not None:
        print("AEF:    ", tuple(e.shape))

    print("Target: ", tuple(y.shape))

    if x.ndim != 4:
        raise RuntimeError(f"Expected x [B,C,H,W], got {x.shape}")

    if x.shape[1] != 40:
        raise RuntimeError(
            f"Expected 40 WSTS channels, got {x.shape[1]}"
        )

    if tuple(x.shape[-2:]) != (expected_crop, expected_crop):
        raise RuntimeError(
            f"Expected WSTS crop {expected_crop}x{expected_crop}, "
            f"got {tuple(x.shape[-2:])}"
        )

    if not torch.isfinite(x).all():
        raise RuntimeError("WSTS batch contains NaN or Inf.")

    if not torch.isfinite(y).all():
        raise RuntimeError("Target contains NaN or Inf.")

    unique_targets = set(torch.unique(y).cpu().tolist())

    if not unique_targets.issubset({0, 1}):
        raise RuntimeError(
            f"Unexpected target values: {unique_targets}"
        )

    if e is not None:
        if e.ndim != 4:
            raise RuntimeError(
                f"Expected AEF [B,C,H,W], got {e.shape}"
            )

        if e.shape[1] != 64:
            raise RuntimeError(
                f"Expected 64 AEF channels, got {e.shape[1]}"
            )

        if e.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                "AEF/WSTS spatial shape mismatch: "
                f"{tuple(e.shape[-2:])} vs {tuple(x.shape[-2:])}"
            )

        if not torch.isfinite(e).all():
            raise RuntimeError("AEF batch contains NaN or Inf.")


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])

    device = resolve_device(args.device)

    print("=" * 64)
    print("M4 RQ1 REAL-DATA SMOKE TEST")
    print("=" * 64)
    print("Mode:          ", args.mode)
    print("Device:        ", device)
    print("Seed:          ", config["seed"])
    print("Crop:          ", config["crop_side_length"])
    print("Learning rate: ", config["training"]["learning_rate"])
    print("BCE weight:    ", config["training"]["bce_weight"])
    print("Threshold:     ", config["threshold"])
    print()

    if args.mode == "wsts-control":
        x, e, y = load_wsts_control_batch(args, config)
        model_input = x
        expected_channels = 40

    else:
        x, e, y = load_proposed_batch(args, config)
        model_input = torch.cat([x, e], dim=1)
        expected_channels = 104

    validate_batch(
        x,
        e,
        y,
        config["crop_side_length"],
    )

    print("Model input:", tuple(model_input.shape))

    if model_input.shape[1] != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} input channels, "
            f"got {model_input.shape[1]}"
        )

    params = inspect.signature(AttentionUNet).parameters

    if "use_se" not in params:
        raise RuntimeError(
            "AttentionUNet does not support use_se. "
            "Pull the latest repository version."
        )

    # Intentionally small for smoke testing only.
    model = AttentionUNet(
        n_in_channels=expected_channels,
        base_channels=args.base_channels,
        use_se=config["model"]["use_se"],
    ).to(device)

    loss_fn = HybridBCEDiceLoss(
        bce_weight=config["training"]["bce_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    model_input = model_input.to(device)
    y = y.to(device)

    model.train()

    optimizer.zero_grad(set_to_none=True)

    prediction = model(model_input)

    print("Prediction: ", tuple(prediction.shape))

    expected_prediction_shape = (
        model_input.shape[0],
        1,
        model_input.shape[-2],
        model_input.shape[-1],
    )

    if tuple(prediction.shape) != expected_prediction_shape:
        raise RuntimeError(
            "Unexpected prediction shape: "
            f"{tuple(prediction.shape)}; "
            f"expected {expected_prediction_shape}"
        )

    if not torch.isfinite(prediction).all():
        raise RuntimeError(
            "Prediction contains NaN or Inf."
        )

    loss = loss_fn(prediction, y)

    print("Loss:       ", float(loss.detach().cpu()))

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or Inf.")

    loss.backward()

    saw_gradient = False

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        saw_gradient = True

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                f"Non-finite gradient in parameter: {name}"
            )

    if not saw_gradient:
        raise RuntimeError(
            "Backward completed but no gradients were produced."
        )

    optimizer.step()

    print()
    print("Forward:        OK")
    print("Finite loss:    OK")
    print("Backward:       OK")
    print("Finite grads:   OK")
    print("Optimizer step: OK")
    print()
    print("REAL-DATA SMOKE TEST PASSED")


if __name__ == "__main__":
    main()