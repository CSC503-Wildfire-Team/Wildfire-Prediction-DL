"""
M4 RQ1 formal 3-fold development runner for Attention U-Net + SE.

Default: 40ch WSTS-only control.
Use --proposed for the 104ch WSTS+AEF model.

This script uses configs/m4_rq1.json and the fixed selected-fire manifest.
It trains/validates the three development folds and saves:
- best checkpoint per fold
- per-fold JSON with full epoch history
- one summary JSON with mean/std validation metrics

Important: it does NOT evaluate the held-out 2021 final test set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wsts_aef_dataset import build_datasets, build_loaders  # noqa: E402
from src.attention_unet.fit import fit  # noqa: E402
from src.attention_unet.losses import HybridBCEDiceLoss  # noqa: E402
from src.attention_unet.model import AttentionUNet  # noqa: E402
from src.attention_unet.reproducibility import set_seed  # noqa: E402
from src.attention_unet.res18_unet import Res18UNetBaseline  # noqa: E402


METRICS = ("loss", "iou", "dice", "precision", "recall", "f1")


def parse_args():
    p = argparse.ArgumentParser(description="M4 RQ1 formal development training.")
    p.add_argument("--data-dir", required=True,
                   help="HDF5 root containing 2018/, 2019/, 2020/, 2021/.")
    p.add_argument("--aef-dir", default=None,
                   help="Directory containing <year>__<fire>.tif; required with --proposed.")
    p.add_argument("--proposed", action="store_true",
                   help="Train 104ch WSTS+AEF instead of the 40ch control.")
    p.add_argument("--folds", nargs="+", type=int, choices=(0, 1, 2),
                   default=(0, 1, 2), help="M4 folds to run (default: 0 1 2).")
    p.add_argument("--res18-baseline", action="store_true",
               help="Train WSTS paper 40ch Res18 U-Net baseline instead of Attention U-Net.")
    p.add_argument("--config", default=str(ROOT / "configs" / "m4_rq1.json"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def default_out_dir():
    kaggle = Path("/kaggle/working")
    return kaggle / "m4_runs" if kaggle.is_dir() else Path("m4_runs")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def metrics_dict(result):
    return {name: float(getattr(result, name)) for name in METRICS}


def validation_summary(fold_reports):
    summary = {}
    for metric in METRICS:
        values = [r["best_val_metrics"][metric] for r in fold_reports]
        summary[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "values": values,
        }
    return summary


def validate_config(cfg):
    tr = cfg["training"]
    if tr["optimizer"] != "AdamW":
        raise ValueError("This runner currently expects optimizer='AdamW'.")
    if tr["loss"] != "bce_dice":
        raise ValueError("This runner currently expects loss='bce_dice'.")
    if cfg["evaluation"]["selection_metric"].lower() != "iou":
        raise ValueError("fit.py currently selects the best checkpoint by validation IoU.")


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg)

    tr_cfg = cfg["training"]
    batch_size = args.batch_size if args.batch_size is not None else tr_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else tr_cfg["num_workers"]
    max_epochs = args.max_epochs if args.max_epochs is not None else tr_cfg["max_epochs"]
    patience = args.patience if args.patience is not None else tr_cfg["early_stopping_patience"]

    if batch_size is None or batch_size < 1:
        raise SystemExit("batch_size must be a positive integer.")
    if num_workers is None or num_workers < 0:
        raise SystemExit("num_workers must be >= 0.")
    if max_epochs is None or max_epochs < 1:
        raise SystemExit("max_epochs must be a positive integer.")
    if patience is not None and patience < 1:
        raise SystemExit("patience must be >= 1 or None.")

    if args.proposed and args.res18_baseline:
        raise SystemExit("Choose either --proposed or --res18-baseline, not both.")

    use_aef = args.proposed
    if use_aef and not args.aef_dir:
        raise SystemExit("--proposed requires --aef-dir.")

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Enable a Kaggle GPU accelerator and restart.")

    device = torch.device("cuda")
    if args.res18_baseline:
        mode = "res18_baseline_40ch"
    elif use_aef:
        mode = "proposed_104ch"
    else:
        mode = "wsts_control_40ch"

    manifest = ROOT / cfg["dataset"]["selected_fires_manifest"]
    if not manifest.is_file():
        raise SystemExit(f"Selected-fire manifest not found: {manifest}")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in = cfg["model"]["wsts_channels"] + (
        cfg["model"]["aef_channels"] if use_aef else 0
    )

    started_at = datetime.now(timezone.utc).isoformat()
    all_reports = []
    total_start = time.perf_counter()

    print("=" * 72)
    print(f"M4 FORMAL TRAINING  mode={mode}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"folds={list(args.folds)}  batch_size={batch_size}  workers={num_workers}")
    print(f"max_epochs={max_epochs}  patience={patience}  seed={cfg['seed']}")
    print(f"manifest={manifest.name}  output={out_dir}")
    print("2021 final test will NOT be evaluated.")
    print("=" * 72)

    for m4_fold in args.folds:
        fold_cfg = cfg["m4_folds"][str(m4_fold)]
        set_seed(cfg["seed"])

        print()
        print("-" * 72)
        print(
            f"FOLD {m4_fold}: train={fold_cfg['train_years']} "
            f"val={fold_cfg['val_year']} "
            f"test={fold_cfg['test_year']} (held out) "
            f"loader_fold={fold_cfg['loader_fold']}"
        )
        print("-" * 72)

        t_build = time.perf_counter()
        train_ds, val_ds, test_ds = build_datasets(
            data_dir=args.data_dir,
            aef_dir=args.aef_dir or str(ROOT / "_unused_aef"),
            fold=fold_cfg["loader_fold"],
            crop_side_length=cfg["crop_side_length"],
            load_from_hdf5=True,
            use_aef=use_aef,
            selected_fires_manifest=str(manifest),
        )
        train_loader, val_loader, _ = build_loaders(
            train_ds, val_ds, test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        print(
            f"datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
            f"(built in {time.perf_counter() - t_build:.1f}s)"
        )

        if args.res18_baseline:
            model = Res18UNetBaseline(n_in_channels=n_in).to(device)
        else:
            model = AttentionUNet(
                n_in_channels=n_in,
                base_channels=cfg["model"]["base_channels"],
                use_se=cfg["model"]["use_se"],
                use_spatial_attention=cfg["model"].get(
                    "use_spatial_attention",
                    True,
                ),
            ).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        loss_fn = HybridBCEDiceLoss(bce_weight=tr_cfg["bce_weight"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tr_cfg["learning_rate"],
        )

        checkpoint = out_dir / f"{mode}_fold{m4_fold}_best.pt"
        print(f"model: {n_in} channels, {n_params / 1e6:.2f}M params")
        print(f"training; best checkpoint -> {checkpoint}")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_fold = time.perf_counter()

        result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epochs=max_epochs,
            use_aef=use_aef,
            threshold=cfg["threshold"],
            checkpoint_path=checkpoint,
            early_stopping_patience=patience,
        )

        torch.cuda.synchronize()
        fold_seconds = time.perf_counter() - t_fold

        best = next(r for r in result.history if r.epoch == result.best_epoch)
        history = [
            {
                "epoch": r.epoch,
                "train": metrics_dict(r.train),
                "val": metrics_dict(r.val),
            }
            for r in result.history
        ]

        report = {
            "mode": mode,
            "m4_fold": m4_fold,
            "loader_fold": fold_cfg["loader_fold"],
            "seed": cfg["seed"],
            "train_years": fold_cfg["train_years"],
            "val_year": fold_cfg["val_year"],
            "test_year": fold_cfg["test_year"],
            "test_evaluated": False,
            "samples": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
            },
            "input_channels": n_in,
            "params": n_params,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "learning_rate": tr_cfg["learning_rate"],
            "loss": tr_cfg["loss"],
            "bce_weight": tr_cfg["bce_weight"],
            "threshold": cfg["threshold"],
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "epochs_ran": len(result.history),
            "best_epoch": result.best_epoch,
            "best_train_metrics": metrics_dict(best.train),
            "best_val_metrics": metrics_dict(best.val),
            "runtime_minutes": fold_seconds / 60.0,
            "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "checkpoint_path": str(checkpoint),
            "history": history,
        }

        fold_json = out_dir / f"{mode}_fold{m4_fold}.json"
        write_json(fold_json, report)
        all_reports.append(report)

        print(
            f"done in {fold_seconds / 60:.1f} min; "
            f"epochs={len(result.history)} best_epoch={result.best_epoch}"
        )
        print(
            f"best val: loss={best.val.loss:.4f} IoU={best.val.iou:.4f} "
            f"Dice={best.val.dice:.4f} Precision={best.val.precision:.4f} "
            f"Recall={best.val.recall:.4f} F1={best.val.f1:.4f}"
        )

        # Save/update summary after every fold so completed results survive
        # if the Kaggle session stops before all requested folds finish.
        partial = {
            "mode": mode,
            "started_at_utc": started_at,
            "config_path": str(cfg_path),
            "manifest": str(manifest),
            "folds_requested": list(args.folds),
            "folds_completed": [r["m4_fold"] for r in all_reports],
            "test_evaluated": False,
            "fold_results": [
                {
                    "m4_fold": r["m4_fold"],
                    "best_epoch": r["best_epoch"],
                    "epochs_ran": r["epochs_ran"],
                    "best_val_metrics": r["best_val_metrics"],
                    "runtime_minutes": r["runtime_minutes"],
                    "checkpoint_path": r["checkpoint_path"],
                }
                for r in all_reports
            ],
            "validation_summary": validation_summary(all_reports),
        }
        write_json(out_dir / f"{mode}_summary.json", partial)

    total_seconds = time.perf_counter() - total_start
    summary = {
        "mode": mode,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "manifest": str(manifest),
        "folds_requested": list(args.folds),
        "folds_completed": [r["m4_fold"] for r in all_reports],
        "test_evaluated": False,
        "total_runtime_minutes": total_seconds / 60.0,
        "fold_results": [
            {
                "m4_fold": r["m4_fold"],
                "loader_fold": r["loader_fold"],
                "train_years": r["train_years"],
                "val_year": r["val_year"],
                "samples": r["samples"],
                "best_epoch": r["best_epoch"],
                "epochs_ran": r["epochs_ran"],
                "best_train_metrics": r["best_train_metrics"],
                "best_val_metrics": r["best_val_metrics"],
                "runtime_minutes": r["runtime_minutes"],
                "checkpoint_path": r["checkpoint_path"],
            }
            for r in all_reports
        ],
        "validation_summary": validation_summary(all_reports),
    }

    summary_path = out_dir / f"{mode}_summary.json"
    write_json(summary_path, summary)

    print()
    print("=" * 72)
    print("ALL REQUESTED FOLDS COMPLETE")
    print(f"total runtime: {total_seconds / 60:.1f} min")
    print("development validation mean ± std:")
    for metric, values in summary["validation_summary"].items():
        print(f"  {metric:9s}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"summary: {summary_path}")
    print("2021 final test remains untouched.")
    print("=" * 72)


if __name__ == "__main__":
    main()
