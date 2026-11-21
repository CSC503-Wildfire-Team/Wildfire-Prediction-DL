"""
M4 RQ1 first GPU test: one epoch of the 40ch WSTS-only control (or 104ch proposed) on one fold.
Reports epoch time and peak GPU memory so the team can lock batch size / epochs.

Uses the shared config (configs/m4_rq1.json) and the shared fire manifest, so the data,
model and optimizer settings match the real runs. Nothing here is a benchmark result.

Kaggle (after `git clone` of the repo and attaching the shared HDF5 dataset):
    python scripts/m4_gpu_timing.py --data-dir /kaggle/input/<dataset>/<hdf5 root> --fold 0

Quick estimate first (stops after 50 train + 20 val batches, extrapolates the full epoch):
    python scripts/m4_gpu_timing.py --data-dir ... --fold 0 --max-train-batches 50 --max-val-batches 20

--data-dir must be the folder that contains the year folders (2018/, 2019/, ...).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wsts_aef_dataset import build_datasets, build_loaders  # noqa: E402
from src.attention_unet.engine import run_epoch  # noqa: E402
from src.attention_unet.losses import HybridBCEDiceLoss  # noqa: E402
from src.attention_unet.model import AttentionUNet  # noqa: E402
from src.attention_unet.reproducibility import set_seed  # noqa: E402


class Limited:
    """Iterate over at most n batches of a DataLoader (n=None means all)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        return iter(self.loader) if self.n is None else itertools.islice(self.loader, self.n)


def parse_args():
    p = argparse.ArgumentParser(description="One-epoch GPU timing/memory test for M4 RQ1.")
    p.add_argument("--data-dir", required=True, help="HDF5 root containing 2018/, 2019/, ...")
    p.add_argument("--aef-dir", default=None, help="only needed with --proposed")
    p.add_argument("--proposed", action="store_true", help="104ch WSTS+AEF instead of 40ch control")
    p.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--config", default=str(ROOT / "configs" / "m4_rq1.json"))
    p.add_argument("--batch-size", type=int, default=None, help="default: value in the config")
    p.add_argument("--num-workers", type=int, default=None, help="default: value in the config")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--out", default=None, help="JSON report path (default: /kaggle/working or ./)")
    return p.parse_args()


def gb(n_bytes):
    return round(n_bytes / 1024 ** 3, 2)


def timed_epoch(name, model, loader, n_total, limit, loss_fn, device, use_aef, threshold, optimizer):
    n_run = n_total if limit is None else min(limit, n_total)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = run_epoch(model=model, loader=Limited(loader, limit), loss_fn=loss_fn, device=device,
                       use_aef=use_aef, threshold=threshold, optimizer=optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    full_est = secs / max(n_run, 1) * n_total
    print(f"{name}: {n_run}/{n_total} batches in {secs:.1f}s "
          f"({secs / max(n_run, 1):.3f}s/batch) -> full epoch ~ {full_est / 60:.1f} min")
    print(f"   loss {result.loss:.4f}  IoU {result.iou:.4f}  Dice {result.dice:.4f}")
    return {"batches_run": n_run, "batches_total": n_total, "seconds": round(secs, 1),
            "sec_per_batch": round(secs / max(n_run, 1), 4), "full_epoch_min_est": round(full_est / 60, 1),
            "loss": result.loss, "iou": result.iou, "dice": result.dice}


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tr_cfg, fold_cfg = cfg["training"], cfg["m4_folds"][str(args.fold)]
    batch_size = args.batch_size or tr_cfg["batch_size"]
    num_workers = tr_cfg["num_workers"] if args.num_workers is None else args.num_workers
    use_aef = args.proposed
    if use_aef and not args.aef_dir:
        raise SystemExit("--proposed needs --aef-dir")
    manifest = ROOT / cfg["dataset"]["selected_fires_manifest"]

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. On Kaggle: Settings > Accelerator > GPU, then restart.")
    device = torch.device("cuda")
    set_seed(cfg["seed"])

    print("=" * 64)
    print(f"M4 GPU TEST  mode={'104ch proposed' if use_aef else '40ch WSTS control'}  fold={args.fold}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({gb(torch.cuda.get_device_properties(0).total_memory)} GB)")
    print(f"batch_size={batch_size}  num_workers={num_workers}  crop={cfg['crop_side_length']}")
    print(f"train {fold_cfg['train_years']}  val {fold_cfg['val_year']}  manifest {manifest.name}")
    print("=" * 64)

    t0 = time.perf_counter()
    train_ds, val_ds, test_ds = build_datasets(
        data_dir=args.data_dir, aef_dir=args.aef_dir or str(ROOT / "_unused_aef"),
        fold=fold_cfg["loader_fold"], crop_side_length=cfg["crop_side_length"],
        load_from_hdf5=True, use_aef=use_aef, selected_fires_manifest=str(manifest))
    train_loader, val_loader, _ = build_loaders(train_ds, val_ds, test_ds,
                                                batch_size=batch_size, num_workers=num_workers)
    print(f"datasets built in {time.perf_counter() - t0:.1f}s: "
          f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)} samples")

    n_in = cfg["model"]["wsts_channels"] + (cfg["model"]["aef_channels"] if use_aef else 0)
    model = AttentionUNet(n_in_channels=n_in, base_channels=cfg["model"]["base_channels"],
                          use_se=cfg["model"]["use_se"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_in} input channels, {n_params / 1e6:.1f}M parameters")
    loss_fn = HybridBCEDiceLoss(bce_weight=tr_cfg["bce_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["learning_rate"])

    torch.cuda.reset_peak_memory_stats()
    train = timed_epoch("train", model, train_loader, len(train_loader), args.max_train_batches,
                        loss_fn, device, use_aef, cfg["threshold"], optimizer)
    val = timed_epoch("val  ", model, val_loader, len(val_loader), args.max_val_batches,
                      loss_fn, device, use_aef, cfg["threshold"], None)

    peak_alloc, peak_res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    epoch_min = train["full_epoch_min_est"] + val["full_epoch_min_est"]
    report = {
        "mode": "proposed_104ch" if use_aef else "wsts_control_40ch", "fold": args.fold,
        "gpu": torch.cuda.get_device_name(0), "gpu_total_gb": gb(total),
        "batch_size": batch_size, "num_workers": num_workers, "params_m": round(n_params / 1e6, 2),
        "samples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "train": train, "val": val,
        "peak_mem_allocated_gb": gb(peak_alloc), "peak_mem_reserved_gb": gb(peak_res),
        "full_epoch_min_est": round(epoch_min, 1),
        "max_epochs": tr_cfg["max_epochs"],
        "worst_case_run_hours_est": round(epoch_min * tr_cfg["max_epochs"] / 60, 1),
    }

    print("-" * 64)
    print(f"peak GPU memory: {gb(peak_alloc)} GB allocated, {gb(peak_res)} GB reserved "
          f"of {gb(total)} GB")
    print(f"train+val epoch ~ {epoch_min:.1f} min -> {tr_cfg['max_epochs']} epochs "
          f"worst case ~ {report['worst_case_run_hours_est']} h per fold")

    out = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    out = out / f"gpu_test_{report['mode']}_fold{args.fold}_bs{batch_size}.json" if out.is_dir() else out
    out.write_text(json.dumps(report, indent=2))
    print(f"report saved to {out}")


if __name__ == "__main__":
    main()
