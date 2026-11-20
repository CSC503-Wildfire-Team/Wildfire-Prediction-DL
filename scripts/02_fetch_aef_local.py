from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio

from aef_utils import (
    BANDS,
    build_medoid_image,
    region_geometry,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Directly download one AEF cosine-medoid image from "
            "Earth Engine to a local GeoTIFF, bypassing Export.toDrive()."
        )
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Earth Engine Google Cloud project ID.",
    )

    parser.add_argument(
        "--index",
        default="data/events_index.csv",
        help="WSTS event index CSV.",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="WSTS fire year.",
    )

    parser.add_argument(
        "--fire",
        required=True,
        help="Fire name, e.g. fire_21458798.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for the locally downloaded AEF GeoTIFF.",
    )

    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Optional validated Drive-exported TIFF. "
            "If provided, compare the local result against it."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Maximum geedim download threads (default: 4).",
    )

    return parser.parse_args()


def load_fire_row(index_path, year, fire):
    df = pd.read_csv(index_path)

    selected = df[
        (df["year"] == year)
        & (df["fire"] == fire)
    ]

    if len(selected) == 0:
        raise RuntimeError(
            f"{year}/{fire} not found in {index_path}"
        )

    if len(selected) > 1:
        raise RuntimeError(
            f"More than one row found for {year}/{fire}"
        )

    row = selected.iloc[0]

    consistent = row["consistent"]

    if isinstance(consistent, str):
        consistent = consistent.lower() == "true"

    if not bool(consistent):
        raise RuntimeError(
            f"{year}/{fire} does not have a consistent WSTS grid."
        )

    return row


def grid_from_row(row):
    return {
        "crs": str(row["crs"]),
        "transform": [
            float(row["a"]),
            float(row["b"]),
            float(row["c"]),
            float(row["d"]),
            float(row["e"]),
            float(row["f"]),
        ],
        "width": int(row["width"]),
        "height": int(row["height"]),
    }


def print_grid(grid):
    print("Target WSTS grid:")
    print("  CRS:       ", grid["crs"])
    print(
        "  dimensions:",
        f"{grid['width']} x {grid['height']}",
        "(width x height)",
    )
    print("  transform: ", grid["transform"])


def validate_output(path, grid):
    print()
    print("=" * 70)
    print("VALIDATING LOCAL OUTPUT")
    print("=" * 70)

    with rasterio.open(path) as ds:
        print("File:       ", path)
        print("Bands:      ", ds.count)
        print("Width:      ", ds.width)
        print("Height:     ", ds.height)
        print("CRS:        ", ds.crs)
        print("Transform:  ", ds.transform)

        if ds.count != 64:
            raise RuntimeError(
                f"Expected 64 bands, got {ds.count}"
            )

        if ds.width != grid["width"]:
            raise RuntimeError(
                f"Width mismatch: {ds.width} != {grid['width']}"
            )

        if ds.height != grid["height"]:
            raise RuntimeError(
                f"Height mismatch: {ds.height} != {grid['height']}"
            )

        if ds.crs is None:
            raise RuntimeError("Downloaded TIFF has no CRS.")

        expected_transform = np.asarray(
            grid["transform"],
            dtype=np.float64,
        )

        actual_transform = np.asarray(
            tuple(ds.transform)[:6],
            dtype=np.float64,
        )

        if not np.allclose(
            actual_transform,
            expected_transform,
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Transform mismatch.\n"
                f"Expected: {expected_transform}\n"
                f"Actual:   {actual_transform}"
            )

        data = np.ma.filled(
            ds.read(masked=True),
            np.nan,
        ).astype(np.float32)

    finite = np.isfinite(data)

    print(
        "Finite values:",
        f"{finite.sum()} / {data.size}",
    )

    if finite.sum() == 0:
        raise RuntimeError(
            "Downloaded raster contains no finite data."
        )

    vals = data[finite]

    print(
        "Value range:  ",
        float(vals.min()),
        "to",
        float(vals.max()),
    )

    print()
    print("LOCAL GRID VALIDATION: PASS")

    return data


def compare_reference(local_path, reference_path):
    print()
    print("=" * 70)
    print("COMPARING AGAINST VALIDATED DRIVE EXPORT")
    print("=" * 70)

    reference_path = Path(reference_path)

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference file not found: {reference_path}"
        )

    with rasterio.open(local_path) as local_ds:
        local = np.ma.filled(
            local_ds.read(masked=True),
            np.nan,
        ).astype(np.float32)

        local_shape = (
            local_ds.count,
            local_ds.height,
            local_ds.width,
        )

        local_crs = local_ds.crs

        local_transform = np.asarray(
            tuple(local_ds.transform)[:6],
            dtype=np.float64,
        )

    with rasterio.open(reference_path) as ref_ds:
        reference = np.ma.filled(
            ref_ds.read(masked=True),
            np.nan,
        ).astype(np.float32)

        ref_shape = (
            ref_ds.count,
            ref_ds.height,
            ref_ds.width,
        )

        ref_crs = ref_ds.crs

        ref_transform = np.asarray(
            tuple(ref_ds.transform)[:6],
            dtype=np.float64,
        )

    print("Local shape:     ", local_shape)
    print("Reference shape: ", ref_shape)

    if local_shape != ref_shape:
        raise RuntimeError(
            "Local/reference raster shapes differ."
        )

    print("Local CRS:       ", local_crs)
    print("Reference CRS:   ", ref_crs)

    if local_crs != ref_crs:
        raise RuntimeError(
            "Local/reference CRS differ."
        )

    if not np.allclose(
        local_transform,
        ref_transform,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Local/reference transforms differ."
        )

    both_finite = (
        np.isfinite(local)
        & np.isfinite(reference)
    )

    n_comparable = int(both_finite.sum())

    print("Comparable values:", n_comparable)

    if n_comparable == 0:
        raise RuntimeError(
            "No comparable finite pixels."
        )

    diff = np.abs(
        local[both_finite]
        - reference[both_finite]
    )

    print(
        "Mean abs diff:   ",
        float(diff.mean()),
    )

    print(
        "Max abs diff:    ",
        float(diff.max()),
    )

    print(
        "99th pct diff:   ",
        float(np.percentile(diff, 99)),
    )

    close = np.allclose(
        local[both_finite],
        reference[both_finite],
        rtol=1e-5,
        atol=1e-6,
    )

    # Also compare each complete 64-D embedding vector
    # by cosine similarity.
    valid_pixels = (
        np.isfinite(local).all(axis=0)
        & np.isfinite(reference).all(axis=0)
    )

    n_pixels = int(valid_pixels.sum())

    print("Comparable 64-D pixels:", n_pixels)

    if n_pixels > 0:
        a = local[:, valid_pixels].T.astype(np.float64)
        b = reference[:, valid_pixels].T.astype(np.float64)

        denom = (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
        )

        valid_norm = denom > 0

        cosine = (
            np.sum(
                a[valid_norm] * b[valid_norm],
                axis=1,
            )
            / denom[valid_norm]
        )

        print(
            "Cosine similarity:",
            "min =",
            float(cosine.min()),
            "mean =",
            float(cosine.mean()),
        )

    if close:
        print()
        print("REFERENCE NUMERICAL COMPARISON: PASS")
    else:
        print()
        print(
            "WARNING: outputs are not identical within "
            "the configured tolerance."
        )


def main():
    args = parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    print("=" * 70)
    print("AEF DIRECT LOCAL DOWNLOAD TEST")
    print("=" * 70)

    print("Initializing Earth Engine...")

    ee.Initialize(
        project=args.project,
    )

    row = load_fire_row(
        args.index,
        args.year,
        args.fire,
    )

    grid = grid_from_row(row)

    print()
    print(
        "Fire:",
        f"{args.year}/{args.fire}",
    )

    aef_year = args.year - 1

    print("AEF year:", aef_year)

    print_grid(grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / f"{args.year}__{args.fire}.tif"
    )

    print()
    print("Building Earth Engine medoid graph...")

    graph_start = time.perf_counter()

    _, coarse = build_medoid_image(
        ee,
        grid,
        aef_year,
    )

    graph_elapsed = (
        time.perf_counter()
        - graph_start
    )

    print(
        f"Graph built in {graph_elapsed:.2f} sec"
    )

    print()
    print("Downloading directly to local GeoTIFF...")
    print("Output:", out_path)
    print("Threads:", args.threads)

    download_start = time.perf_counter()

    geemap.download_ee_image(
            image=coarse.toFloat(),
            filename=str(out_path),
            crs=grid["crs"],
            crs_transform=grid["transform"],
            shape=(grid["height"], grid["width"]),
            resampling="near",
            dtype="float32",
            overwrite=True,
    )

    download_elapsed = (
        time.perf_counter()
        - download_start
    )

    total_elapsed = (
        time.perf_counter()
        - graph_start
    )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Direct download time: "
        f"{download_elapsed:.1f} sec "
        f"({download_elapsed / 60:.2f} min)"
    )

    print(
        f"Total elapsed:        "
        f"{total_elapsed:.1f} sec "
        f"({total_elapsed / 60:.2f} min)"
    )

    validate_output(
        out_path,
        grid,
    )

    if args.reference:
        compare_reference(
            out_path,
            args.reference,
        )

    print()
    print("=" * 70)
    print("DIRECT LOCAL AEF TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()