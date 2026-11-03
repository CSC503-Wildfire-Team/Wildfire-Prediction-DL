"""
Check the Earth Engine medoid against the NumPy reference on a small window (default 3x3 WSTS pixels)
of one fire. Run this BEFORE exporting all fires.

    python scripts/03_validate_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --year 2020
"""
import argparse

import ee
import numpy as np
import pandas as pd

from aef_utils import BANDS, N_SUB, block_medoids, build_medoid_image, fine_grid, region_geometry


def sample(img, region, shape_expected):
    props = img.select(BANDS).sampleRectangle(region=region, defaultValue=-9999).getInfo()["properties"]
    arr = np.stack([np.array(props[b], dtype=np.float32) for b in BANDS])
    arr[arr == -9999] = np.nan
    if arr.shape[1:] != shape_expected:
        raise SystemExit(f"Unexpected sample shape {arr.shape[1:]} (expected {shape_expected}); "
                         "check the grid / inset logic before trusting any export.")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--index", default="data/events_index.csv")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--fire", help="default: first fire of that year")
    ap.add_argument("--cells", type=int, default=3)
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    df = pd.read_csv(args.index)
    df = df[df.year == args.year]
    r = (df[df.fire == args.fire] if args.fire else df).iloc[0]

    k = args.cells
    i0, j0 = int(r.width) // 2 - k // 2, int(r.height) // 2 - k // 2          # window near the centre
    a, b, c, d, e, f = r.a, r.b, r.c, r.d, r.e, r.f
    sub = {"crs": r.crs, "transform": [a, b, c + a * i0, d, e, f + e * j0], "width": k, "height": k}

    fine, coarse = build_medoid_image(ee, sub, int(r.year) - 1)
    fine_arr = sample(fine, region_geometry(ee, fine_grid(sub), 0.5), (k * N_SUB, k * N_SUB))
    gee_arr = sample(coarse, region_geometry(ee, sub, 0.5), (k, k))
    ref_arr = block_medoids(fine_arr)

    cos = (gee_arr * ref_arr).sum(0)                                          # both unit vectors
    valid = np.isfinite(cos)
    print(f"{r.year}/{r.fire}: window {k}x{k} cells, {valid.sum()} comparable")
    print("cosine(GEE medoid, NumPy medoid): min %.6f  mean %.6f" % (np.nanmin(cos), np.nanmean(cos)))
    print("PASS" if np.nanmin(cos) > 0.9999 else "FAIL: investigate before exporting everything")


if __name__ == "__main__":
    main()
