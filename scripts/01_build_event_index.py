"""
Walk the WildfireSpreadTS GeoTIFFs (<data_dir>/<year>/<fire>/<date>.tif) and write events_index.csv:
one row per fire with CRS, affine transform, size, dates and lon/lat bounds.
This table is the georeferencing source for the AlphaEarth export (02_export_aef_medoids.py).

    python scripts/01_build_event_index.py --data-dir data/WildfireSpreadTS --out data/events_index.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing the year folders (2018, 2019, ...)")
    ap.add_argument("--out", default="data/events_index.csv")
    args = ap.parse_args()

    root = Path(args.data_dir)
    year_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if not year_dirs:
        raise SystemExit(f"No year folders found in {root}. Expected {root}/2018/<fire>/*.tif")

    rows, problems = [], []
    fire_dirs = [(y, f) for y in year_dirs for f in sorted(p for p in y.iterdir() if p.is_dir())]
    for ydir, fdir in tqdm(fire_dirs, desc="fires"):
        tifs = sorted(fdir.glob("*.tif"))
        if not tifs:
            problems.append(f"{ydir.name}/{fdir.name}: no .tif files")
            continue
        with rasterio.open(tifs[0]) as ds:
            ref = dict(h=ds.height, w=ds.width, t=ds.transform, crs=ds.crs, count=ds.count)
            west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            lon, lat = ds.lnglat()
            descriptions = ds.descriptions
        consistent = True
        for p in tifs[1:]:                              # cheap metadata-only check of every day
            with rasterio.open(p) as ds:
                if (ds.height, ds.width, ds.transform, ds.crs, ds.count) != (
                        ref["h"], ref["w"], ref["t"], ref["crs"], ref["count"]):
                    consistent = False
        if not consistent:
            problems.append(f"{ydir.name}/{fdir.name}: days do not share one grid")
        t = ref["t"]
        rows.append(dict(
            year=int(ydir.name), fire=fdir.name, n_days=len(tifs),
            first_date=tifs[0].stem.split("_")[0], last_date=tifs[-1].stem.split("_")[0],
            height=ref["h"], width=ref["w"], n_bands=ref["count"], crs=ref["crs"].to_string(),
            a=t.a, b=t.b, c=t.c, d=t.d, e=t.e, f=t.f,
            lon=lon, lat=lat, west=west, south=south, east=east, north=north,
            consistent=consistent))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} fires to {args.out}")
    print(df.groupby("year").agg(fires=("fire", "count"), days=("n_days", "sum")))
    print("\nGrid size (pixels):\n", df[["height", "width"]].describe().loc[["min", "50%", "max"]])
    print("\nCRS values:", df["crs"].nunique(), "distinct;", df["crs"].value_counts().head(3).to_dict())
    print("Pixel size (a):", sorted(df["a"].round(3).unique())[:5])
    print("Band counts:", df["n_bands"].value_counts().to_dict(), "(expected 23)")
    print("Band descriptions in first file:", descriptions)
    for p in problems:
        print("WARNING:", p)


if __name__ == "__main__":
    main()
