"""
Download AEF embeddings with Xarray, compute one 64-band cosine-medoid GeoTIFF per WSTS fire on
that fire's exact 375 m grid, and save it locally. Uses the AEF layer of the year BEFORE the fire's
year (no leakage).

    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --limit 1 --dry-run
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --limit 3
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --years 2018 2019
"""
import argparse
from pathlib import Path

import ee
import pandas as pd

from aef_utils import block_medoids, download_fine_embeddings, write_medoid_geotiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine Cloud project ID")
    ap.add_argument("--index", default="data/events_index.csv")
    ap.add_argument("--output-dir", default="data/aef", help="directory for medoid GeoTIFFs")
    ap.add_argument("--years", type=int, nargs="*", help="only these fire years")
    ap.add_argument("--fire", help="only this fire name")
    ap.add_argument("--limit", type=int, help="process at most N fires (use 1-3 to test first)")
    ap.add_argument("--dry-run", action="store_true", help="list selected fires without downloading")
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    df = pd.read_csv(args.index)
    if args.years:
        df = df[df.year.isin(args.years)]
    if args.fire:
        df = df[df.fire == args.fire]
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} fires selected")

    for r in df.itertuples():
        if not r.consistent:
            print(f"skip {r.year}/{r.fire}: days do not share one grid")
            continue
        grid = {"crs": r.crs, "transform": [r.a, r.b, r.c, r.d, r.e, r.f],
                "width": int(r.width), "height": int(r.height)}
        name = f"{r.year}__{r.fire}"
        aef_year = int(r.year) - 1
        if args.dry_run:
            print(f"[dry-run] {name}: AEF {aef_year}, grid {r.height}x{r.width}, "
                  f"output {Path(args.output_dir) / (name + '.tif')}")
            continue
        fine = download_fine_embeddings(ee, grid, aef_year)
        medoids = block_medoids(fine)
        output_path = Path(args.output_dir) / f"{name}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_medoid_geotiff(output_path, medoids, grid)
        print(f"saved {output_path} (AEF {aef_year})")


if __name__ == "__main__":
    main()
