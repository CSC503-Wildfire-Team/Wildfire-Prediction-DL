"""
Export one 64-band AlphaEarth cosine-medoid GeoTIFF per WSTS fire, on that fire's exact 375 m grid.
Uses the AEF layer of the year BEFORE the fire's year (no leakage). Files land in Google Drive as
<year>__<fire>.tif -- download them into data/aef/ for the dataset class.

    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --limit 1 --dry-run
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --limit 3
    python scripts/02_export_aef_medoids.py --project YOUR_PROJECT --index data/events_index.csv --years 2018 2019
"""
import argparse
import time

import ee
import pandas as pd

from aef_utils import build_medoid_image, export_to_drive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine Cloud project ID")
    ap.add_argument("--index", default="data/events_index.csv")
    ap.add_argument("--folder", default="wsts_aef", help="Google Drive folder for the exports")
    ap.add_argument("--years", type=int, nargs="*", help="only these fire years")
    ap.add_argument("--fire", help="only this fire name")
    ap.add_argument("--limit", type=int, help="export at most N fires (use 1-3 to test first)")
    ap.add_argument("--dry-run", action="store_true", help="build the image graph but do not start tasks")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between task submissions")
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
        _, coarse = build_medoid_image(ee, grid, aef_year)
        if args.dry_run:
            print(f"[dry-run] {name}: AEF {aef_year}, grid {r.height}x{r.width}, bands {coarse.bandNames().size().getInfo()}")
            continue
        task = export_to_drive(ee, coarse, grid, name, args.folder)
        print(f"started {name} (AEF {aef_year}) task {task.id}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
