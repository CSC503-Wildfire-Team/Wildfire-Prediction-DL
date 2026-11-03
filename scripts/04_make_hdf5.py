"""
Convert WSTS GeoTIFFs to the per-fire HDF5 files the WSTS dataset class reads (much faster training).
Same output layout as the authors' CreateHDF5Dataset.py, but written with pathlib so it works on Windows
(the original splits paths on '/'). Keep the GeoTIFFs: the HDF5 files only store the centre lon/lat,
not the full CRS/transform needed to align AlphaEarth.

    python scripts/04_make_hdf5.py --data-dir data/WildfireSpreadTS --target-dir data/WildfireSpreadTS_hdf5
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--years", type=int, nargs="*", default=[2018, 2019, 2020, 2021])
    args = ap.parse_args()

    fires = [(y, f) for y in args.years for f in sorted(p for p in (Path(args.data_dir) / str(y)).iterdir() if p.is_dir())]
    for year, fdir in tqdm(fires, desc="fires"):
        out = Path(args.target_dir) / str(year) / f"{fdir.name}.hdf5"
        if out.is_file():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tifs = sorted(fdir.glob("*.tif"))
        if not tifs:
            continue
        imgs, lnglat = [], None
        for p in tifs:
            with rasterio.open(p) as ds:
                imgs.append(ds.read())
                lnglat = lnglat or ds.lnglat()
        x = np.stack(imgs, axis=0)                                   # (T, 23, H, W)
        x[:, -1] = np.nan_to_num(x[:, -1], nan=0)                    # active fire: NaN = no detection
        x[:, -1] = np.floor_divide(x[:, -1], 100)                    # detection time hhmm -> hh
        with h5py.File(out, "w") as f:
            d = f.create_dataset("data", x.shape, data=x)
            d.attrs["year"], d.attrs["fire_name"] = year, fdir.name
            d.attrs["img_dates"] = [p.stem.split("_")[0] for p in tifs]
            d.attrs["lnglat"] = lnglat


if __name__ == "__main__":
    main()
