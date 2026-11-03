"""
Smoke test for WSTSAEFDataset on synthetic data (no real download needed). Verifies that the embedding
channels stay pixel-aligned with the fire data through random crops, flips and rotations.

    set WSTS_REPO=C:\\wildfire-project\\WildfireSpreadTS
    python scripts/05_smoke_test_dataset.py
"""
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from wsts_aef_dataset import WSTSAEFDataset

H, W = 150, 170
rng = np.random.default_rng(0)
ramp = (np.arange(H)[:, None] * W + np.arange(W)[None, :]).astype(np.float32)      # unique value per pixel


def write(path, arr):
    prof = dict(driver="GTiff", height=H, width=W, count=arr.shape[0], dtype="float32",
                crs="EPSG:32610", transform=from_origin(500000, 5500000, 375, 375))
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(arr.astype(np.float32))


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for year in (2018, 2019):
        fdir = tmp / "wsts" / str(year) / "fire_1"
        fdir.mkdir(parents=True)
        for day in range(1, 5):
            a = rng.normal(size=(23, H, W)).astype(np.float32)
            a[16] = rng.integers(1, 18, size=(H, W))                 # land cover class
            a[14] = ramp                                             # elevation = position ramp
            a[22] = np.where(rng.random((H, W)) < 0.01, 1230.0, np.nan)   # active fire detection time
            write(fdir / f"{year}-07-{day:02d}.tif", a)
        e = rng.normal(size=(64, H, W)).astype(np.float32)
        e[0] = ramp                                                  # embedding channel 0 = same ramp
        (tmp / "aef").mkdir(exist_ok=True)
        write(tmp / "aef" / f"{year}__fire_1.tif", e)
        # write() above used count=arr.shape[0], so the AEF file has 64 bands

    for train in (True, False):
        ds = WSTSAEFDataset(data_dir=str(tmp / "wsts"), aef_dir=str(tmp / "aef"), included_fire_years=[2018, 2019],
                            n_leading_observations=1, crop_side_length=64, load_from_hdf5=False, is_train=train,
                            remove_duplicate_features=False, stats_years=[2018, 2019])
        x, e, y = ds[0]
        print(f"is_train={train}: x {tuple(x.shape)} e {tuple(e.shape)} y {tuple(y.shape)} "
              f"len={len(ds)} fire_px={int(y.sum())}")
        assert x.shape[0] == 40 and e.shape[0] == 64 and x.shape[1:] == e.shape[1:] == y.shape
        r = np.corrcoef(x[14].flatten().numpy(), e[0].flatten().numpy())[0, 1]      # elevation(std) vs ramp
        print(f"   alignment correlation (elevation vs embedding ramp): {r:.6f}")
        assert r > 0.999999, "embeddings are misaligned with the fire data!"
    print("OK")
