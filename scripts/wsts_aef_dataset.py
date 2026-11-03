"""
WildfireSpreadTS dataset + AlphaEarth cosine-medoid embeddings.

Subclasses the authors' FireSpreadDataset (MIT licence, https://github.com/SebastianGer/WildfireSpreadTS)
so cropping, flips/rotations (with wind-angle correction), standardization and one-hot land cover stay
identical to the benchmark. The 64 embedding channels are carried through the SAME random crop/flip/rotation
as the fire data, are not standardized (they are already unit-length), and are returned separately:

    x : (40, H, W)  WSTS features after the authors' preprocessing (single day)
    e : (64, H, W)  AlphaEarth medoid embedding of the preceding calendar year
    y : (H, W)      next-day active-fire mask (long, 0/1)

Model input = torch.cat([x, e], dim=0) -> 104 channels (use x alone for the no-embedding baseline).

Setup (no second repo needed): copy FireSpreadDataset.py, utils.py and LICENSE from the authors' repo
(src/dataloader/) into scripts/wsts_vendor/ and add an empty __init__.py there. Keep their LICENSE file.
Alternative: clone their repo anywhere and set WSTS_REPO to its path.
"""
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

try:   # preferred: the authors' two files copied into scripts/wsts_vendor/ (see module docstring)
    from wsts_vendor.FireSpreadDataset import FireSpreadDataset
except ImportError:   # fallback: a separate clone of the authors' repo pointed to by WSTS_REPO
    _repo = os.environ.get("WSTS_REPO")
    if not _repo:
        raise ImportError("Copy FireSpreadDataset.py, utils.py, LICENSE and an empty __init__.py into "
                          "scripts/wsts_vendor/, or set WSTS_REPO to a clone of WildfireSpreadTS.")
    sys.path.insert(0, _repo)
    from src.dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402

N_AEF = 64
N_RAW_BASE = 23          # bands in a WSTS GeoTIFF
E_RAW_POS = 22           # embeddings are spliced in just before 'Active fire' (must stay the last band)
# layout after the base class's one-hot encoding of land cover: 16 + 17 + 5 forecast = 38 channels precede E
E_FINAL = slice(38, 38 + N_AEF)
FOLDS = [(2018, 2019, 2020, 2021), (2018, 2019, 2021, 2020), (2018, 2020, 2019, 2021),
         (2018, 2020, 2021, 2019), (2018, 2021, 2019, 2020), (2018, 2021, 2020, 2019),
         (2019, 2020, 2018, 2021), (2019, 2020, 2021, 2018), (2019, 2021, 2018, 2020),
         (2019, 2021, 2020, 2018), (2020, 2021, 2018, 2019), (2020, 2021, 2019, 2018)]


@lru_cache(maxsize=16)
def _read_aef(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)          # (64, H, W)


class WSTSAEFDataset(FireSpreadDataset):
    def __init__(self, *, aef_dir, use_aef=True, **kwargs):
        if kwargs.get("n_leading_observations", 1) != 1:
            raise NotImplementedError("Only single-day input is supported (n_leading_observations=1).")
        if kwargs.get("features_to_keep") is not None or kwargs.get("remove_duplicate_features"):
            raise NotImplementedError("features_to_keep / remove_duplicate_features are not supported.")
        if kwargs.get("return_doy"):
            raise NotImplementedError("return_doy is not supported.")
        self.aef_dir = Path(aef_dir)
        self.use_aef = use_aef
        super().__init__(**kwargs)

    # The authors' version splits paths on '/', which breaks on Windows. Same result, via pathlib.
    def read_list_of_images(self):
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}
            year_dir = Path(self.data_dir) / str(fire_year)
            if not year_dir.is_dir():
                raise FileNotFoundError(f"{year_dir} not found")
            if not self.load_from_hdf5:
                for fire_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
                    paths = sorted(str(p) for p in fire_dir.glob("*.tif"))
                    imgs_per_fire[fire_year][fire_dir.name] = paths
                    if not paths:
                        warnings.warn(f"Fire {fire_year}: {fire_dir.name} contains no images.", RuntimeWarning)
            else:
                for h5 in sorted(year_dir.glob("*.hdf5")):
                    imgs_per_fire[fire_year][h5.stem] = [str(h5)]
        return imgs_per_fire

    def aef_path(self, fire_year, fire_name):
        return self.aef_dir / f"{fire_year}__{fire_name}.tif"

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        x, y = super().load_imgs(found_fire_year, found_fire_name, in_fire_index)   # x: (T, 23, H, W)
        if not self.use_aef:
            return x, y
        e = _read_aef(str(self.aef_path(found_fire_year, found_fire_name)))
        if e.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"AEF grid {e.shape[-2:]} != WSTS grid {x.shape[-2:]} for "
                             f"{found_fire_year}/{found_fire_name}; re-export this fire.")
        e = np.broadcast_to(e[None], (x.shape[0],) + e.shape)
        x = np.concatenate([x[:, :E_RAW_POS], e, x[:, E_RAW_POS:]], axis=1)         # (T, 87, H, W)
        return x, y

    def standardize_features(self, x):
        if not self.use_aef:
            return super().standardize_features(x)
        lo, hi = E_RAW_POS, E_RAW_POS + N_AEF
        base = torch.cat([x[:, :lo], x[:, hi:]], dim=1)                              # 23 WSTS channels
        base = super().standardize_features(base)
        return torch.cat([base[:, :lo], x[:, lo:hi], base[:, lo:]], dim=1)           # embeddings untouched

    def __getitem__(self, index):
        x, y = super().__getitem__(index)                                            # x: (1, C, H, W)
        if not self.use_aef:
            return x[0], y
        e = x[0, E_FINAL]
        x = torch.cat([x[0, :E_FINAL.start], x[0, E_FINAL.stop:]], dim=0)            # (40, H, W)
        return x, e, y


def build_datasets(data_dir, aef_dir, fold=0, crop_side_length=128, load_from_hdf5=True, use_aef=True):
    """Same year-based splits as the benchmark (fold 0: train 2018-19, val 2020, test 2021)."""
    tr, va, te = list(FOLDS[fold][:2]), [FOLDS[fold][2]], [FOLDS[fold][3]]
    common = dict(data_dir=data_dir, aef_dir=aef_dir, use_aef=use_aef, n_leading_observations=1,
                  crop_side_length=crop_side_length, load_from_hdf5=load_from_hdf5,
                  remove_duplicate_features=False, stats_years=tr)
    train = WSTSAEFDataset(included_fire_years=tr, is_train=True, **common)
    val = WSTSAEFDataset(included_fire_years=va, is_train=True, **common)
    test = WSTSAEFDataset(included_fire_years=te, is_train=False,
                          n_leading_observations_test_adjustment=5, **common)
    return train, val, test


def build_loaders(train, val, test, batch_size=64, num_workers=4):
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
            DataLoader(test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True))