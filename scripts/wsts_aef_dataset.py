"""
WildfireSpreadTS dataset + AlphaEarth cosine-medoid embeddings.

Subclasses the authors' FireSpreadDataset (MIT licence, https://github.com/SebastianGer/WildfireSpreadTS)
so cropping, flips/rotations (with wind-angle correction), standardization and one-hot land cover stay
identical to the benchmark. The 64 embedding channels are carried through the SAME random crop/flip/rotation
as the fire data, are not standardized (they are already unit-length), and are returned separately:

    x : (40, H, W)  WSTS features after the authors' preprocessing (single day)
    e : (64, H, W)  AlphaEarth medoid embedding of the preceding calendar year
    y : (H, W)      next-day active-fire mask (long, 0/1)

Model input = torch.cat([x, e], dim=0) -> 104 channels
(use x alone for the no-embedding baseline).

For Milestone 4, an optional CSV manifest can be supplied to restrict all
models to exactly the same fire subset. The manifest must contain:

    year,fire
    2018,fire_xxx
    ...

The train/validation split is still performed by year/fire first. All valid
consecutive day-to-next-day samples inside each selected fire are then used.

Setup (no second repo needed): copy FireSpreadDataset.py, utils.py and LICENSE
from the authors' repo (src/dataloader/) into scripts/wsts_vendor/ and add an
empty __init__.py there. Keep their LICENSE file.

Alternative: clone their repo anywhere and set WSTS_REPO to its path.
"""

import csv
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader


# Prefer the vendored WSTS loader.
try:
    from wsts_vendor.FireSpreadDataset import FireSpreadDataset
except ImportError:
    try:
        from .wsts_vendor.FireSpreadDataset import FireSpreadDataset
    except ImportError:
        _repo = os.environ.get("WSTS_REPO")

        if not _repo:
            raise ImportError(
                "Copy FireSpreadDataset.py, utils.py, LICENSE and an empty "
                "__init__.py into scripts/wsts_vendor/, or set WSTS_REPO "
                "to a clone of WildfireSpreadTS."
            )

        sys.path.insert(0, _repo)

        from src.dataloader.FireSpreadDataset import FireSpreadDataset  # noqa: E402


N_AEF = 64
N_RAW_BASE = 23

# Embeddings are inserted just before the raw "Active fire" feature.
E_RAW_POS = 22

# After WSTS preprocessing:
# 16 original features
# + 17 one-hot land-cover channels
# + 5 forecast features
# = 38 WSTS channels before the AEF position.
E_FINAL = slice(38, 38 + N_AEF)


# Original WSTS year-fold definitions.
FOLDS = [
    (2018, 2019, 2020, 2021),
    (2018, 2019, 2021, 2020),
    (2018, 2020, 2019, 2021),
    (2018, 2020, 2021, 2019),
    (2018, 2021, 2019, 2020),
    (2018, 2021, 2020, 2019),
    (2019, 2020, 2018, 2021),
    (2019, 2020, 2021, 2018),
    (2019, 2021, 2018, 2020),
    (2019, 2021, 2020, 2018),
    (2020, 2021, 2018, 2019),
    (2020, 2021, 2019, 2018),
]


def load_selected_fires(manifest_path):
    """
    Load selected fires from a CSV containing:

        year,fire

    Returns
    -------
    dict[int, set[str]] | None

        Example:

        {
            2018: {"fire_1", "fire_2"},
            2019: {"fire_3", "fire_4"},
        }

    If manifest_path is None, no fire filtering is performed.
    """
    if manifest_path is None:
        return None

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Selected-fire manifest not found: {manifest_path}"
        )

    selected = {}

    with manifest_path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {"year", "fire"}

        if (
            reader.fieldnames is None
            or not required_columns.issubset(reader.fieldnames)
        ):
            raise ValueError(
                "Selected-fire manifest must contain columns: "
                "'year' and 'fire'."
            )

        for row in reader:

            year_text = row["year"].strip()
            fire_name = row["fire"].strip()

            if not year_text or not fire_name:
                continue

            year = int(year_text)

            selected.setdefault(
                year,
                set(),
            ).add(fire_name)

    if not selected:
        raise ValueError(
            f"No fires were loaded from manifest: {manifest_path}"
        )

    return selected


@lru_cache(maxsize=16)
def _read_aef(path: str) -> np.ndarray:
    """
    Read one 64-band AEF GeoTIFF.

    Cached because all day-pairs from the same fire share the same
    annual AEF embedding.
    """
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


class WSTSAEFDataset(FireSpreadDataset):

    def __init__(
        self,
        *,
        aef_dir,
        use_aef=True,
        selected_fires_manifest=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        aef_dir:
            Directory containing AEF GeoTIFFs named
            <year>__<fire>.tif.

            This path is not accessed when use_aef=False.

        use_aef:
            True:
                return WSTS + AEF data for the 104-channel model.

            False:
                return WSTS only for the 40-channel control.

        selected_fires_manifest:
            Optional CSV path containing year,fire.

            When supplied, ONLY fires listed in the manifest are
            included in this dataset.
        """

        if kwargs.get(
            "n_leading_observations",
            1,
        ) != 1:
            raise NotImplementedError(
                "Only single-day input is supported "
                "(n_leading_observations=1)."
            )

        if (
            kwargs.get("features_to_keep") is not None
            or kwargs.get("remove_duplicate_features")
        ):
            raise NotImplementedError(
                "features_to_keep / remove_duplicate_features "
                "are not supported."
            )

        if kwargs.get("return_doy"):
            raise NotImplementedError(
                "return_doy is not supported."
            )

        self.aef_dir = Path(aef_dir)

        self.use_aef = use_aef

        # Must be loaded BEFORE super().__init__(), because the parent
        # constructor immediately calls read_list_of_images().
        self.selected_fires = load_selected_fires(
            selected_fires_manifest
        )

        super().__init__(**kwargs)

    def read_list_of_images(self):
        """
        Build the WSTS fire inventory.

        If selected_fires_manifest was supplied, only fires in that
        manifest are included.

        This filtering happens at the FIRE level before day-pairs are
        constructed, preventing samples from the same wildfire event
        from being split inconsistently.
        """

        imgs_per_fire = {}

        for fire_year in self.included_fire_years:

            fire_year = int(fire_year)

            imgs_per_fire[fire_year] = {}

            year_dir = (
                Path(self.data_dir)
                / str(fire_year)
            )

            if not year_dir.is_dir():
                raise FileNotFoundError(
                    f"{year_dir} not found"
                )

            allowed_fires = None

            if self.selected_fires is not None:

                if fire_year not in self.selected_fires:
                    raise ValueError(
                        f"Selected-fire manifest contains no fires "
                        f"for required year {fire_year}."
                    )

                allowed_fires = (
                    self.selected_fires[fire_year]
                )

            if not self.load_from_hdf5:

                fire_dirs = sorted(
                    p
                    for p in year_dir.iterdir()
                    if p.is_dir()
                )

                for fire_dir in fire_dirs:

                    fire_name = fire_dir.name

                    if (
                        allowed_fires is not None
                        and fire_name not in allowed_fires
                    ):
                        continue

                    paths = sorted(
                        str(p)
                        for p in fire_dir.glob("*.tif")
                    )

                    imgs_per_fire[fire_year][
                        fire_name
                    ] = paths

                    if not paths:
                        warnings.warn(
                            f"Fire {fire_year}: "
                            f"{fire_name} contains no images.",
                            RuntimeWarning,
                        )

            else:

                hdf5_files = sorted(
                    year_dir.glob("*.hdf5")
                )

                for h5 in hdf5_files:

                    fire_name = h5.stem

                    if (
                        allowed_fires is not None
                        and fire_name not in allowed_fires
                    ):
                        continue

                    imgs_per_fire[fire_year][
                        fire_name
                    ] = [str(h5)]

            # If a manifest is being used, fail loudly if any selected
            # WSTS fire is missing instead of silently training on a
            # smaller/different subset.
            if allowed_fires is not None:

                found_fires = set(
                    imgs_per_fire[
                        fire_year
                    ].keys()
                )

                missing_fires = (
                    allowed_fires
                    - found_fires
                )

                if missing_fires:

                    examples = sorted(
                        missing_fires
                    )[:10]

                    raise FileNotFoundError(
                        f"{fire_year}: "
                        f"{len(missing_fires)} selected fires "
                        f"are missing from {year_dir}. "
                        f"Examples: {examples}"
                    )

                if len(found_fires) != len(
                    allowed_fires
                ):
                    raise RuntimeError(
                        f"{fire_year}: expected "
                        f"{len(allowed_fires)} selected fires "
                        f"but loaded {len(found_fires)}."
                    )

        return imgs_per_fire

    def aef_path(
        self,
        fire_year,
        fire_name,
    ):
        return (
            self.aef_dir
            / f"{fire_year}__{fire_name}.tif"
        )

    def load_imgs(
        self,
        found_fire_year,
        found_fire_name,
        in_fire_index,
    ):
        """
        Load WSTS day-pair data.

        For use_aef=True, the annual AEF embedding for the same fire
        is inserted into every input day before the shared WSTS
        augmentation/preprocessing pipeline.
        """

        x, y = super().load_imgs(
            found_fire_year,
            found_fire_name,
            in_fire_index,
        )

        # WSTS-only 40-channel control.
        if not self.use_aef:
            return x, y

        aef_file = self.aef_path(
            found_fire_year,
            found_fire_name,
        )

        if not aef_file.is_file():
            raise FileNotFoundError(
                f"Missing AEF file for "
                f"{found_fire_year}/{found_fire_name}: "
                f"{aef_file}"
            )

        e = _read_aef(
            str(aef_file)
        )

        if e.shape[0] != N_AEF:
            raise ValueError(
                f"AEF file must contain {N_AEF} bands, "
                f"but {aef_file} has {e.shape[0]}."
            )

        if e.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"AEF grid {e.shape[-2:]} != "
                f"WSTS grid {x.shape[-2:]} for "
                f"{found_fire_year}/{found_fire_name}; "
                "re-export this fire."
            )

        # One annual AEF image is shared by all day-pairs from
        # the same wildfire event.
        e = np.broadcast_to(
            e[None],
            (x.shape[0],) + e.shape,
        )

        # Original raw WSTS has 23 channels.
        # Insert 64 AEF channels before Active fire.
        x = np.concatenate(
            [
                x[:, :E_RAW_POS],
                e,
                x[:, E_RAW_POS:],
            ],
            axis=1,
        )

        return x, y

    def standardize_features(
        self,
        x,
    ):
        """
        Standardize only WSTS features.

        AEF vectors are already unit-normalized and therefore remain
        untouched.
        """

        if not self.use_aef:
            return super().standardize_features(
                x
            )

        lo = E_RAW_POS
        hi = E_RAW_POS + N_AEF

        # Remove AEF temporarily.
        base = torch.cat(
            [
                x[:, :lo],
                x[:, hi:],
            ],
            dim=1,
        )

        base = (
            super()
            .standardize_features(base)
        )

        # Put unchanged AEF channels back in their original position.
        return torch.cat(
            [
                base[:, :lo],
                x[:, lo:hi],
                base[:, lo:],
            ],
            dim=1,
        )

    def __getitem__(
        self,
        index,
    ):

        x, y = super().__getitem__(
            index
        )

        # Parent dataset returns:
        #
        # x: (1, C, H, W)
        #
        # because n_leading_observations=1.

        if not self.use_aef:
            return x[0], y

        # Extract AEF after WSTS preprocessing.
        e = x[
            0,
            E_FINAL,
        ]

        # Remove AEF from x so the caller gets:
        #
        # x = 40 WSTS channels
        # e = 64 AEF channels
        #
        # The training engine concatenates them only when use_aef=True.
        x = torch.cat(
            [
                x[
                    0,
                    :E_FINAL.start,
                ],
                x[
                    0,
                    E_FINAL.stop:,
                ],
            ],
            dim=0,
        )

        return x, e, y


def build_datasets(
    data_dir,
    aef_dir,
    fold=0,
    crop_side_length=128,
    load_from_hdf5=True,
    use_aef=True,
    selected_fires_manifest=None,
):
    """
    Build one WSTS year-based fold.

    The fold parameter refers to the ORIGINAL WildfireSpreadTS loader
    fold index.

    M4 uses:

        M4 fold 0 -> loader fold 0
            train 2018 + 2019
            val   2020
            test  2021

        M4 fold 1 -> loader fold 2
            train 2018 + 2020
            val   2019
            test  2021

        M4 fold 2 -> loader fold 6
            train 2019 + 2020
            val   2018
            test  2021

    If selected_fires_manifest is supplied, all splits are restricted
    to the fires listed in that manifest.
    """

    if fold < 0 or fold >= len(FOLDS):
        raise ValueError(
            f"Invalid loader fold {fold}. "
            f"Expected 0-{len(FOLDS) - 1}."
        )

    tr = list(
        FOLDS[fold][:2]
    )

    va = [
        FOLDS[fold][2]
    ]

    te = [
        FOLDS[fold][3]
    ]

    common = dict(
        data_dir=data_dir,
        aef_dir=aef_dir,
        use_aef=use_aef,
        selected_fires_manifest=selected_fires_manifest,
        n_leading_observations=1,
        crop_side_length=crop_side_length,
        load_from_hdf5=load_from_hdf5,
        remove_duplicate_features=False,
        stats_years=tr,
    )

    train = WSTSAEFDataset(
        included_fire_years=tr,
        is_train=True,
        **common,
    )

    val = WSTSAEFDataset(
        included_fire_years=va,
        is_train=True,
        **common,
    )

    test = WSTSAEFDataset(
        included_fire_years=te,
        is_train=False,
        n_leading_observations_test_adjustment=5,
        **common,
    )

    return train, val, test


def build_loaders(
    train,
    val,
    test,
    batch_size=64,
    num_workers=4,
):
    """
    Build PyTorch DataLoaders for one fold.
    """

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )