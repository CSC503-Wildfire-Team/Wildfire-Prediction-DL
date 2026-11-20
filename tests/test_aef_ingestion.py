from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aef_utils import BANDS, block_medoids, validate_grid, validate_medoid_image


GRID = {
    "crs": "EPSG:32610",
    "transform": [375, 0, 500000, 0, -375, 5500000],
    "width": 4,
    "height": 3,
}


def test_block_medoids_returns_a_valid_input_vector():
    fine = np.full((2, 2, 2), np.nan, dtype=np.float32)
    fine[:, 0, 0] = [1, 0]
    fine[:, 0, 1] = [0, 1]
    fine[:, 1, 0] = [0, 1]
    fine[:, 1, 1] = [0, 1]

    medoids = block_medoids(fine, n=2)

    np.testing.assert_allclose(medoids[:, 0, 0], [0, 1])


def test_grid_validation_rejects_rotated_or_sheared_grids():
    grid = {**GRID, "transform": [375, 1, 500000, 0, -375, 5500000]}

    with pytest.raises(ValueError, match="Rotated/sheared"):
        validate_grid(grid)


class _Info:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class _Sample:
    def __init__(self, feature):
        self.feature = feature

    def first(self):
        return self

    def getInfo(self):
        return self.feature


class _Image:
    def __init__(self, grid, vector):
        self.grid = grid
        self.vector = vector

    def bandNames(self):
        return _Info(BANDS)

    def projection(self):
        return _Info({"crs": self.grid["crs"], "transform": self.grid["transform"]})

    def sample(self, **kwargs):
        assert kwargs["numPixels"] == 1
        assert kwargs["dropNulls"] is True
        return _Sample({"properties": dict(zip(BANDS, self.vector))})


class _EE:
    class Geometry:
        @staticmethod
        def Rectangle(*args, **kwargs):
            return None


def test_medoid_image_validation_checks_grid_and_unit_norm():
    image = _Image(GRID, np.full(len(BANDS), 1 / 8))

    validate_medoid_image(_EE, image, GRID)


def test_medoid_image_validation_rejects_georeferencing_mismatch():
    wrong_grid = {**GRID, "transform": [375, 0, 500001, 0, -375, 5500000]}
    image = _Image(wrong_grid, np.full(len(BANDS), 1 / 8))

    with pytest.raises(ValueError, match="transform"):
        validate_medoid_image(_EE, image, GRID)