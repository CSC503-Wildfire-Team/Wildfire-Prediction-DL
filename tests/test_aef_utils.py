import numpy as np

from scripts.aef_utils import block_medoids


def test_block_medoids_returns_one_of_the_input_vectors_and_ignores_invalid_pixels():
    fine = np.full((2, 2, 4), np.nan, dtype=np.float32)
    fine[:, 0, 0] = [1, 0]
    fine[:, 0, 1] = [0, 1]
    fine[:, 1, 0] = [0, 1]
    fine[:, 1, 1] = [0, 1]
    fine[:, :, 2:] = 0

    medoids = block_medoids(fine, n=2)

    np.testing.assert_allclose(medoids[:, 0, 0], [0, 1])
    assert np.isnan(medoids[:, 0, 1]).all()
