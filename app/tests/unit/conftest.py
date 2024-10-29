import numpy as np
import pytest


@pytest.fixture
def fixed_signal():
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)


@pytest.fixture
def random_signal():
    def create(rows, columns, seed=None):
        if seed:
            np.random.seed(seed)

        return np.random.rand(rows, columns)

    return create
