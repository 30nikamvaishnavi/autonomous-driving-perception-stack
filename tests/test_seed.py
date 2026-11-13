import random

import numpy as np

from adpstack.utils.seed import seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences():
    seed_everything(123)
    first = (random.random(), np.random.random())

    seed_everything(123)

    assert (random.random(), np.random.random()) == first
