from typing import Tuple

import numpy as np
from numpy import random


def bigaussian(
    n_particles: int,
    mean: Tuple[float, float] = (0, 0),
    emittances: Tuple[float, float] = (3.5, 3.5),
):
    rng = random.default_rng()
    cov = np.diag(emittances)
    return rng.mulitvariate_normal(mean, cov, n_particles)
