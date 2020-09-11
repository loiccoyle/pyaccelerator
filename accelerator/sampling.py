from typing import Tuple

import numpy as np


def bigaussian(
    n_particles: int,
    mean: Tuple[float, float],
    geometric_emittance: float,
) -> np.array:
    """Generate a bigaussian distributed distribution.

    Args:
        n_particles: number of particles.
        meam: distribution centers.
        geometric_emittance: geometric emittance.

    Returns:
        Array of position and angle phase space coordinates of the distribution.
    """
    cov = np.diag((geometric_emittance, geometric_emittance))
    return np.random.multivariate_normal(mean, cov, n_particles).T
