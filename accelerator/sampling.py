from typing import Tuple

import numpy as np


def bigaussian(
    n_particles: int,
    mean: Tuple[float, float, float],
    geometric_emittance: float,
    sigma_p: float,
) -> np.array:
    """Generate a bigaussian distributed distribution.

    Args:
        n_particles: Number of particles.
        meam: Distribution centers.
        geometric_emittance: Geometric emittance.

    Returns:
        Array of position and angle phase space coordinates of the distribution.
    """
    cov = np.diag((geometric_emittance, geometric_emittance, sigma_p))
    return np.random.multivariate_normal(mean, cov, n_particles).T
