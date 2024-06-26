from typing import Tuple

import numpy as np


def bigaussian(
    n_particles: int,
    mean: Tuple[float, float, float, float, float],
    geometric_emittance_h: float,
    geometric_emittance_v: float,
    sigma_p: float,
) -> np.ndarray:
    """Generate a bigaussian distributed distribution.

    Args:
        n_particles: Number of particles.
        meam: Distribution centers.
        geometric_emittance: Geometric emittance.
        sigma_p: Absolute momentum spread.

    Returns:
        Array of position and angle phase space coordinates of the distribution.
    """
    cov = np.diag(
        (
            geometric_emittance_h,
            geometric_emittance_h,
            geometric_emittance_v,
            geometric_emittance_v,
            sigma_p**2,
        )
    )
    return np.random.multivariate_normal(mean, cov, n_particles).T
