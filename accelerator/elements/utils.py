import numpy as np

from ..utils import to_v_vec


def straight_element(theta: float, d_s: float) -> np.ndarray:
    return np.array([np.cos(theta) * d_s, np.sin(theta) * d_s, 0])


def bent_element(theta: float, d_s: float, rho: float) -> np.ndarray:
    phi = d_s / rho

    R = 2 * rho * np.sin(phi / 2)
    d_x = R * np.cos(theta + phi / 2)
    d_y = R * np.sin(theta + phi / 2)
    d_theta = phi
    return np.array([d_x, d_y, d_theta])
