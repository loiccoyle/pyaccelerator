from typing import Tuple

import numpy as np

from .base import BaseElement
from .utils import bent_element


class Dipole(BaseElement):
    """Dipole element"""

    def __init__(self, rho: float, theta: float):
        """Dipole element.

        Args:
            rho: bending radius in meters.
            theta: bending angle in radians.

        Attributes:
            rho: bending radius in meters.
            theta: bending angle in radians.
            m_h: element transfer matrix horizontal plane.
            m_v: element transfer matrix vertical plane.
        """
        super().__init__()
        self.rho = rho
        self.theta = theta
        self.length = rho * theta

    def transfer_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # horizontal
        m_h = np.zeros((2, 2))
        m_h[0][0] = np.cos(self.theta)
        m_h[0][1] = self.rho * np.sin(self.theta)
        m_h[1][0] = -(1 / self.rho) * np.sin(self.theta)
        m_h[1][1] = np.cos(self.theta)
        # vertical
        m_v = np.zeros((2, 2))
        m_v[0][0] = 1
        m_v[0][1] = self.length
        # m_v[1][0] = 0
        m_v[1][1] = 1
        return m_h, m_v

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return bent_element(theta, d_s, self.rho)

    def __repr__(self) -> str:
        return f"Dipole(rho={self.rho:.4f}, theta={self.theta:.4f})"
