from typing import Tuple

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import bent_element


class Dipole(BaseElement):
    """Dipole element.

    Args:
        rho: Bending radius in meters.
        theta: Bending angle in radians.

    Attributes:
        length: Element length in meters.
        rho: Bending radius in meters.
        theta: Bending angle in radians.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
    """

    def __init__(self, rho: float, theta: float):
        super().__init__(rho * theta)
        self.rho = rho
        self.theta = theta

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

    def slice(self, n_dipoles: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_dipoles: number of :py:class:`Dipole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Dipole` elements.
        """
        return Lattice([Dipole(self.rho, self.theta / n_dipoles)] * n_dipoles)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.Rectangle(
            (s, -0.5), self.length, 1, facecolor="lightcoral", label="Dipole"
        )

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return bent_element(theta, d_s, self.rho)

    def __repr__(self) -> str:
        return f"Dipole(rho={self.rho:.4f}, theta={self.theta:.4f})"
