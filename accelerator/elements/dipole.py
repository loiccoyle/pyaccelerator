from itertools import count
from typing import Optional

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
        name (optional): Element name.

    Attributes:
        length: Element length in meters.
        rho: Bending radius in meters.
        theta: Bending angle in radians.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, rho: float, theta: float, name: Optional[str] = None):
        self.rho = rho
        self.theta = theta
        if name is None:
            name = f"dipole_{next(self._instance_count)}"
        super().__init__("rho", "theta", "name")
        self.name = name

    def _get_length(self) -> float:
        return self.rho * self.theta

    def _get_transfer_matrix_h(self) -> np.ndarray:
        # horizontal
        m_h = np.zeros((2, 2))
        m_h[0][0] = np.cos(self.theta)
        m_h[0][1] = self.rho * np.sin(self.theta)
        m_h[1][0] = -(1 / self.rho) * np.sin(self.theta)
        m_h[1][1] = np.cos(self.theta)
        return m_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        # vertical
        m_v = np.zeros((2, 2))
        m_v[0][0] = 1
        m_v[0][1] = self.length
        # m_v[1][0] = 0
        m_v[1][1] = 1
        return m_v

    def slice(self, n_dipoles: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_dipoles: Number of :py:class:`Dipole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Dipole` elements.
        """
        out = [
            Dipole(self.rho, self.theta / n_dipoles, name=f"{self.name}_slice_{i}")
            for i in range(n_dipoles)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.Rectangle(
            (s, -0.75), self.length, 1.5, facecolor="lightcoral", label="Dipole"
        )

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return bent_element(theta, d_s, self.rho)


class DipoleThin(BaseElement):
    """Thin Dipole element.

    Args:
        theta: Bending angle in radians.
        name (optional): Element name.

    Attributes:
        length: Element length in meters.
        theta: Bending angle in radians.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, theta: float, name: Optional[str] = None):
        self.theta = theta
        if name is None:
            name = f"dipole_thin_{next(self._instance_count)}"
        super().__init__("theta", "name")
        self.name = name

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix_h(self) -> np.ndarray:
        # horizontal
        return np.identity(2)

    def _get_transfer_matrix_v(self) -> np.ndarray:
        # vertical
        return np.identity(2)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.FancyArrowPatch(
            (s, 0.75),
            (s, -0.75),
            arrowstyle=patches.ArrowStyle("-"),
            label="Thin Dipole",
            edgecolor="lightcoral",
            facecolor="lightcoral",
        )

    def _dxztheta_ds(
        self, theta: float, d_s: float  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, self.theta])
