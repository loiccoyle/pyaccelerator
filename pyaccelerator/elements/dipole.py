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
        m: Element phase space transfer matrix.
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

    def _transport(self, phase_coords: np.ndarray) -> np.ndarray:
        rho = self.rho * (1 + phase_coords[4])
        out = np.zeros(phase_coords.shape)
        out[0] = (
            np.cos(self.theta) * phase_coords[0]
            + rho * np.sin(self.theta) * phase_coords[1]
            + rho * (1 - np.cos(self.theta)) * phase_coords[4]
        )
        out[1] = (
            -(1 / rho) * np.sin(self.theta) * phase_coords[0]
            + np.cos(self.theta) * phase_coords[1]
            + np.sin(self.theta) * phase_coords[4]
        )

        out[2] = phase_coords[2] + self.length * phase_coords[3]
        out[3] = phase_coords[3]
        out[4] = phase_coords[4]
        return out

    def _get_transfer_matrix(self) -> np.ndarray:
        out = np.zeros((5, 5))
        out[0, 0] = np.cos(self.theta)
        out[0, 1] = self.rho * np.sin(self.theta)
        out[1, 0] = -(1 / self.rho) * np.sin(self.theta)
        out[1, 1] = np.cos(self.theta)

        out[2, 2] = 1
        out[2, 3] = self.length
        out[3, 3] = 1

        out[0, 4] = self.rho * (1 - np.cos(self.theta))
        out[1, 4] = np.sin(self.theta)
        out[4, 4] = 1
        return out

    def slice(self, n_slices: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_slices: Number of :py:class:`Dipole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Dipole` elements.
        """
        out = [
            Dipole(self.rho, self.theta / n_slices, name=f"{self.name}_slice_{i}")
            for i in range(n_slices)
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
        m: Element phase space transfer matrix.
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

    def _get_transfer_matrix(self) -> np.ndarray:
        return np.identity(5)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.FancyArrowPatch(
            (s, 0.75),
            (s, -0.75),
            arrowstyle=patches.ArrowStyle("-"),
            label="Thin Dipole",
            edgecolor="lightcoral",
            facecolor="lightcoral",
        )

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return np.array([0, 0, self.theta])
