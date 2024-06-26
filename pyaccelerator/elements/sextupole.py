from itertools import count
from typing import Optional

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import straight_element


class Sextupole(BaseElement):
    """Sextupole element.

    Args:
        k: Strength in meters^-3.
        l: Element length in meters.
        name (optional): Element name.

    Attributes:
        k: Sextupole strength in meters^-3.
        l: Element length in meters.
        length: Element length in meters.
        m: Element phase space transfer matrix.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, k: float, l: float, name: Optional[str] = None):
        self.k = k
        self.l = l
        if name is None:
            name = f"sextupole_{next(self._instance_count)}"
        super().__init__("k", "l", "name")
        self.name = name

    def _non_linear_term(self, phase_coords: np.ndarray) -> np.ndarray:
        # see http://cern.ch/mad8/doc/phys_guide.pdf chapter 5.5.3
        out = np.zeros(phase_coords.shape)
        # strength depends on dp/p
        k = self.k / (1 + phase_coords[4])
        # k = self.k

        out[1] = -0.5 * k * self.l * (phase_coords[0] ** 2 - phase_coords[2] ** 2)
        out[3] = k * self.l * (phase_coords[0] * phase_coords[2])
        return out

    def _get_length(self) -> float:
        return self.l

    def _get_transfer_matrix(self) -> np.ndarray:
        out = np.identity(5)
        out[0, 1] = self.l
        out[2, 3] = self.l
        return out

    def slice(self, n_slices: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_slices): Number of :py:class:`Sextupole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Sextupole` elements.
        """
        out = [
            Sextupole(
                self.k,
                self.l / n_slices,
                name=f"{self.name}_slice_{i}",
            )
            for i in range(n_slices)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> patches.Patch:
        label = "Sextupole"
        colour = "tab:green"
        return patches.Rectangle((s, -1), self.length, 2, facecolor=colour, label=label)

    @staticmethod
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)


class SextupoleThin(BaseElement):
    """Thin Sextupole element.

    Args:
        k: Strength in meters^-2.
        name (optional): Element name.

    Attributes:
        k: Sextupole strength in meters^-2.
        length: Element length in meters.
        m: Element phase space transfer matrix.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, k: float, name: Optional[str] = None):
        self.k = k
        if name is None:
            name = f"sextupole_thin_{next(self._instance_count)}"
        super().__init__("k", "name")
        self.name = name

    def _non_linear_term(self, phase_coords: np.ndarray) -> np.ndarray:
        out = np.zeros(phase_coords.shape)
        out[1] = -0.5 * self.k * (phase_coords[0] ** 2 - phase_coords[2] ** 2)
        out[3] = self.k * (phase_coords[0] * phase_coords[2])
        return out

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix(self) -> np.ndarray:
        return np.identity(5)

    def _get_patch(self, s: float) -> patches.Patch:
        label = "Thin Sextupole"
        colour = "tab:green"
        return patches.FancyArrowPatch(
            (s, 1),
            (s, -1),
            arrowstyle=patches.ArrowStyle("|-|"),
            label=label,
            edgecolor=colour,
            facecolor=colour,
        )

    @staticmethod
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return np.array([0, 0, 0])
