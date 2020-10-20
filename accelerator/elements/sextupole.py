from itertools import count
from typing import Optional

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import NonLinearTerm


class SextupoleThin(BaseElement):
    """Sextupole element.

    Args:
        k: Strength in meters^-2.
        name (optional): Element name.

    Attributes:
        k: Sextupole strength in meters^-2.
        length: Element length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, k: float, name: Optional[str] = None):
        self.k = k
        if name is None:
            name = f"sextupole_{next(self._instance_count)}"
        super().__init__("k", "name")
        self.name = name

    def _non_linear_term(self, phase_coord: np.ndarray) -> np.ndarray:
        out = np.zeros_like(phase_coord)
        out[1] = -(1 / 2) * self.k * phase_coord[0] ** 2
        return out

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix_h(self) -> np.ndarray:
        m_h = np.zeros((3, 3))
        m_h[0, 0] = 1
        m_h[0, 1] = 0
        # m_h[0, 2] = 0
        # m_h[1, 0] = 0
        m_h[1, 1] = 1
        # m_h[1, 2] = 0
        m_h[2, 2] = 1
        return m_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        return self._get_transfer_matrix_h()

    def _get_patch(self, s: float) -> patches.Patch:
        # if self.k < 0:
        #     label = "Defocussing Quad"
        #     colour = "tab:red"
        # elif self.k > 0:
        #     label = "Focussing Quad"
        #     colour = "tab:blue"
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
    def _dxztheta_ds(
        theta: float, d_s: float  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, 0])
