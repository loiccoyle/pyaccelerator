from typing import Tuple

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import straight_element


class Drift(BaseElement):
    """Drift element.

    Args:
        length: Drift length in meters.

    Attributes:
        length: Element length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
    """

    def __init__(self, length: float):
        super().__init__(length)

    def transfer_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        m_h = np.zeros((2, 2))
        m_h[0][0] = 1
        m_h[0][1] = self.length
        # m_h[1][0] = 0
        m_h[1][1] = 1
        m_v = m_h
        return m_h, m_v

    def slice(self, n_drifts: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_drifts: number of `Drift` elements.

        Returns:
            `Lattice` of sliced `Drift` elements.
        """
        return Lattice([Drift(self.length / n_drifts)] * n_drifts)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.Rectangle(
            (s, -0.5), self.length, 1, facecolor="tab:gray", alpha=0.5, label="Drift"
        )

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)

    def __repr__(self):
        return f"Drift(length={self.length})"
