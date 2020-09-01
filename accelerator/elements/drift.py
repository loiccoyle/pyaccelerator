from typing import Tuple

import numpy as np

from ..lattice import Lattice
from .base import BaseElement
from .utils import straight_element


class Drift(BaseElement):
    """Drift element"""

    def __init__(self, length: float):
        """Drift element.

        Args:
            length: drift length in meters.

        Attributes:
            length: drift length in meters.
            m_h: element transfer matrix horizontal plane.
            m_v: element transfer matrix vertical plane.
        """
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

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)

    def __repr__(self):
        return f"Drift(length={self.length})"
