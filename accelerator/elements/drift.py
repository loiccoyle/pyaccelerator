from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseElement
from .utils import straight_element


class Drift(BaseElement):
    """Drift element"""

    def __init__(self, l: float):
        """Drift element.

        Args:
            l: drift length in meters.

        Attributes:
            l: drift length.
            m_h: element transfer matrix horizontal plane.
            m_v: element transfer matrix vertical plane.
        """
        super().__init__()
        self.l = l
        self.length = l

    def transfer_matrix(self) -> np.ndarray:
        m_h = np.zeros((2, 2))
        m_h[0][0] = 1
        m_h[0][1] = self.l
        # m_h[1][0] = 0
        m_h[1][1] = 1
        m_v = m_h
        return m_h, m_v

    def _dxztheta_ds(self, theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)

    def __repr__(self):
        return f"Drift(l={self.l})"
