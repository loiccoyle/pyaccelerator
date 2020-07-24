from typing import Tuple

import numpy as np

from .base import BaseElement
from .utils import straight_element


class Quadrupole(BaseElement):
    def __init__(self, f: float):
        """Quadrupole element.

        Thin lense approximation.

        Args:
            f: quadrupole focal length in meters.

        Attributes:
            f: quadrupole focal length in meters.
            m_h: element transfer matrix horizontal plane.
            m_v: element transfer matrix vertical plane.
        """
        super().__init__()
        self.f = f
        self.length = 0

    def transfer_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        m_h = np.zeros((2, 2))
        m_h[0][0] = 1
        # m_h[0][1] = 0
        m_h[1][0] = -1 / self.f
        m_h[1][1] = 1

        m_v = np.zeros((2, 2))
        m_v[0][0] = 1
        # m_v[0][1] = 0
        m_v[1][0] = 1 / self.f
        m_v[1][1] = 1
        return m_h, m_v

    def __repr__(self) -> str:
        return f"Quadrupole(f={self.f})"
