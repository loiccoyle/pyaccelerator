from itertools import count
from typing import Optional

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import straight_element


class Drift(BaseElement):
    """Drift element.

    Args:
        l: Drift length in meters.
        name (optional): Element name.

    Attributes:
        l: Element length in meters.
        length: Element length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, l: float, name: Optional[str] = None):
        super().__init__("l", "name")
        self.l = l
        if name is None:
            name = f"drift_{next(self._instance_count)}"
        self.name = name

    def _get_length(self) -> float:
        return self.l

    def _get_transfer_matrix_h(self) -> np.ndarray:
        m_h = np.zeros((3, 3))
        m_h[0, 0] = 1
        m_h[0, 1] = self.length
        # m_h[0, 2] = 0
        # m_h[1, 0] = 0
        m_h[1, 1] = 1
        # m_h[1, 2] = 0
        m_h[2, 2] = 1
        return m_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        # same as horizontal
        return self._get_transfer_matrix_h()

    def slice(self, n_drifts: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_drifts: Number of :py:class:`Drift` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class`Drift`
            elements.
        """
        out = [
            Drift(self.length / n_drifts, name=f"{self.name}_slice_{i}")
            for i in range(n_drifts)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.Rectangle(
            (s, -0.5), self.length, 1, facecolor="tab:gray", alpha=0.5, label="Drift"
        )

    @staticmethod
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)
