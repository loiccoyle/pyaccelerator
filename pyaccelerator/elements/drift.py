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
        m: Element phase space transfer matrix.
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

    def _get_transfer_matrix(self) -> np.ndarray:
        out = np.identity(5)
        out[0, 1] = self.length
        out[2, 3] = self.length
        return out

    def slice(self, n_slices: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_slices: Number of :py:class:`Drift` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Drift`
            elements.
        """
        out = [
            Drift(self.length / n_slices, name=f"{self.name}_slice_{i}")
            for i in range(n_slices)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.Rectangle(
            (s, -0.5), self.length, 1, facecolor="tab:gray", alpha=0.5, label="Drift"
        )

    @staticmethod
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)
