from typing import Tuple, Union

import numpy as np
from matplotlib import patches

from .base import BaseElement


class Quadrupole(BaseElement):
    """Quadrupole element.

    Thin lense approximation.

    Args:
        f: Quadrupole focal length in meters.

    Attributes:
        f: Element focal length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
    """

    def __init__(self, f: float):
        super().__init__(0)  # 0 length (thin lense)
        self.f = f

    def _get_transfer_matrix_h(self) -> np.ndarray:
        m_h = np.zeros((2, 2))
        m_h[0][0] = 1
        # m_h[0][1] = 0
        m_h[1][0] = -1 / self.f
        m_h[1][1] = 1
        return m_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        m_v = np.zeros((2, 2))
        m_v[0][0] = 1
        # m_v[0][1] = 0
        m_v[1][0] = 1 / self.f
        m_v[1][1] = 1
        return m_v

    def _get_patch(self, s: float) -> Union[None, patches.Patch]:
        if self.f < 0:
            head_length = -10
            label = "Defocussing Quad"
            colour = "tab:red"
        elif self.f > 0:
            head_length = 10
            label = "Focussing Quad"
            colour = "tab:blue"
        else:
            # if for whatever reason the strength is 0 skip
            return

        return patches.FancyArrowPatch(
            (s, 1),
            (s, -1),
            arrowstyle=patches.ArrowStyle(
                "<->", head_length=head_length, head_width=10
            ),
            label=label,
            edgecolor=colour,
            facecolor=colour,
        )

    def __repr__(self) -> str:
        return f"Quadrupole(f={self.f})"
