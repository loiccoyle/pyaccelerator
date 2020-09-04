from typing import Tuple, Union

import numpy as np
from matplotlib import patches

from .base import BaseElement


class Quadrupole(BaseElement):
    """Quadrupole element"""

    def __init__(self, f: float):
        """Quadrupole element.

        Thin lense approximation.

        Args:
            f: quadrupole focal length in meters.

        Attributes:
            f: element focal length in meters.
            m_h: element phase space transfer matrix in the horizontal plane.
            m_v: element phase space transfer matrix in the vertical plane.
        """
        super().__init__(0)  # 0 length (thin lense)
        self.f = f

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
