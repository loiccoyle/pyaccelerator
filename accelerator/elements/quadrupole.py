from itertools import count
from typing import Optional, Union

import numpy as np
from matplotlib import patches

from ..lattice import Lattice
from .base import BaseElement
from .utils import straight_element


class Quadrupole(BaseElement):
    """Quadrupole element.

    Args:
        k: Strength in meters^-2.
        l: Length in meters.
        name (optional): Element name.

    Attributes:
        k: Quadrupole trength in meters^-2.
        l: Element length in meters.
        length: Element length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(self, k: float, l: float, name: Optional[str] = None):
        self.l = l
        self.k = k
        if name is None:
            name = f"quadrupole_{next(self._instance_count)}"
        super().__init__("k", "l", "name")
        self.name = name

    def _get_length(self) -> float:
        return self.l

    def _get_transfer_matrix_h(self) -> np.ndarray:
        if self.k >= 0:
            return self.__focussing(self.k)
        return self.__defocussing(self.k)

    def _get_transfer_matrix_v(self) -> np.ndarray:
        if self.k >= 0:
            return self.__defocussing(-self.k)
        return self.__focussing(-self.k)

    def __focussing(self, k) -> np.ndarray:
        """Compute a foccusing matrix, with k >= 0."""
        sqrt_k = np.sqrt(k)
        m_f = np.zeros((3, 3))
        m_f[0, 0] = np.cos(sqrt_k * self.l)
        m_f[0, 1] = (1 / sqrt_k) * np.sin(sqrt_k * self.l)
        # m_f[0, 2] = 0
        m_f[1, 0] = -sqrt_k * np.sin(sqrt_k * self.l)
        m_f[1, 1] = np.cos(sqrt_k * self.l)
        # m_f[1, 2] = 0
        m_f[2, 2] = 1
        return m_f

    def __defocussing(self, k) -> np.ndarray:
        """Compute a defoccusing matrix, with k < 0."""
        sqrt_k = np.sqrt(abs(k))
        m_d = np.zeros((3, 3))
        m_d[0, 0] = np.cosh(sqrt_k * self.l)
        m_d[0, 1] = (1 / sqrt_k) * np.sinh(sqrt_k * self.l)
        # m_d[0, 2] = 0
        m_d[1, 0] = (sqrt_k) * np.sinh(sqrt_k * self.l)
        m_d[1, 1] = np.cosh(sqrt_k * self.l)
        # m_d[1, 2] = 0
        m_d[2, 2] = 1
        return m_d

    def slice(self, n_quadrupoles: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_quadrupoles: Number of :py:class:`Quadrupole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Quadrupole` elements.
        """
        out = [
            Quadrupole(
                self.k,
                self.l / n_quadrupoles,
                name=f"{self.name}_slice_{i}",
            )
            for i in range(n_quadrupoles)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> patches.Patch:
        if self.k < 0:
            label = "Defocussing Quad"
            colour = "tab:red"
        elif self.k > 0:
            label = "Focussing Quad"
            colour = "tab:blue"
        else:
            # if for whatever reason the strength is 0 skip
            return
        return patches.Rectangle((s, -1), self.length, 2, facecolor=colour, label=label)

    @staticmethod
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return straight_element(theta, d_s)


class QuadrupoleThin(BaseElement):
    """Thin Quadrupole element.

    Thin lense approximation.

    Args:
        f: Quadrupole focal length in meters.
        name (optional): Element name.

    Attributes:
        f: Element focal length in meters.
        m_h: Element phase space transfer matrix in the horizontal plane.
        m_v: Element phase space transfer matrix in the vertical plane.
    """

    _instance_count = count(0)

    def __init__(self, f: float, name: Optional[str] = None):
        self.f = f
        if name is None:
            name = f"quadrupole_thin_{next(self._instance_count)}"
        super().__init__("f", "name")
        self.name = name

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix_h(self) -> np.ndarray:
        m_h = np.zeros((3, 3))
        m_h[0, 0] = 1
        # m_h[0, 1] = 0
        # m_h[0, 2] = 0
        m_h[1, 0] = -1 / self.f
        m_h[1, 1] = 1
        # m_h[1, 2] = 0
        m_h[2, 2] = 1
        return m_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        m_v = np.zeros((3, 3))
        m_v[0, 0] = 1
        # m_v[0, 1] = 0
        # m_v[0, 2] = 0
        m_v[1, 0] = 1 / self.f
        m_v[1, 1] = 1
        # m_v[1, 2] = 0
        m_v[2, 2] = 1
        return m_v

    def _get_patch(self, s: float) -> Union[None, patches.Patch]:
        if self.f < 0:
            head_length = -10
            label = "Defocussing Thin Quad"
            colour = "tab:red"
        elif self.f > 0:
            head_length = 10
            label = "Focussing Thin Quad"
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

    @staticmethod
    def _dxztheta_ds(
        theta: float, d_s: float  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, 0])
