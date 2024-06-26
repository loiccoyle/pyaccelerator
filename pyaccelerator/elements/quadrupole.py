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
        m: Element phase space transfer matrix.
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

    def _get_transfer_matrix(self) -> np.ndarray:
        sin_x, cos_x, sin_y, cos_y = self._compute_sin_cos(self.k)

        out = np.zeros((5, 5))
        out[0, 0] = cos_x
        out[0, 1] = sin_x
        out[1, 0] = -self.k * sin_x
        out[1, 1] = cos_x

        out[2, 2] = cos_y
        out[2, 3] = sin_y
        out[3, 2] = self.k * sin_y
        out[3, 3] = cos_y

        out[4, 4] = 1
        return out

    def _transport(self, phase_coords: np.ndarray) -> np.ndarray:
        # strength depends on dp/p
        k = self.k / (1 + phase_coords[4])

        sin_x, cos_x, sin_y, cos_y = self._compute_sin_cos(k)
        out = np.zeros(phase_coords.shape)
        out[0] = cos_x * phase_coords[0] + sin_x * phase_coords[1]
        out[1] = -k * sin_x * phase_coords[0] + cos_x * phase_coords[1]

        out[2] = cos_y * phase_coords[2] + sin_y * phase_coords[3]
        out[3] = k * sin_y * phase_coords[2] + cos_y * phase_coords[3]

        out[4] = phase_coords[4]
        return out

    def _compute_sin_cos(self, k):
        # needs to work with arrays in order to track a distribution of particles.
        if isinstance(k, float):
            k = np.array(k)

        sqrt_k = np.sqrt(abs(k))

        sin_x = np.zeros(k.shape)
        cos_x = np.zeros(k.shape)
        sin_y = np.zeros(k.shape)
        cos_y = np.zeros(k.shape)

        k_pos_mask = k >= 0
        sqrt_k_pos = sqrt_k[k_pos_mask]

        k_neg_mask = k < 0
        sqrt_k_neg = sqrt_k[k_neg_mask]

        sin_x[k_pos_mask] = np.sin(sqrt_k_pos * self.l) / sqrt_k_pos
        cos_x[k_pos_mask] = np.cos(sqrt_k_pos * self.l)
        sin_y[k_pos_mask] = np.sinh(sqrt_k_pos * self.l) / sqrt_k_pos
        cos_y[k_pos_mask] = np.cosh(sqrt_k_pos * self.l)

        sin_x[k_neg_mask] = np.sinh(sqrt_k_neg * self.l) / sqrt_k_neg
        cos_x[k_neg_mask] = np.cosh(sqrt_k_neg * self.l)
        sin_y[k_neg_mask] = np.sin(sqrt_k_neg * self.l) / sqrt_k_neg
        cos_y[k_neg_mask] = np.cos(sqrt_k_neg * self.l)

        return sin_x, cos_x, sin_y, cos_y

    def slice(self, n_slices: int) -> Lattice:
        """Slice the element into a many smaller elements.

        Args:
            n_slices: Number of :py:class:`Quadrupole` elements.

        Returns:
            :py:class:`~accelerator.lattice.Lattice` of sliced :py:class:`Quadrupole` elements.
        """
        out = [
            Quadrupole(
                self.k,
                self.l / n_slices,
                name=f"{self.name}_slice_{i}",
            )
            for i in range(n_slices)
        ]
        return Lattice(out)

    def _get_patch(self, s: float) -> Optional[patches.Patch]:
        if self.k < 0:
            label = "Defocussing Quad."
            colour = "tab:red"
        elif self.k > 0:
            label = "Focussing Quad."
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
        m: Element phase space transfer matrix.
        name: Element name.
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

    def _get_transfer_matrix(self) -> np.ndarray:
        out = np.zeros((5, 5))
        out[0, 0] = 1
        out[1, 0] = -1 / self.f
        out[1, 1] = 1

        out[2, 2] = 1
        out[3, 2] = 1 / self.f
        out[3, 3] = 1

        out[4, 4] = 1
        return out

    def _transport(self, phase_coords: np.ndarray) -> np.ndarray:
        # overwrite transport method to have a focal length which depends on dp/p
        f = self.f * (1 + phase_coords[4])
        one_over_f = 1 / f

        out = np.zeros(phase_coords.shape)
        out[0] = phase_coords[0]
        out[1] = -one_over_f * phase_coords[0] + phase_coords[1]

        out[2] = phase_coords[2]
        out[3] = one_over_f * phase_coords[2] + phase_coords[3]

        out[4] = phase_coords[4]
        return out

    def _get_patch(self, s: float) -> Union[None, patches.Patch]:
        if self.f < 0:
            head_length = -10
            label = "Defocussing Thin Quad."
            colour = "tab:red"
        elif self.f > 0:
            head_length = 10
            label = "Focussing Thin Quad."
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
    def _dxztheta_ds(theta: float, d_s: float) -> np.ndarray:
        return np.array([0, 0, 0])
