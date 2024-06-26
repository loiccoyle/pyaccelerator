from itertools import count
from typing import Optional

import numpy as np
from matplotlib import patches

from .base import BaseElement


class KickerThin(BaseElement):
    """Thin Kicker element.

    Args:
        h_kick: kick angle in the horizontal plane.
        v_kick: kick angle in the vertical plane.
        name (optional): Element name.

    Attributes:
        h_kick: kick angle in the horizontal plane.
        v_kick: kick angle in the vertical plane.
        m: Element phase space transfer matrix.
        name: Element name.
    """

    _instance_count = count(0)

    def __init__(
        self, h_kick: float = 0, v_kick: float = 0, name: Optional[str] = None
    ):
        self.h_kick = h_kick
        self.v_kick = v_kick
        if name is None:
            name = f"kicker_thin_{next(self._instance_count)}"
        super().__init__("h_kick", "v_kick", "name")
        self.name = name

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix(self) -> np.ndarray:
        return np.identity(5)

    def _transport(self, phase_coords: np.ndarray) -> np.ndarray:
        out = phase_coords.copy()
        out[1] += self.h_kick
        out[3] += self.v_kick
        return out

    def _get_patch(self, s: float) -> patches.Patch:
        return patches.FancyArrowPatch(
            (s, 0.75),
            (s, -0.75),
            arrowstyle=patches.ArrowStyle("-"),
            label="Kicker",
            edgecolor="maroon",
            facecolor="maroon",
        )

    def _dxztheta_ds(
        self,
        theta: float,
        d_s: float,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, self.h_kick])
