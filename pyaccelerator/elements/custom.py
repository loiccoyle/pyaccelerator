from itertools import count
from typing import Optional

import numpy as np
from matplotlib import patches

from .base import BaseElement


class CustomThin(BaseElement):
    _instance_count = count(0)

    def __init__(
        self,
        transfer_matrix: np.ndarray,
        name: Optional[str] = None,
    ):
        """Custom element.

        Args:
            transfer_matrix: Transfer matrix of the element.
            name (optional): Element name.

        Attributes:
            transfer_matrix: Element phase space transfer matrix.
            length: Element length in meters.
            m: Element phase space transfer matrix in the horizonal plane.
            name: Element name.
        """
        if not isinstance(transfer_matrix, np.ndarray):
            transfer_matrix = np.array(transfer_matrix)

        self.transfer_matrix = transfer_matrix

        if name is None:
            name = f"custom_thin_{next(self._instance_count)}"
        super().__init__("transfer_matrix", "name")
        self.name = name

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix(self) -> np.ndarray:
        return self.transfer_matrix

    def _get_patch(self, s: float) -> patches.Patch:
        label = self.name
        colour = "black"

        return patches.FancyArrowPatch(
            (s, 1),
            (s, -1),
            arrowstyle=patches.ArrowStyle("-"),
            label=label,
            edgecolor=colour,
            facecolor=colour,
        )

    @staticmethod
    def _dxztheta_ds(
        theta: float,
        d_s: float,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, 0])
