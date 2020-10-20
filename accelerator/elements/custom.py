from typing import Optional, Tuple, Union
from itertools import count

import numpy as np
from matplotlib import patches

from .base import BaseElement


class CustomThin(BaseElement):

    _instance_count = count(0)

    def __init__(
        self,
        transfer_matrix: Union[np.ndarray, Tuple[np.ndarray]],
        name: Optional[str] = None,
    ):
        if isinstance(transfer_matrix, tuple):
            transfer_matrix_h, transfer_matrix_v = transfer_matrix
        else:
            transfer_matrix_h = transfer_matrix
            transfer_matrix_v = transfer_matrix

        self.transfer_matrix_h = transfer_matrix_h
        self.transfer_matrix_v = transfer_matrix_v

        if name is None:
            name = f"custom_thin_{next(self._instance_count)}"
        super().__init__("transfer_matrix_h", "transfer_matrix_v", "name")
        self.name = name

    def _get_length(self) -> float:
        return 0

    def _get_transfer_matrix_h(self) -> np.ndarray:
        return self.transfer_matrix_h

    def _get_transfer_matrix_v(self) -> np.ndarray:
        return self.transfer_matrix_v

    def _get_patch(self, s: float) -> Union[None, patches.Patch]:
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
        theta: float, d_s: float  # pylint: disable=unused-argument
    ) -> np.ndarray:
        return np.array([0, 0, 0])
