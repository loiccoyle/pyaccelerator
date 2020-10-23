from itertools import count
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import patches

from .base import BaseElement


class CustomThin(BaseElement):

    _instance_count = count(0)

    def __init__(
        self,
        transfer_matrix_h: Optional[np.ndarray] = None,
        transfer_matrix_v: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ):
        """Custom element.

        Args:
            transfer_matrix_h: Transfer matrix of the element in the
                horizonal plane. If only one transfer matrix is provided it will
                also be used for the other plane.
            transfer_matrix_v: Transfer matrix of the element in the
                vertical plane. If only one transfer matrix is provided it will
                also be used for the other plane.
            name (optional): Element name.

        Attributes:
            transfer_matrix_h: Element phase space transfer matrix in the
                horizonal plane.
            transfer_matrix_v: Element phase space transfer matrix in the
                vertical plane.
            length: Element length in meters.
            m_h: Element phase space transfer matrix in the horizonal plane.
            m_h: Element phase space transfer matrix in the vertical plane.
            name: Element name.
        """
        if transfer_matrix_h is None and transfer_matrix_v is None:
            raise ValueError(
                "Provide at least one of 'transfer_matrix_h', 'transfer_matrix_v'."
            )
        if transfer_matrix_h is None:
            transfer_matrix_h = transfer_matrix_v
        if transfer_matrix_v is None:
            transfer_matrix_v = transfer_matrix_h

        if not isinstance(transfer_matrix_h, np.ndarray):
            transfer_matrix_h = np.array(transfer_matrix_h)
        if not isinstance(transfer_matrix_v, np.ndarray):
            transfer_matrix_v = np.array(transfer_matrix_v)

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
