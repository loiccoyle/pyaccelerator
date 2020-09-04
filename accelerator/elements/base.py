from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..transfer_matrix import TransferMatrix


class BaseElement:
    """Element"""

    def __init__(self, length: float):
        """Base class of a lattice element.

        Args:
            length: length of the element.
        """
        self._m_h = None
        self._m_v = None
        self.length = length

    @property
    def m_h(self) -> TransferMatrix:
        if self._m_h is None or self._m_v is None:
            ms = self.transfer_matrix()
            self._m_h = TransferMatrix(ms[0])
            self._m_v = TransferMatrix(ms[1])
        return self._m_h

    @property
    def m_v(self) -> TransferMatrix:
        if self._m_h is None or self._m_v is None:
            ms = self.transfer_matrix()
            self._m_h = TransferMatrix(ms[0])
            self._m_v = TransferMatrix(ms[1])
        return self._m_v

    @abstractmethod
    def transfer_matrix(self) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        pass

    def _serialize(self) -> Dict[str, Any]:
        """Serialize the element.

        Returns:
            A serializable dictionary.
        """
        # remove private attributes
        out = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        # add element name
        out["element"] = self.__class__.__name__
        return out

    @abstractmethod
    def _get_patch(self, s: float) -> Patch:
        """Generate a `matplotlib.patches.Patch` object to represent the
        element when plotting the lattice.

        Args:
            s: s coordinate where the patch should appear.
        """

    def plot(
        self,
        *args,
        u_init: Optional[Sequence[float]] = None,
        plane="h",
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot the effect of the element on the phase space coords.

        Args:
            u_init: initial phase space coords, defaults to [1, np.pi/8].
            plane: ether "h" or "v".
            args, kwargs: plotting args/kwargs.

        Returns
            Plotted Figure and array of axes.
        """
        if u_init is None:
            u_init = [1, np.pi / 8]
        plane_map = {"h": self.m_h, "v": self.m_v}
        coord_map = {"h": "x", "v": "y"}
        coord = coord_map[plane]
        m = plane_map[plane]
        u_1 = m @ u_init
        x_axis = [0, self.length]
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(x_axis, [u_init[0], u_1[0]], *args, label=coord, **kwargs)
        axes[0].legend()
        axes[0].set_ylabel(f"{coord} (m)")
        axes[1].plot(x_axis, [u_init[1], u_1[1]], *args, label=f"{coord}'", **kwargs)
        axes[1].legend()
        axes[1].set_ylabel(f"{coord}'")
        axes[1].set_xlabel("s (m)")
        return fig, axes
