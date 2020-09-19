from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..transfer_matrix import TransferMatrix


class BaseElement:
    """Base class of a lattice element.

    Args:
        *instance_args: Arguments required to make the instance of this
            class's subclasses.
    """

    def __init__(self, *instance_args):
        # args of the subclass instance.
        self._instance_args = instance_args

    @property
    def length(self) -> float:
        return self._get_length()

    @property
    def m_h(self) -> TransferMatrix:
        """Horizontal phase space transfer matrix."""
        return TransferMatrix(self._get_transfer_matrix_h())

    @property
    def m_v(self) -> TransferMatrix:
        """Vertical phase space transfer matrix."""
        return TransferMatrix(self._get_transfer_matrix_v())

    @abstractmethod
    def _get_transfer_matrix_h(self) -> np.ndarray:  # pragma: no cover
        pass

    @abstractmethod
    def _get_transfer_matrix_v(self) -> np.ndarray:  # pragma: no cover
        pass

    @abstractmethod
    def _get_length(self) -> float:  # pragma: no cover
        pass

    @abstractmethod
    def _get_patch(self, s: float) -> Patch:
        """Generate a ``matplotlib.patches.Patch`` object to represent the
        element when plotting the lattice.

        Args:
            s: s coordinate where the patch should appear.

        Returns:
            ``matplotlib.patches.Patch`` which represents the element.
        """

    def _serialize(self) -> Dict[str, Any]:
        """Serialize the element.

        Returns:
            A serializable dictionary.
        """
        out = {key: getattr(self, key) for key in self._instance_args}
        # add element
        out["element"] = self.__class__.__name__
        return out

    def plot(
        self,
        *args,
        u_init: Optional[Sequence[float]] = None,
        plane="h",
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot the effect of the element on the phase space coords.

        Args:
            u_init: Initial phase space coords, defaults to [1, np.pi/8].
            plane: Ether "h" or "v".
            args, kwargs: Passed to ``matplotlib.pyplot.plot``.

        Returns:
            Plotted ``plt.Figure` and array of ``plt.Axes``.
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

    def copy(self) -> "BaseElement":
        """Create a copy of this instance.

        Returns:
            A copy of this instance.
        """
        return self.__class__(*[getattr(self, atr) for atr in self._instance_args])

    def __repr__(self) -> str:
        args = [f"{arg}={repr(getattr(self, arg))}" for arg in self._instance_args]
        return f"{self.__class__.__name__}({', '.join(args)})"
