from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from matplotlib.patches import Patch

from ..transfer_matrix import TransferMatrix

if TYPE_CHECKING:
    from ..lattice import Lattice


class BaseElement:
    name: str

    def __init__(self, *instance_args):
        """Base class of a lattice element.

        Args:
            *instance_args: Arguments required to make the instance of this
                class's subclasses.
        """
        # args of the subclass instance.
        self._instance_args = instance_args

    @property
    def length(self) -> float:
        return self._get_length()

    @property
    def m(self) -> TransferMatrix:
        """Phase space transfer matrix."""
        return TransferMatrix(self._get_transfer_matrix())

    @abstractmethod
    def _get_transfer_matrix(self) -> np.ndarray:  # pragma: no cover
        pass

    @abstractmethod
    def _get_length(self) -> float:  # pragma: no cover
        pass

    @abstractmethod
    def _get_patch(self, s: float) -> Optional[Patch]:
        """Generate a ``matplotlib.patches.Patch`` object to represent the
        element when plotting the lattice.

        Args:
            s: s coordinate where the patch should appear.

        Returns:
            ``matplotlib.patches.Patch`` which represents the element.
        """

    @abstractmethod
    def slice(self, n_slices: int) -> "Lattice":
        """Slice the element into many smaller elements.
        Args:
            n_slices: Number of elements to slice the element into.
        Returns:
            A list of sliced elements.
        """

    def _transport(self, phase_coords: np.ndarray) -> np.ndarray:
        return (self._get_transfer_matrix() @ phase_coords) + self._non_linear_term(
            phase_coords
        )

    def _non_linear_term(self, phase_coords: np.ndarray) -> np.ndarray:
        return np.zeros(phase_coords.shape)

    def _serialize(self) -> Dict[str, Any]:
        """Serialize the element.

        Returns:
            A serializable dictionary.
        """
        out = {key: getattr(self, key) for key in self._instance_args}
        # add element
        out["element"] = self.__class__.__name__
        return out

    def copy(self) -> "BaseElement":
        """Create a copy of this instance.

        Returns:
            A copy of this instance.
        """
        return self.__class__(*[getattr(self, atr) for atr in self._instance_args])

    def __repr__(self) -> str:
        args = [f"{arg}={repr(getattr(self, arg))}" for arg in self._instance_args]
        return f"{self.__class__.__name__}({', '.join(args)})"
