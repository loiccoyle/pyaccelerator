import numpy as np

from .utils import compute_m_twiss, compute_twiss_invariant


class TransferMatrix(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 2:
            raise ValueError(f"'{obj}' should be of shape (2, 2)")

        obj.twiss = TwissTransferMatrix(compute_m_twiss(obj))
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # pylint: disable=attribute-defined-outside-init
        self.twiss = getattr(obj, "twiss", None)


class TwissTransferMatrix(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 3:
            raise ValueError(f"'{obj}' should be of shape (3, 3)")

        try:
            obj.invariant = compute_twiss_invariant(obj)
        except ValueError:
            obj.invariant = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # pylint: disable=attribute-defined-outside-init
        self.invariant = getattr(obj, "invariant", None)
