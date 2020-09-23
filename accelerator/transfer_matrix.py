import numpy as np

from .utils import compute_m_twiss, compute_twiss_invariant


class TransferMatrix(np.ndarray):
    """Phase space transfer matrix."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 2:
            raise ValueError(f"'{obj}' should be of shape (2, 2)")

        @property
        def twiss(obj):
            return TwissTransferMatrix(compute_m_twiss(obj))

        setattr(obj.__class__, "twiss", twiss)
        return obj

    def __array_finalize__(self, obj):
        # I don't what this does but I found this snippet in the numpy docs
        # and I'm scared to remove it
        if obj is None:  # pragma: no cover
            return
        # pylint: disable=attribute-defined-outside-init
        setattr(self.__class__, "twiss", getattr(obj.__class__, "twiss", None))


class TwissTransferMatrix(np.ndarray):
    """Twiss parameter transfer matrix."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 3:
            raise ValueError(f"'{obj}' should be of shape (3, 3)")

        @property
        def invariant(obj):
            try:
                out = compute_twiss_invariant(obj)
            except ValueError:
                out = None
            return out

        setattr(obj.__class__, "invariant", invariant)
        return obj

    def __array_finalize__(self, obj):
        # I don't what this does but I found this snippet in the numpy docs
        # and I'm scared to remove it
        if obj is None:  # pragma: no cover
            return
        # pylint: disable=attribute-defined-outside-init
        setattr(self.__class__, "invariant", getattr(obj.__class__, "invariant", None))
