import numpy as np

from .utils import compute_dispersion_solution, compute_m_twiss, compute_twiss_solution


class TransferMatrix(np.ndarray):
    """Phase space transfer matrix."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 3:
            raise ValueError(f"'{obj}' should be of shape (3, 3)")

        @property
        def twiss(obj):
            return TwissTransferMatrix(compute_m_twiss(obj))

        @property
        def twiss_solution(obj):
            try:
                out = compute_twiss_solution(obj)
            except ValueError:
                out = None
            return out

        @property
        def dispersion_solution(obj):
            try:
                out = compute_dispersion_solution(obj)
            except ValueError:
                out = None
            return out

        setattr(obj.__class__, "twiss", twiss)
        setattr(obj.__class__, "twiss_solution", twiss_solution)
        setattr(obj.__class__, "dispersion_solution", dispersion_solution)
        return obj

    def __array_finalize__(self, obj):
        # I don't what this does but I found this snippet in the numpy docs
        # and I'm scared to remove it
        if obj is None:  # pragma: no cover
            return
        # pylint: disable=attribute-defined-outside-init
        setattr(self.__class__, "twiss", getattr(obj.__class__, "twiss", None))
        setattr(
            self.__class__,
            "twiss_solution",
            getattr(obj.__class__, "twiss_solution", None),
        )
        setattr(
            self.__class__,
            "dispersion_solution",
            getattr(obj.__class__, "dispersion_solution", None),
        )


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
        return obj

    def __array_finalize__(self, obj):
        # I don't what this does but I found this snippet in the numpy docs
        # and I'm scared to remove it
        if obj is None:  # pragma: no cover
            return
