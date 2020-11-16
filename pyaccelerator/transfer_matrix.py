import numpy as np

from .utils import PLANE_SLICES, compute_m_twiss


class TransferMatrix(np.ndarray):
    """Phase space transfer matrix."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 2:
            raise ValueError(f"'{obj}' should be 2D.")
        if obj.shape[0] != obj.shape[1]:
            raise ValueError(f"'{obj}' is not square.")
        if obj.shape[0] != 5:
            raise ValueError(f"'{obj}' should be of shape (5, 5)")

        # @property
        def twiss(obj, plane="h"):
            plane = plane.lower()
            return compute_m_twiss(obj[PLANE_SLICES[plane], PLANE_SLICES[plane]])

        @property
        def h(obj):
            return obj[PLANE_SLICES["h"], PLANE_SLICES["h"]]

        @property
        def v(obj):
            return obj[PLANE_SLICES["v"], PLANE_SLICES["v"]]

        setattr(obj.__class__, "twiss", twiss)
        setattr(obj.__class__, "h", h)
        setattr(obj.__class__, "v", v)
        return obj

    def __array_finalize__(self, obj):
        # I don't what this does but I found this snippet in the numpy docs
        # and I'm scared to remove it
        if obj is None:  # pragma: no cover
            return
        # pylint: disable=attribute-defined-outside-init
        setattr(self.__class__, "twiss", getattr(obj.__class__, "twiss", None))
        setattr(self.__class__, "h", getattr(obj.__class__, "h", None))
        setattr(self.__class__, "v", getattr(obj.__class__, "v", None))
