"""Accelerator lattice"""
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from .beam import Beam
from .transfer_matrix import TransferMatrix
from .utils import compute_one_turn, to_phase_coord, to_twiss


class Lattice(list):
    """A list of accelerator elements."""

    def __init__(self, *args):
        """Looks like a list, smells like a list and tastes like a list.
        With a few added bells and whistles.

        Attributes:
            m_h: horizontal transfer matrix
            m_v: vertical transfer matrix.

        Methods:
            transport: transport either the phase space coords or the twiss
                parameters through the lattice.
            transport_beam: Same as `transport` but returns the phase space
                ellipses along with the twiss parameters through the lattice.
        """
        super().__init__(*args)
        self._m_h = None
        self._m_v = None

    @property
    def m_h(self):
        if self._m_h is None:
            self._m_h = TransferMatrix(
                compute_one_turn([element.m_h for element in self])
            )
        return self._m_h

    @property
    def m_v(self):
        if self._m_v is None:
            self._m_v = TransferMatrix(
                compute_one_turn([element.m_v for element in self])
            )
        return self._m_v

    def _clear_cache(self):
        self._m_h = None
        self._m_v = None

    def slice(self, element_type: Type["BaseElement"], n_element: int) -> "Lattice":
        """Slice the `element_type` elements of the `Lattice` into `n_element`.

        Args:
            element_type: element class to slice.
            n_element: slice `element_type` into `n_element` smaller elements.

        Return
            Sliced `Lattice`.
        """
        new_lattice = []
        for element in self:
            if isinstance(element, element_type) and element.length > 0:
                new_lattice.extend(element.slice(n_element))
            else:
                new_lattice.append(element)
        return Lattice(new_lattice)

    def transport(
        self,
        init: List[float],
        plane: str = "h",
        twiss: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transport the given phase space along the lattice.

        Args:
            init: list of phase space coordinates, position[m] and angle[rad],
                if `twiss` is True, `init` should be the initial
                twiss parameters a list [beta, alpha, gamma], one twiss
                parameter can be None.
            plane (optional): plane of interest.
            twiss (optional): if True will use the twiss parameter transfer
                matrices.

        Returns:
            vec, s: phase coordinates or twiss parameters along with the
                lattice and the s coordinates.
        """
        if twiss:
            init = to_twiss(init)
        else:
            init = to_phase_coord(init)
        out = [init]
        s = [0]
        transfer_ms = [getattr(element, "m_" + plane.lower()) for element in self]
        if twiss:
            transfer_ms = [m.twiss for m in transfer_ms]
        for i, m in enumerate(transfer_ms):
            out.append(m @ out[i])
            s.append(s[i] + self[i].length)
        return np.hstack(out), np.array(s)

    def transport_beam(
        self,
        twiss_init: List[float],
        beam: Beam,
        plane: str = "h",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the phase space ellipse along the lattice.

        Args:
            twiss_init: initial twiss parameters, as a list [beta, alpha,
                gamma], one twiss parameter can be None.
            beam: Beam instance.
            plane (optional): plane of interest.

        Returns:
            u, u_prime, twiss, s: phase space ellipse positions, phase space
                ellispe angles, twiss parameters along the lattice and
                the s coordinates.
        """
        twiss_init = to_twiss(twiss_init)

        def calc_phasespace(u):
            return beam.phasespace(u, plane=plane)

        init_phase = calc_phasespace(twiss_init)
        twiss = [twiss_init]
        u = [init_phase[0]]
        u_prime = [init_phase[1]]
        s = [0]
        transfer_ms = [getattr(element, "m_" + plane).twiss for element in self]
        for i, m in enumerate(transfer_ms):
            new_twiss = m @ twiss[i]
            phase_space = calc_phasespace(new_twiss)
            u.append(phase_space[0])
            u_prime.append(phase_space[1])
            twiss.append(new_twiss)
            s.append(s[i] + self[i].length)
        return np.vstack(u).T, np.vstack(u_prime).T, np.hstack(twiss), np.array(s)

    # Very ugly way of clearing cached one turn matrices on in place
    # modification of the sequence.
    def append(self, *args, **kwargs):
        self._clear_cache()
        return super().append(*args, **kwargs)

    def clear(self, *args, **kwargs):
        self._clear_cache()
        return super().clear(*args, **kwargs)

    def extend(self, *args, **kwargs):
        self._clear_cache()
        return super().extend(*args, **kwargs)

    def insert(self, *args, **kwargs):
        self._clear_cache()
        return super().insert(*args, **kwargs)

    def pop(self, *args, **kwargs):
        self._clear_cache()
        return super().pop(*args, **kwargs)

    def remove(self, *args, **kwargs):
        self._clear_cache()
        return super().remove(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        self._clear_cache()
        return super().reverse(*args, **kwargs)

    def sort(self, *args, **kwargs):
        self._clear_cache()
        return super().sort(*args, **kwargs)

    def __add__(self, other):
        return Lattice(list.__add__(self, other))

    def __mul__(self, other):
        return Lattice(list.__mul__(self, other))

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        try:
            return Lattice(result)
        except TypeError:
            return result

    def plot(
        self,
        n_s_per_element: int = int(1e3),
        xztheta_init: List[float] = [0, 0, np.pi / 2],
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot the s coordinate in the horizontal plane of the lattice.

        Args:
            n_s_per_element: number of steps along the s coord.
            xztheta_init: initial vector.

        Returns:
            Plotted Figure and array of axes.
        """
        xztheta = [np.array(xztheta_init)]
        s_start = 0
        for element in self:
            # skip thin elements
            if element.length == 0:
                continue
            d_s = element.length / n_s_per_element
            for _ in range(n_s_per_element):
                xztheta.append(xztheta[-1] + element._dxztheta_ds(xztheta[-1][2], d_s))
            s_start += element.length
        xztheta = np.vstack(xztheta)

        fig, ax = plt.subplots(1, 1)
        ax.plot(xztheta[:, 0], xztheta[:, 1], label="s")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.legend()
        return fig, ax
