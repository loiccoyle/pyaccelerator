"""Accelerator lattice"""
import json
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from .constraints import Constraints
from .transfer_matrix import TransferMatrix
from .utils import compute_one_turn, to_phase_coord, to_twiss

if TYPE_CHECKING:  # pragma: no cover
    from .elements.base import BaseElement


class Lattice(list):
    """A lattice of accelerator elements.

    Looks like a list, smells like a list and tastes like a list.
    Is infact an accelerator lattice.

    Examples:
        Create a simple lattice.

           >>> Lattice([Drift(1), QuadrupoleThin(0.8)])
           Lattice([Drift(l=1, name="drift_0"), QuadrupoleThin(f=0.8, name="quadrupole_thin_0")])
    """

    @classmethod
    def load(cls, path: os.PathLike) -> "Lattice":
        """Load a lattice from a file.

        Args:
            path: File path.

        Returns:
            Loaded :py:class:`Lattice` instance.

        Examples:
            Save and load a lattice:

                >>> lat = Lattice([Drift(1)])
                >>> lat.save("drift.json")
                >>> lat_loaded = Lattice.load("drift.json")
        """
        # TODO: non top level import to avoid circular imports
        from .elements.utils import deserialize

        with open(path, "r") as fp:
            serialized = json.load(fp)
        return cls([deserialize(element) for element in serialized])

    def __init__(self, *args):
        super().__init__(*args)
        self._m_h = None
        self._m_v = None
        self.plot = Plotter(self)
        self.constraints = Constraints(self)

    @property
    def m_h(self):
        """Horizontal :py:class:`~accelerator.transfer_matrix.TransferMatrix` of the lattice."""
        if self._m_h is None:
            self._m_h = TransferMatrix(
                compute_one_turn([element.m_h for element in self])
            )
        return self._m_h

    @property
    def m_v(self):
        """Vertical :py:class:`~accelerator.transfer_matrix.TransferMatrix` of the lattice."""
        if self._m_v is None:
            self._m_v = TransferMatrix(
                compute_one_turn([element.m_v for element in self])
            )
        return self._m_v

    def _clear_cache(self):
        self._m_h = None
        self._m_v = None

    def slice(self, element_type: Type["BaseElement"], n_element: int) -> "Lattice":
        """Slice the `element_type` elements of the lattice into `n_element`.

        Args:
            element_type: Element class to slice.
            n_element: Slice `element_type` into `n_element` smaller elements.

        Returns:
            Sliced :py:class:`Lattice`.

        Examples:
            Slice the :py:class:`~accelerator.elements.drift.Drift` elements
            into 2:

                >>> lat = Lattice([Drift(1), QuadrupoleThin(0.8)])
                >>> lat.slice(Drift, 2)
                Lattice([Drift(l=0.5, name="drift_0_slice_0"),
                         Drift(l=0.5, name="drift_0_slice_1"),
                         Quadrupole(f=0.8, name="quadrupole_thin_0")])
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
        phasespace: Optional[Sequence[Union[float, np.ndarray]]] = None,
        twiss: Optional[Union[str, Sequence[Union[float, np.ndarray]]]] = None,
        plane: str = "h",
    ) -> Tuple[np.ndarray, ...]:
        """Transport phase space coordinates or twiss parameters along the lattice.

        Args:
            phasespace (optional): phase space coords to transport through
                the lattice, a sequence of u[m], u_prime[rad], dp/p.
            twiss (optional): twiss parameters to transport through the
                lattice, a sequence of beta[m], alpha[rad], gamma[m^-1].
                If "solution" is provided or if neither `phasespace` nor
                `twiss` is provided, the twiss periodic solution is computed
                and used for the transport.
            plane: the plane of interest, either "h" or "v", defaults
                to "h".

        Raises:
            ValueError: if both `phasespace` and `twiss` are provided.
            ValueError: if the twiss solution computation fails.

        Returns:
            If `phasespace` is provided, returns the phase space position, angle, dp/p
            and s coordinates along the lattice.

            If `phasespace` is a distribution of phase space coordinates,
            returns the position distribution, the angle distribution, the dp/p
            distribution and the s coordinate along the lattice.

            If a `twiss` is provided, returns the twiss parameters, beta,
            alpha, gamma and the s coordinate along the lattice.


        Examples:
            Transport phase space coords through a
            :py:class:`~accelerator.elements.drift.Drift`:

                >>> lat = Lattice([Drift(1)])
                >>> lat.transport(phasespace=[1, 1, 0])
                (array([1., 2.]), array([1., 1.]), array([0., 0.]), array([0, 1]))

            Transport twiss parameters through a
            :py:class:`~accelerator.elements.drift.Drift`:

                >>> lat = Lattice([Drift(1)])
                >>> lat.transport(twiss=[1, 0, 1])
                (array([1., 2.]), array([ 0., -1.]), array([1., 1.]), array([0, 1]))

            Transport a distribution of phase space coordinates through the
            lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> u, u_prime, dp, s = lat.transport(phasespace=beam.match([1, 0, 1]))
                >>> plt.plot(s, u)
                ...

            Transport a phase space ellipse's coordinates through the lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> u, u_prime, dp, s = lat.transport(phasespace=beam.ellipse([1, 0, 1]))
                >>> plt.plot(u, u_prime)
                ...
        """
        # TODO: the _transport and the _transport_distribution share a lot of
        # code they could easily be merged.
        plane = plane.lower()
        if twiss is not None and phasespace is not None:
            raise ValueError("Provide either 'twiss' or 'phasespace'.")
        if (isinstance(twiss, str) and twiss == "solution") or (
            twiss is None and phasespace is None
        ):
            twiss = getattr(self, "m_" + plane).twiss_solution
            if twiss is None:
                raise ValueError("Lattice has no periodic twiss solution.")

        twiss_bool = twiss is not None

        if twiss_bool:
            # tranporting twiss
            return self._transport(twiss, plane=plane, twiss=twiss_bool)
        else:
            if all([isinstance(v, Iterable) and len(v) > 1 for v in phasespace]):
                # distribution of phase space coords
                return self._transport_distribution(*phasespace, plane=plane)
            else:
                # transport phase space coords
                return self._transport(phasespace, plane=plane, twiss=twiss_bool)

    def _transport(
        self,
        init: Sequence[float],
        plane: str = "h",
        twiss: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transport the given phase space along the lattice.

        Args:
            init: list of phase space coordinates, position[m] and angle[rad],
                if `twiss` is True, `init` should be the initial twiss
                parameters a list [beta, alpha, gamma], one twiss parameter can
                be None.
            plane: plane of interest, defaults to "h".
            twiss: If True will use the twiss parameter transfer matrices,
                defaults to False.

        Returns:
            `*coords`, `s`, with `coords` the phase space coordinates or twiss
                parameters, depending on the `twiss` flag. along the lattice and
                `s` the s coordinates.
        """
        if twiss:
            init = to_twiss(init)
        else:
            init = to_phase_coord(init)
        out = [init]
        s_coords = [0]
        transfer_matrix = "m_" + plane
        transfer_ms = [getattr(element, transfer_matrix) for element in self]
        if twiss:
            transfer_ms = [m.twiss for m in transfer_ms]
        for i, m in enumerate(transfer_ms):
            out.append(m @ out[i])
            s_coords.append(s_coords[i] + self[i].length)
        out = np.hstack(out)
        return tuple([*out] + [np.array(s_coords)])

    def _transport_distribution(
        self,
        u: np.ndarray,
        u_prime: np.ndarray,
        dp: np.ndarray,
        plane: str = "h",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transport a distribution of in phase space along the lattice.

        Args:
            u: phase space position[m] coordinates, 1D array same length as `u_prime`.
            u_prime: phase space angle[rad] coordinate, 1D array same length as `u`.
            plane: plane of interest, either "h" or "v".

        Returns:
            `u_coords`, `u_prime_coords`, `s`, with `u_coords` the position
                array and `u_prime_coords` the angle array, both of shape
                (len(`u`), number of elements in the lattice + 1) and `s` the s
                coordinate along the lattice.
        """
        coords = np.vstack([u, u_prime, dp])
        out = [coords]
        s_coords = [0]
        transfer_matrix = "m_" + plane
        transfer_ms = [getattr(element, transfer_matrix) for element in self]
        for i, m in enumerate(transfer_ms):
            out.append(m @ out[i])
            s_coords.append(s_coords[i] + self[i].length)
        u_coords, u_prime_coords, dp_coords = zip(*out)
        u_coords = np.vstack(u_coords).T
        u_prime_coords = np.vstack(u_prime_coords).T
        dp_coords = np.vstack(dp_coords).T
        return u_coords, u_prime_coords, dp_coords, np.array(s_coords)

    def search(self, pattern: str, *args, **kwargs) -> List[int]:
        """Search the lattice for elements with `name` matching the pattern.

        Args:
            pattern: RegEx pattern.
            *args: Passed to ``re.search``.
            **kwargs: Passed to ``re.search``.

        Raises:
            ValueError: If not elements match the provided pattern.

        Return:
            List of indexes in the lattice where the element's name matches the pattern.
        """
        pattern = re.compile(pattern)
        out = [i for i, element in enumerate(self) if re.search(pattern, element.name)]
        if not out:
            raise ValueError(f"'{pattern}' does not match with any elements in {self}")
        return out

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

    # Disable sorting
    # TODO: is there a way to remove the method altogether?
    def sort(self, *args, **kwargs):
        """DISABLED."""

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

    def save(self, path: os.PathLike):
        """Save a lattice to file.

        Args:
            path: File path.

        Examples:
            Save a lattice:

                >>> lat = Lattice([Drift(1)])
                >>> lat.save('drift.json')
        """
        serializable = [element._serialize() for element in self]
        with open(path, "w") as fp:
            json.dump(serializable, fp, indent=4)

    def copy(self, deep=True) -> "Lattice":
        """Create a copy of the lattice.

        Args:
            deep: If True create copies of the elements themselves.

        Returns:
            A copy of the lattice.
        """
        if deep:
            return Lattice([element.copy() for element in self])
        return Lattice(self)

    def __repr__(self):
        return f"Lattice({super().__repr__()})"


class Plotter:
    """Lattice plotter.

    Args:
        Lattice: :py:class:`Lattice` instance.

    Examples:
        Plot a lattice:

            >>> lat = Lattice([QuadrupoleThin(-0.6), Drift(1), QuadrupoleThin(0.6)])
            >>> lat.plot.lattice()  # or lat.plot("lattice")
            ...

        Plot the top down view of the lattice:

            >>> lat = Lattice([Drift(1), Dipole(1, np.pi/2)])
            >>> lat.plot.top_down()  # or lat.plot("top_down")
            ...
    """

    def __init__(self, lattice: Lattice):
        self._lattice = lattice

    def top_down(
        self,
        n_s_per_element: int = int(1e3),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the s coordinate in the horizontal plane of the lattice.

        Args:
            n_s_per_element: Number of steps along the s coordinate for each
                element in the lattice.

        Returns:
            Plotted ``plt.Figure`` and ``plt.Axes``.
        """
        xztheta = [np.array([0, 0, np.pi / 2])]
        s_start = 0
        for element in self._lattice:
            if element.length == 0:
                # thin elements don't waste time on slicing them and running
                # this many times
                xztheta.append(xztheta[-1] + element._dxztheta_ds(xztheta[-1][2], 0))
            else:
                d_s = element.length / n_s_per_element
                for _ in range(n_s_per_element):
                    xztheta.append(
                        xztheta[-1] + element._dxztheta_ds(xztheta[-1][2], d_s)
                    )
            s_start += element.length
        xztheta = np.vstack(xztheta)

        fig, ax = plt.subplots(1, 1)
        ax.plot(xztheta[:, 0], xztheta[:, 1], label="s")
        # forcefully adding margins, this might cause issues
        if xztheta[:, 0].max() - xztheta[:, 0].min() < 0.1:
            ax.set_xlim((-1, 1))
        if xztheta[:, 1].max() - xztheta[:, 1].min() < 0.1:
            ax.set_ylim((-1, 1))
        ax.set_aspect("equal")
        ax.margins(0.05)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.legend()
        return fig, ax

    def lattice(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the lattice.

        Returns:
            Plotted ``plt.Figure`` and ``plt.Axes``.
        """
        fig, ax = plt.subplots(1, 1)

        s_coord = 0
        for element in self._lattice:
            patch = element._get_patch(s_coord)
            s_coord += element.length
            # skip elements which don't have a defined patch
            if patch is None:
                continue
            ax.add_patch(patch)

        ax.hlines(0, 0, s_coord, color="tab:gray", ls="dashed")
        ax.axes.yaxis.set_visible(False)
        ax.margins(0.05)
        ax.set_xlabel("s [m]")
        # remove duplicates from the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_indexes = sorted([labels.index(label) for label in set(labels)])
        new_handles = [handles[i] for i in unique_indexes]
        new_labels = [labels[i] for i in unique_indexes]
        ax.legend(
            handles=new_handles,
            labels=new_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        return fig, ax

    def __call__(self, *args, plot_type="lattice", **kwargs):
        return getattr(self, plot_type)(*args, **kwargs)

    def __repr__(self):
        return f"Plotter({repr(self._lattice)})"
