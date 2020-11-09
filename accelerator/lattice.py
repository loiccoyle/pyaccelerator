# TODO: redo the docstrings
"""Accelerator lattice"""
import json
import os
import re
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, root

from .constraints import Constraints
from .transfer_matrix import TransferMatrix
from .utils import (
    TransportedPhasespace,
    TransportedTwiss,
    compute_one_turn,
    compute_twiss_solution,
    to_twiss,
    PLANE_INDICES,
    PLANE_SLICES,
)

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
        self._m = None
        self.plot = Plotter(self)
        self.constraints = Constraints(self)

    @property
    def m(self):
        if self._m is None:
            self._m = TransferMatrix(compute_one_turn([element.m for element in self]))
        return self._m

    def _clear_cache(self):
        self._m = None

    def closed_orbit(self, dp: float, **solver_kwargs) -> TransportedPhasespace:
        """Compute the closed orbit for a given dp/p.

        Args:
            dp: dp/p for which to compute the closed orbit.
            **solver_kwargs: passed to `scipy.root`.

        Returns:
            Closed orbit solution transported through the lattice.
        """
        return self._transport_phasespace(
            *self.closed_orbit_solution(dp, **solver_kwargs)
        )

    def closed_orbit_solution(self, dp: float, **solver_kwargs) -> np.ndarray:
        """Compute the closed orbit solution for a given dp/p.

        Args:
            dp: dp/p for which to compute the closed orbit.
            **solver_kwargs: passed to `scipy.root`.

        Returns:
            Closed orbit solution.
        """

        def try_solve(x_x_prime_y_y_prime):
            init = np.zeros(5)
            init[:4] = x_x_prime_y_y_prime
            init[4] = dp
            _, *transported = self.transport(init)
            out = [point[-1] for point in transported]
            return (init - out)[:4]

        opt_res = root(try_solve, [0, 0, 0, 0], **solver_kwargs)
        print(opt_res)
        solution = np.zeros(5)
        solution[4] = dp
        if opt_res.success:
            solution[:4] = opt_res.x
        else:
            raise ValueError("Failed to compute dispersion solution.")
        return solution

    def dispersion(self, **solver_kwargs) -> TransportedPhasespace:
        """Compute the dispersion, i.e. the closed orbit for a particle with dp/p = 1.

        Args:
            **solver_kwargs: passed to `scipy.root`.

        Return:
            Dispersion solution transported through the lattice.
        """
        dp = 0.1
        out = self.closed_orbit(dp=dp, **solver_kwargs)
        x = out.x / dp
        y = out.y / dp
        return TransportedPhasespace(out.s, x, out.x_prime, y, out.y_prime, out.dp)

    def twiss(self, plane="h") -> TransportedTwiss:
        """Compute the twiss parameters through the lattice for a given plane.

        Args:
            plane: plane of interest, either "h" or "v".

        Returns:
            Twiss parameters through the lattice.
        """
        plane = plane.lower()
        return self._transport_twiss(self.twiss_solution(plane=plane), plane=plane)

    def twiss_solution(self, plane: str = "h") -> np.ndarray:
        """Compute the twiss periodic solution.

        Args:
            plane: plane of interest, either "h" or "v".

        Returns:
            Twiss periodic soluton.
        """
        plane = plane.lower()
        return compute_twiss_solution(self.m[PLANE_SLICES[plane], PLANE_SLICES[plane]])

    def tune(self, plane: str = "h", n_turns: int = 1028, dp: float = 0) -> float:
        """Compute the fractional part of the tune.

        Note: the whole tune value would be Q = n + q or Q = n + (1 - q) with q
        the fractional part of the tune returned by this method and n an integer.

        Args:
            plane: plane of interest, either "h" or "v".
            n_turns: number of turns for which to track the particle, higher
                values lead to more precise values at the expense of computation
                time.
            dp: dp/p value of the tracked particle.

        Returns:
            The fractional part of the tune.
        """
        init = np.zeros(5)
        init[PLANE_INDICES[plane]] = [1e-9, 0]
        init[4] = dp
        out_turns = [init]
        # track for n_turns
        for _ in range(n_turns - 1):
            _, *transported = self.transport(out_turns[-1])
            out_turns.append([point[-1] for point in transported])
        out_turns = np.array(out_turns)
        # get the frequency with the highest amplitude
        position = out_turns[:, PLANE_INDICES[plane][0]]
        # remove the very first frequency and amplitude as for some reason
        # the 0 frequency can have a very high amplitude, I guess it is the
        # constant component
        pos_fft = abs(np.fft.rfft(position))[1:]
        freqs = np.fft.rfftfreq(n_turns)[1:]
        print(pos_fft.shape)
        print(len(freqs))
        plt.plot(freqs, pos_fft, marker=".", linewidth=0)
        plt.yscale("log")
        plt.show()
        plt.plot(position)
        plt.show()
        plt.scatter(position, out_turns[:, PLANE_INDICES[plane][1]], s=0.1)
        plt.show()
        return freqs[np.argmax(pos_fft)]

    def chromaticity(self, plane: str = "h", delta_dp=0.1, **kwargs) -> float:
        """Compute the chromaticity. Tracks 2 particles with different dp/p and
        computes the chromaticity from the tune change.

        Args:
            plane: plane of interest, either "h" of "v".
            delta_dp: dp/p difference between the 2 particles.
            **kwargs: passed to the compute tune method.

        Returns:
            Chromaticity value.
        """
        tune_0 = self.tune(plane=plane, dp=0, **kwargs)
        tune_1 = self.tune(plane=plane, dp=delta_dp, **kwargs)
        print(tune_0)
        print(tune_1)
        # if tune_1 > tune_0:
        #     tune_1 = 1 -
        return (tune_1 - tune_0) / delta_dp

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
        initial: Optional[Sequence[Union[float, np.ndarray]]] = None,
    ) -> TransportedPhasespace:
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
            If `phasespace` is provided, returns a named tuple containing the
            coordinates along the lattice, phase space position, angle and
            dp/p, named 's', 'u', 'u_prime' and 'dp' respectively.

            If `phasespace` is a distribution of phase space coordinates,
            returns a named tuple containing the coordinate along the lattice,
            the position distribution, the angle distribution, the dp/p
            distribution and , named 's', 'u', 'u_prime', and 'dp' respectively.

            If a `twiss` is provided, returns a named tuple containing the
            coordinate along the lattice and the twiss parameters, beta, alpha,
            gamma, named 's', 'beta', 'alpha', and 'gamma' respectively.


        Examples:
            Transport phase space coords through a
            :py:class:`~accelerator.elements.drift.Drift`:

                >>> lat = Lattice([Drift(1)])
                >>> lat.transport(phasespace=[1, 1, 0])
                TransportedPhasespace(s=array([0, 1], u=array([1., 2.]), u_prime=array([1., 1.]), dp=array([0., 0.]))

            Transport twiss parameters through a
            :py:class:`~accelerator.elements.drift.Drift`:

                >>> lat = Lattice([Drift(1)])
                >>> lat.transport(twiss=[1, 0, 1])
                TransportedTwiss(s=array([0, 1]), beta=array([1., 2.]), alpha=array([ 0., -1.]), gamma=array([1., 1.]))

            Transport a distribution of phase space coordinates through the
            lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> tranported = lat.transport(phasespace=beam.match([1, 0, 1]))
                >>> plt.plot(tranported.s, tranported.u)
                ...

            Transport a phase space ellipse's coordinates through the lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> tranported = lat.transport(phasespace=beam.ellipse([1, 0, 1]))
                >>> plt.plot(tranported.u, tranported.u_prime)
                ...
        """
        # TODO: clean this up
        return self._transport_phasespace(*initial)

    def _transport_twiss(
        self,
        twiss: Sequence[float],
        plane: str = "h",
    ) -> TransportedTwiss:
        """Transport the given twiss parameters along the lattice.

        Args:
            twiss: list of twiss parameters, beta[m], alpha[rad], and
                gamma[m^-1], one twiss parameter can be None.
            plane: plane of interest, defaults to "h".

        Returns:
            named tuple containing 's', '*coords', with 'coords' the phase
                space coordinates ('u', 'u_prime', 'dp') or the twiss
                parameters, depending on the `twiss` flag. along the lattice and
                's' the coordinates along the ring.
        """
        twiss = to_twiss(twiss)
        out = [twiss]
        s_coords = [0]
        # transfer_matrix = "m_" + plane
        transfer_ms = [element.m.twiss(plane=plane) for element in self]
        for i, m in enumerate(transfer_ms):
            out.append(m @ out[i])
            s_coords.append(s_coords[i] + self[i].length)
        out = np.hstack(out)
        return TransportedTwiss(np.array(s_coords), *out)

    def _transport_phasespace(
        self,
        x: np.ndarray,
        x_prime: np.ndarray,
        y: np.ndarray,
        y_prime: np.ndarray,
        dp: np.ndarray,
    ) -> TransportedPhasespace:
        """Transport a distribution of in phase space along the lattice.

        Args:
            u: phase space position[m] coordinates, 1D array same length as
                `u_prime`.
            u_prime: phase space angle[rad] coordinate, 1D array same length as
                `u`.
            plane: plane of interest, either "h" or "v".

        Returns:
            named tuple containing 's' the coordinate along the lattice, 'u'
                the position array and 'u_prime' the angle array and 'dp' the
                momentum deviation.
        """
        coords = np.vstack([x, x_prime, y, y_prime, dp])
        out = [coords]
        s_coords = [0]
        transfer_ms = [(element.m, element._non_linear_term) for element in self]
        for i, (transfer_m, non_linear) in enumerate(transfer_ms):
            # for most elements there are no non linear effects
            # print('non_linear')
            # print(out[i])
            print(non_linear(out[i]))
            post_element = (transfer_m @ out[i]) + non_linear(out[i])
            out.append(post_element)
            s_coords.append(s_coords[i] + self[i].length)
        x_coords, x_prime_coords, y_coords, y_prime_coords, dp_coords = zip(*out)
        x_coords = np.vstack(x_coords).squeeze().T
        x_prime_coords = np.vstack(x_prime_coords).squeeze().T
        y_coords = np.vstack(y_coords).squeeze().T
        y_prime_coords = np.vstack(y_prime_coords).squeeze().T
        dp_coords = np.vstack(dp_coords).squeeze().T
        return TransportedPhasespace(
            np.array(s_coords),
            x_coords,
            x_prime_coords,
            y_coords,
            y_prime_coords,
            dp_coords,
        )

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
            >>> lat.plot.lattice()  # or lat.plot("layout")
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

    def layout(self) -> Tuple[plt.Figure, plt.Axes]:
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

    def __call__(self, *args, plot_type="layout", **kwargs):
        return getattr(self, plot_type)(*args, **kwargs)

    def __repr__(self):
        return f"Plotter({repr(self._lattice)})"
