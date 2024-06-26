"""Accelerator lattice"""

import json
import logging
import os
import re
from typing import TYPE_CHECKING, List, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.optimize import root

from .constraints import Constraints
from .harmonic_analysis import HarmonicAnalysis
from .transfer_matrix import TransferMatrix
from .utils import (
    PLANE_INDICES,
    PLANE_SLICES,
    TransportedPhasespace,
    TransportedTwiss,
    compute_one_turn,
    compute_twiss_solution,
    to_twiss,
)

if TYPE_CHECKING:  # pragma: no cover
    from .elements.base import BaseElement


class Lattice(list):
    """A lattice of accelerator elements.

    Looks like a list, smells like a list and tastes like a list.
    Is in fact an accelerator lattice.

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
        # non top level import to avoid circular imports
        from .elements.utils import deserialize

        with open(path, "r") as fp:
            serialized = json.load(fp)
        return cls([deserialize(element) for element in serialized])

    def __init__(self, *args):
        super().__init__(*args)
        self._m = None
        self.plot = Plotter(self)
        self.constraints = Constraints(self)
        self._log = logging.getLogger(__name__)

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
        return self.transport(self.closed_orbit_solution(dp, **solver_kwargs))

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
        self._log.info("Closed orbit optimization:\n %s", opt_res)
        solution = np.zeros(5)
        solution[4] = dp
        if opt_res.success:
            solution[:4] = opt_res.x
        else:
            raise ValueError("Failed to compute closed orbit solution.")
        return solution

    def dispersion(self, **solver_kwargs) -> TransportedPhasespace:
        """Compute the dispersion, i.e. the closed orbit for a particle with dp/p = 1.

        Args:
            **solver_kwargs: passed to `scipy.root`.

        Return:
            Dispersion solution transported through the lattice.
        """
        dp = 1e-3
        out = self.closed_orbit(dp=dp, **solver_kwargs)
        x = out.x / dp
        y = out.y / dp
        x_prime = out.x_prime / dp
        y_prime = out.y_prime / dp
        return TransportedPhasespace(out.s, x, x_prime, y, y_prime, out.dp)

    def dispersion_solution(self, **solver_kwargs):
        """Compute the dispersion solution.

        Args:
            **solver_kwargs: passed to `scipy.root`.

        Returns:
            Dispersion solution.
        """
        dp = 1e-3
        out = self.closed_orbit_solution(dp=dp, **solver_kwargs)
        out /= dp
        return out

    def twiss(self, plane="h") -> TransportedTwiss:
        """Compute the twiss parameters through the lattice for a given plane.

        Args:
            plane: plane of interest, either "h" or "v".

        Returns:
            Twiss parameters through the lattice.
        """
        plane = plane.lower()
        return self.transport_twiss(self.twiss_solution(plane=plane), plane=plane)

    def twiss_solution(self, plane: str = "h") -> np.ndarray:
        """Compute the twiss periodic solution.

        Args:
            plane: plane of interest, either "h" or "v".

        Returns:
            Twiss periodic solution.
        """
        plane = plane.lower()
        return compute_twiss_solution(self.m[PLANE_SLICES[plane], PLANE_SLICES[plane]])

    def tune(
        self, plane: str = "h", n_turns: int = 1024, dp: float = 0, tol=1e-4
    ) -> float:
        """Compute the fractional part of the tune.

        Note: the whole tune value would be Q = n + q or Q = n + (1 - q) with q
        the fractional part of the tune returned by this method and n an integer.

        Args:
            plane: plane of interest, either "h" or "v".
            n_turns: number of turns for which to track the particle, higher
                values lead to more precise values at the expense of computation
                time.
            dp: dp/p value of the tracked particle.
            tol: numerical tolerance for DC component.

        Returns:
            The fractional part of the tune.
        """
        init = np.zeros(5)
        init[PLANE_INDICES[plane]] = [1e-6, 0]
        init[4] = dp
        out_turns = [init]
        # track for n_turns
        for _ in range(n_turns - 1):
            _, *transported = self.transport(out_turns[-1])
            out_turns.append([point[-1] for point in transported])  # type: ignore
        out_turns_ = np.array(out_turns)
        # get the frequency with the highest amplitude
        position = out_turns_[:, PLANE_INDICES[plane][0]]
        angle = out_turns_[:, PLANE_INDICES[plane][1]]

        beta, alpha, _ = self.twiss_solution(plane=plane)
        sqrt_beta = np.sqrt(beta)

        norm_eta = position / sqrt_beta
        norm_eta_prime = position * alpha / sqrt_beta + sqrt_beta * angle

        complex_signal = norm_eta - 1j * norm_eta_prime

        tune, _ = HarmonicAnalysis(complex_signal).laskar_method(2)
        self._log.info("Harmonics: %s", tune)
        if abs(tune[0]) < tol:
            self._log.info("Dropped DC component.")
            # if there is a DC component to the signal then the tune will be the
            # second harmonic
            tune = tune[1]
        else:
            # if not then the tune will be the first harmonic
            tune = tune[0]
        return tune

    def chromaticity(self, plane: str = "h", delta_dp=1e-3, **kwargs) -> float:
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
        initial: ArrayLike,
    ) -> TransportedPhasespace:
        """Transport phase space coordinates or twiss parameters along the lattice.

        Args:
            initial: phase space coords to transport through the
                lattice.

        Returns:
            Transported phase space coords through the lattice.

        Examples:
            Transport phase space coords through a
            :py:class:`~accelerator.elements.drift.Drift`:

                >>> lat = Lattice([Drift(1)])
                >>> lat.transport(phasespace=[1, 1, 0, 0, 0])
                TransportedPhasespace(s=array([0, 1], x=array([1., 2.]), x_prime=array([1., 1.]), y=array([0, 0]), y_prime=array([0, 0]), dp=array([0., 0.]))

            Transport a distribution of phase space coordinates through the
            lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> transported = lat.transport(beam.match([1, 0, 1]))
                >>> plt.plot(tranported.s, transported.x)
                ...

            Transport a phase space ellipse's coordinates through the lattice:

                >>> beam = Beam()
                >>> lat = Lattice([Drift(1)])
                >>> transported = lat.transport(beam.ellipse([1, 0, 1]))
                >>> plt.plot(transported.x, transported.x_prime)
                ...
        """
        if not isinstance(initial, np.ndarray):
            initial = np.array(initial)
        out = [initial]
        s_coords = [0]
        for i, element in enumerate(self):
            post_element = element._transport(out[i])
            out.append(post_element)
            s_coords.append(s_coords[i] + element.length)
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

    def transport_twiss(
        self,
        twiss: Union[Sequence[Union[float, None]], np.ndarray],
        plane: str = "h",
    ) -> TransportedTwiss:
        """Transport the given twiss parameters along the lattice.

        Args:
            twiss: list of twiss parameters, beta[m], alpha[rad], and
                gamma[m^-1], one twiss parameter can be None.
            plane: plane of interest, either "h" or "v".

        Returns:
            Named tuple containing the twiss parameters along the lattice the
                coordinates along the ring.
        """
        out = [to_twiss(twiss)]  # type: ignore
        s_coords = [0]
        transfer_ms = [element.m.twiss(plane=plane) for element in self]
        for i, m in enumerate(transfer_ms):
            out.append(m @ out[i])
            s_coords.append(s_coords[i] + self[i].length)  # type: ignore
        out = np.hstack(out)
        return TransportedTwiss(np.array(s_coords), *out)

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
        pattern = re.compile(pattern)  # type: ignore
        out = [
            i
            for i, element in enumerate(self)
            if re.search(pattern, element.name, *args, **kwargs)
        ]
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
            # if the item is a slice return a lattice
            return Lattice(result)
        except TypeError:
            # if the item is not a slice return the element
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
            >>> lat.plot.layout()  # or lat.plot("layout")
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
    ) -> Tuple[Figure, Axes]:
        """Plot the s coordinate in the horizontal plane of the lattice.

        Args:
            n_s_per_element: Number of steps along the s coordinate for each
                element in the lattice.

        Returns:
            Plotted ``Figure`` and ``Axes``.
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

    def layout(self) -> Tuple[Figure, Axes]:
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
