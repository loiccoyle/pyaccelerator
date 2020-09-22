from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .elements.base import BaseElement

if TYPE_CHECKING:  # pragma: no cover
    from scipy.optimize import OptimizeResult

    from .lattice import Lattice


@dataclass
class Target:
    """Constraint target."""

    element: Union[str, "BaseElement"]
    value: Sequence[float]
    init: Union[str, Sequence[float]]
    plane: str = "h"

    def __post_init__(self):
        self.init = tuple(self.init)
        self.plane = self.plane.lower()
        self.value = np.array(self.value)
        if isinstance(self.element, BaseElement):
            self.element = self.element.name
        if len(self.init) != len(self.value):
            raise ValueError(
                f"Length of value: {repr(self.value)} does not match "
                f"length of init: {repr(self.init)}"
            )


@dataclass
class FreeParameter:
    """Constraint free parameter."""

    element: Union[str, "BaseElement"]
    attribute: Sequence[float]

    def __post_init__(self):
        if isinstance(self.element, BaseElement):
            self.element = self.element.name


class Constraints:
    """Match a lattice to constraints.

    Args:
        lattice: :py:class:`~accelerator.lattice.Lattice` instance on which to
            match.

    Examples:
        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a x coord of 10 meters:

            >>> lat = Lattice([Drift(1)])
            >>> lat.constraints.add_free_parameter(element="drift", attribute="l")
            >>> lat.constraints.add_target(element="drift", value=[10, None], init=[0, 1], plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=10, name='drift_0')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a beta of 5 meters from initial twiss parameters [1, 0, 1]:

            >>> lat = Lattice([Drift(1)])
            >>> lat.constraints.add_free_parameter("drift", "l")
            >>> lat.constraints.add_target("drift", [5, None, None], [1, 0, 1], "h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=2, name='drift_0')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a x coord of 5 meters after the first Drift:

            >>> lat = Lattice([Drift(1), Drift(1)])
            >>> lat.constraints.add_free_parameter("drift_0", "l")
            >>> lat.constraints.add_target("drift_0", [5, None], [0, 1], "h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=5, name='drift_0'), Drift(length=1, name='drift_1')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a x coord of 5 meters after the second Drift with equal lengths
        of both Drifts:

            >>> lat = Lattice([Drift(1), Drift(1)])
            >>> lat.constraints.add_free_parameter("drift", "l")
            >>> lat.constraints.add_target("drift_1", [5, None], [0, 1], "h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=2.5, name='drift_0'), Drift(length=2.5, name='drift_1')])

        Compute the :py:class:`~accelerator.elements.quadrupole.Quadrupole`
        strengths of a FODO cell to achieve a minimum beta of 0.5 meters:

            >>> lat = Lattice([Quadrupole(1.6), Drift(1), Quadrupole(-0.8),
            ... Drift(1), Quadrupole(1.6)])
            >>> lat.constraints.add_free_parameter("quad", "f")
            >>> lat.constraints.add_target("drift_0", [0.5, None, None], "solution", "h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Quadrupole(f=1.323..., name='quad_f'), Drift(length=1, name='drift_0'),
                     Quadrupole(f=-0.828..., name='quad_d'), Drift(1, name='drift_1'),
                     Quadrupole(f=1.323..., name='quad_f')])
    """

    # TODO: implement the twiss solution used in the last example

    def __init__(self, lattice: "Lattice"):
        self._lattice = lattice
        self.targets = []
        self.free_parameters = []

    def add_target(
        self,
        element: Union[str, "BaseElement"],
        value: Sequence[float],
        init: Union[str, Sequence[float]],
        plane: str = "h",
    ):
        """Add a constraint target.

        Args:
            element: Element name pattern or element instance at which the
                `value` should be achieved.
            value: If `value` is of length 2, phase space parameters are assumed.
                If `value` is of length 3, twiss parameters are assumed. Use
                None to ignore parameters.
            init: Initial parameters for tracking, must be of same length as
                `value`.
            plane: Plane of interest, either "h" or "v", defaults to "h".

        Examples:
            Adding a beta value target of 0.6 meters after a
            :py:class:`~accelerator.element.quadrupole.Quadrupole`:

                >>> quad = Quadrupole(0.8)
                >>> lat = Lattice([Drift(1), quad])
                >>> lat
                Lattice([Drift(l=1, name='drift_0'), Quadrupole(f=0.8, name='quadrupole_0')])
                >>> lat.constraints.add_target("quadrupole_0", [0.6, None, None], [1, 0 ,1])
                ... # or lat.constraints.add_target(quad, [0.6, None, None], [1, 0 ,1])
        """
        self.targets.append(Target(element, value, init, plane))

    def add_free_parameter(self, element: str, attribute: str):
        """Add a constraint free parameter.

        Args:
            element: Element name pattern or element instance for which the
                provided `attribute` will be considered a free parameter.
            attribute: attribute of `element`.

        Examples:
            Setting a :py:class:`~accelerator.element.drift.Drift`'s length as a
            free parameters:

                >>> drift = Drift(1)
                >>> lat = Lattice([drift])
                >>> lat
                Lattice([Drift(l=1, name='drift_0')])
                >>> lat.constraints.add_free_parameter("drift_0", "l")
                ... # or lat.constraints.add_free_parameter(drift, "l")
        """
        self.free_parameters.append(FreeParameter(element, attribute))

    def clear(self):
        """Clear the targets and free parameters."""
        self.targets.clear()
        self.free_parameters.clear()

    def match(
        self, *args, method="Nelder-Mead", **kwargs
    ) -> Tuple["Lattice", "OptimizeResult"]:
        """Match lattice properties to constraints using
        ``scipy.optimize.minimize``.

        Args:
            method: Scipy minimization method.
            *args: Passed to ``scipy.optimize.minimize``.
            **kwargs: Passed to ``scipy.optimize.minimize``.

        Raises:
            ValueError: If no targets or free parameters specified.

        Returns:
            New matched :py:class:`~accelerator.lattce.Lattice` instance and
            ``scipy.optimize.OptmizeResult``.
        """
        if self.targets == [] or self.free_parameters == []:
            raise ValueError("No targets or free parameters specified.")

        lattice = self._lattice.copy()
        root_start = self._get_initial(lattice)

        def match_function(new_settings):
            self._set_parameters(new_settings, lattice)
            transports = self._run_transports(lattice)
            out = []
            for target in self.targets:
                out.append(
                    self._transport_to_scalar(
                        target,
                        transports[target.plane][target.init],
                        lattice,
                    )
                )
            return np.mean(out)

        res = minimize(match_function, *args, x0=root_start, method=method, **kwargs)
        if res.success:
            self._set_parameters(res.x, lattice)
        return lattice, res

    def _run_transports(
        self, lattice: "Lattice"
    ) -> Dict[str, Dict[Tuple[float, float], np.ndarray]]:
        """Run all the required `Lattice.transport` calls."""
        # TODO: somewhere in this function add the twiss solution computation.
        inits = {target.init for target in self.targets}
        planes = {target.plane for target in self.targets}
        out = {}
        for plane in planes:
            if plane not in out.keys():
                out[plane] = {}
            for init in inits:
                *transported, _ = lattice.transport(init, plane=plane)
                out[plane][init] = np.vstack(transported)
        return out

    def _set_parameters(self, new_settings: Sequence[float], lattice: "Lattice"):
        """Set the new lattice settings."""
        for param, value in zip(self.free_parameters, new_settings):
            for i in lattice.search(param.element):
                setattr(lattice[i], param.attribute, value)

    def _get_initial(self, lattice: "Lattice") -> Sequence[float]:
        """Get the starting point for the minimization algorithm."""
        out = []
        for param in self.free_parameters:
            out.append(
                np.mean(
                    [
                        getattr(lattice[i], param.attribute)
                        for i in lattice.search(param.element)
                    ]
                )
            )
        return out

    @staticmethod
    def _transport_to_scalar(
        target: Target, transported: np.ndarray, lattice: "Lattice"
    ) -> float:
        """Compute a scalar value from the tranported parameters, for the given
        constraint.
        """
        transported_columns = [i + 1 for i in lattice.search(target.element)]
        transported_rows = [
            i for i, value in enumerate(target.value) if value is not None
        ]
        result = transported[transported_rows, transported_columns]
        return np.linalg.norm(result - target.value[transported_rows], 2)

    def __repr__(self):
        return f"Constraints({repr(self._lattice)})"
