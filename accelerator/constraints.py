from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .elements.base import BaseElement

if TYPE_CHECKING:  # pragma: no cover
    from scipy.optimize import OptimizeResult

    from .lattice import Lattice


class Target:
    """Constraint target.

    Args:
        element: Element name pattern or element instance at which the
            `value` should be achieved.
        value: If `value` is of length 2, phase space parameters are assumed.
            If `value` is of length 3, twiss parameters are assumed. Use
            None to ignore parameters.
        phasespace (optional): Initial phase space coords, a sequence of
            u[m], u_prime[rad], dp/p.
        twiss (optional): Initial twiss parameters, a sequence of beta[m],
            alpha[rad], gamma[m^-1]. If "solution" is provided or if
            neither `phasespace` nor `twiss` is provided, the twiss
            periodic solution is computed and used for the transport.
        plane: Plane of interest, either "h" or "v", defaults to "h".
    """

    def __init__(
        self,
        element: Union[str, "BaseElement"],
        value: Sequence[float],
        twiss: Optional[Union[str, Sequence[float]]] = None,
        phasespace: Optional[Sequence[float]] = None,
        plane: str = "h",
    ):
        if isinstance(twiss, str):
            if twiss != "solution":
                raise ValueError(
                    f"Unrecognized twiss string: '{twiss}'. Did you mean 'solution'?"
                )
        elif twiss is not None:
            twiss = tuple(twiss)
        if phasespace is not None:
            phasespace = tuple(phasespace)

        plane = plane.lower()
        value = np.array(value)
        if isinstance(element, BaseElement):
            element = element.name
        self.element = element
        self.value = value
        self.twiss = twiss
        self.phasespace = phasespace
        self.plane = plane.lower()

    def __repr__(self) -> str:
        args = ["element", "value", "twiss", "phasespace", "plane"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"Target({arg_string})"


class FreeParameter:
    """Constraint free parameter.

    Args:
        element: Element name pattern or element instance for which the
            provided `attribute` will be considered a free parameter.
        attribute: attribute of `element`.
    """

    def __init__(self, element: Union[str, "BaseElement"], attribute: str):
        if isinstance(element, BaseElement):
            element = element.name
        self.element = element
        self.attribute = attribute

    def __repr__(self) -> str:
        args = ["element", "attribute"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"FreeParameter({arg_string})"


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
            >>> lat.constraints.add_target(element="drift", value=[10, None, None], phasespace=[0, 1, 0], plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=10, name='drift_0')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a beta of 5 meters from initial twiss parameters [1, 0, 1]:

            >>> lat = Lattice([Drift(1)])
            >>> lat.constraints.add_free_parameter("drift", "l")
            >>> lat.constraints.add_target("drift", [5, None, None], twiss=[1, 0, 1], plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=2, name='drift_0')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a x coord of 5 meters after the first Drift:

            >>> lat = Lattice([Drift(1), Drift(1)])
            >>> lat.constraints.add_free_parameter("drift_0", "l")
            >>> lat.constraints.add_target("drift_0", [5, None, None], phasespace=[0, 1, 0], plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=5, name='drift_0'), Drift(length=1, name='drift_1')])

        Compute :py:class:`~accelerator.elements.drift.Drift` length to
        reach a x coord of 5 meters after the second Drift with equal lengths
        of both Drifts:

            >>> lat = Lattice([Drift(1), Drift(1)])
            >>> lat.constraints.add_free_parameter("drift", "l")
            >>> lat.constraints.add_target("drift_1", [5, None, None], phasespace=[0, 1, 0], plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([Drift(length=2.5, name='drift_0'), Drift(length=2.5, name='drift_1')])

        Compute the :py:class:`~accelerator.elements.quadrupole.Quadrupole`
        strengths of a FODO cell to achieve a minimum beta of 0.5 meters:

            >>> lat = Lattice([QuadrupoleThin(1.6, name='quad_f'), Drift(1), QuadrupoleThin(-0.8, name='quad_d'),
            ... Drift(1), QuadrupoleThin(1.6, name='quad_f')])
            >>> lat.constraints.add_free_parameter("quad_f", "f")
            >>> lat.constraints.add_free_parameter("quad_d", "f")
            >>> lat.constraints.add_target("quad_d", [0.5, None, None], twiss="solution", plane="h")
            >>> matched_lat, _ = lat.constraints.match()
            >>> matched_lat
            Lattice([QuadrupoleThin(f=1.319, name='quad_f'), Drift(length=1, name='drift_0'),
                     QuadrupoleThin(f=-0.918, name='quad_d'), Drift(1, name='drift_1'),
                     QuadrupoleThin(f=1.319, name='quad_f')])
    """

    def __init__(self, lattice: "Lattice"):
        self._lattice = lattice
        self.targets = []
        self.free_parameters = []

    def add_target(
        self,
        element: Union[str, "BaseElement"],
        value: Sequence[float],
        twiss: Union[str, Sequence[float]] = None,
        phasespace: Sequence[float] = None,
        plane: str = "h",
    ):
        """Add a constraint target.

        Args:
            element: Element name pattern or element instance at which the
                `value` should be achieved.
            value: If `value` is of length 2, phase space parameters are assumed.
                If `value` is of length 3, twiss parameters are assumed. Use
                None to ignore parameters.
            phasespace (optional): Initial phase space coords, a sequence of
                u[m], u_prime[rad], dp/p.
            twiss (optional): Initial twiss parameters a sequence of beta[m],
                alpha[rad], gamma[m^-1]. If "solution" is provided or if
                neither `phasespace` nor `twiss` is provided, the twiss
                periodic solution is computed and used for the transport.
            plane: Plane of interest, either "h" or "v", defaults to "h".

        Examples:
            Adding a beta value target of 0.6 meters after a
            :py:class:`~accelerator.element.quadrupole.Quadrupole`:

                >>> quad = QuadrupoleThin(0.8)
                >>> lat = Lattice([Drift(1), quad])
                >>> lat
                Lattice([Drift(l=1, name='drift_0'), QuadrupoleThin(f=0.8, name='quad_0')])
                >>> lat.constraints.add_target("quad_0", [0.6, None, None], twiss=[1, 0 ,1])
                ... # or lat.constraints.add_target(quad, [0.6, None, None], twiss=[1, 0 ,1])
        """
        self.targets.append(
            Target(element, value, twiss=twiss, phasespace=phasespace, plane=plane)
        )

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

    def match(self, *args, **kwargs) -> Tuple["Lattice", "OptimizeResult"]:
        """Match lattice properties to constraints using
        ``scipy.optimize.minimize``.

        Args:
            *args: Passed to ``scipy.optimize.minimize``.
            **kwargs: Passed to ``scipy.optimize.minimize``.

        Raises:
            ValueError: If no targets or free parameters specified.

        Returns:
            New matched :py:class:`~accelerator.lattce.Lattice` instance and
            ``scipy.optimize.OptmizeResult``.
        """
        if "constraints" in kwargs.keys():
            # use the scipy constraint default
            default_method = None
        else:
            # set the non constrained minimization method to Nelder Mead
            default_method = "Nelder-Mead"

        if self.targets == []:
            raise ValueError("No targets specified.")
        if self.free_parameters == []:
            raise ValueError("No free parameters specified.")

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
                        transports[target.plane][(target.twiss, target.phasespace)],
                        lattice,
                    )
                )
            # mean of the norm of the targets
            return np.mean(out)

        res = minimize(
            match_function, *args, x0=root_start, method=default_method, **kwargs
        )
        if res.success:
            # sometimes the last iteration is not the minimum, set the real
            # solution
            self._set_parameters(res.x, lattice)
        return lattice, res

    def _run_transports(
        self, lattice: "Lattice"
    ) -> Dict[str, Dict[Tuple[float, float], np.ndarray]]:
        """Run all the required `Lattice.transport` calls."""
        # create a set of unique combinations of inputs for lattice.tranport
        # to not run any transport twice
        inits_planes = {
            (target.twiss, target.phasespace, target.plane) for target in self.targets
        }
        out = {}
        for twiss, phasespace, plane in inits_planes:
            if plane not in out.keys():
                out[plane] = {}
            try:
                # ignore the s coord
                _, *transported = lattice.transport(
                    twiss=twiss, phasespace=phasespace, plane=plane
                )
                transported = np.vstack(transported)
            except ValueError:
                transported = None
            out[plane][(twiss, phasespace)] = transported

        return out

    def _set_parameters(self, new_settings: Sequence[float], lattice: "Lattice"):
        """Set the new lattice settings."""
        for param, value in zip(self.free_parameters, new_settings):
            for i in lattice.search(param.element):
                setattr(lattice[i], param.attribute, value)
        # TODO: decide if we keep the caching of the one turn matrices?
        lattice._clear_cache()

    def _get_initial(self, lattice: "Lattice") -> Sequence[float]:
        """Get the starting point for the minimization algorithm."""
        out = []
        for param in self.free_parameters:
            out.append(
                # this mean might cause issues, maybe switch to the median ?
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
        if transported is None:
            # is None when 'solution' was specified but the lattice
            # has no twiss periodic solution.
            return np.inf
        transported_columns = [i + 1 for i in lattice.search(target.element)]
        transported_rows = [
            i for i, value in enumerate(target.value) if value is not None
        ]
        result = transported[transported_rows, transported_columns]
        # l-2 norm
        return np.linalg.norm(result - target.value[transported_rows], 2)

    def __repr__(self):
        return f"Constraints({repr(self._lattice)})"
