from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .elements.base import BaseElement
from .utils import PLANE_INDICES

if TYPE_CHECKING:  # pragma: no cover
    from scipy.optimize import OptimizeResult

    from .lattice import Lattice


class BaseTarget:
    @abstractmethod
    def loss(self, lattice: "Lattice"):
        pass


class TargetPhasespace(BaseTarget):
    """Constraint target.

    Args:
        element: Element name pattern or element instance at which the
            `value` should be achieved.
        value: Target value of twiss parameters or phase space coordinates
            at the given `element`.
        initial (optional): Initial phase space coords, a sequence of
            u[m], u_prime[rad], dp/p.
        # twiss (optional): Initial twiss parameters, a sequence of beta[m],
        #     alpha[rad], gamma[m^-1]. If "solution" is provided or if
        #     neither `phasespace` nor `twiss` is provided, the twiss
        #     periodic solution is computed and used for the transport.
    """

    def __init__(
        self,
        element: Union[str, "BaseElement"],
        value: Sequence[float],
        initial: Optional[Sequence[float]] = None,
    ):
        if initial is not None:
            initial = tuple(initial)

        value = np.array(value)
        if isinstance(element, BaseElement):
            element = element.name
        self.element = element
        self.value = value
        self.initial = initial

    def _transport(self, lattice: "Lattice"):
        _, *tranported = lattice.transport(self.initial)
        return np.vstack(tranported)

    def loss(self, lattice: "Lattice") -> float:
        # if transported is None:
        #     # is None when 'solution' was specified but the lattice
        #     # has no twiss periodic solution.
        #     return np.inf
        transported = self._transport(lattice)

        transported_columns = [i + 1 for i in lattice.search(self.element)]
        transported_rows = [
            i for i, value in enumerate(self.value) if value is not None
        ]
        result = transported[transported_rows, transported_columns]
        # l-2 norm
        return np.linalg.norm(result - self.value[transported_rows], 2)

    def __repr__(self) -> str:
        args = ["element", "value", "initial"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"TargetPhasespace({arg_string})"


class TargetTwiss(BaseTarget):
    def __init__(
        self,
        element: Union[str, "BaseElement"],
        value: Sequence[float],
        plane: str = "h",
    ):
        value = np.array(value)
        if isinstance(element, BaseElement):
            element = element.name
        self.element = element
        self.value = value
        self.plane = plane

    def _transport(self, lattice: "Lattice"):
        _, *twiss = lattice.twiss(plane=self.plane)
        return np.vstack(twiss)

    def loss(self, lattice: "Lattice") -> float:
        # if transported is None:
        #     # is None when 'solution' was specified but the lattice
        #     # has no twiss periodic solution.
        #     return np.inf
        try:
            transported = self._transport(lattice)
        except ValueError:
            return np.inf

        transported_columns = [i + 1 for i in lattice.search(self.element)]
        transported_rows = [
            i for i, value in enumerate(self.value) if value is not None
        ]
        result = transported[transported_rows, transported_columns]
        # l-2 norm
        return np.linalg.norm(result - self.value[transported_rows], 2)

    def __repr__(self) -> str:
        args = ["element", "value", "plane"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"TargetTwiss({arg_string})"


class TargetDispersion(BaseTarget):
    def __init__(
        self,
        element: Union[str, "BaseElement"],
        value: float,
        plane: str = "h",
    ):
        value = np.array(value)
        if isinstance(element, BaseElement):
            element = element.name
        self.element = element
        self.value = value
        self.plane = plane

    def _transport(self, lattice: "Lattice"):
        _, *transported = lattice.dispersion(plane=self.plane)
        dispersion = transported[PLANE_INDICES[self.plane][0]]
        return np.array(dispersion)

    def loss(self, lattice: "Lattice") -> float:
        try:
            transported = self._transport(lattice)
        except ValueError:
            return np.inf

        transported_columns = [i + 1 for i in lattice.search(self.element)]
        transported_rows = [
            i for i, value in enumerate(self.value) if value is not None
        ]
        result = transported[transported_rows, transported_columns]
        return abs(result - self.value)

    def __repr__(self) -> str:
        args = ["element", "value", "plane"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"TargetDispersion({arg_string})"


class TargetGlobal(BaseTarget):
    def __init__(self, method: str, value: float, **method_kwargs):
        self.method = method
        self.value = value
        self.method_kwargs = method_kwargs

    def loss(self, lattice: "Lattice"):
        out = getattr(lattice, self.method)(**self.method_kwargs)
        return abs(out - self.value)

    def __repr__(self) -> str:
        args = ["element", "value", "method_kwargs"]
        arg_string = ", ".join([arg + "=" + repr(getattr(self, arg)) for arg in args])
        return f"TargetGlobal({arg_string})"


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

    def add_target(self, target: BaseTarget):
        """Add a constraint target.

        Args:
            element: Element name pattern or element instance at which the
                `value` should be achieved.
            value: Target value of twiss parameters or phase space coordinates
                at the given `element`.
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
        target_dict = {
            "dispersion": TargetDispersion,
            "phasespace": TargetPhasespace,
            "twiss": TargetTwiss,
            "global": TargetGlobal,
        }
        # if target is None:
        #     target = target_dict[target_type](*args, **kwargs)

        self.targets.append(target)

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
            out = []
            for target in self.targets:
                out.append(target.loss(lattice))
            # mean of the losses
            print(out)
            print(np.linalg.norm(out, 2))
            return np.linalg.norm(out, 2)

        res = minimize(
            match_function, *args, x0=root_start, method=default_method, **kwargs
        )
        if res.success:
            # sometimes the last iteration is not the minimum, set the real
            # solution
            self._set_parameters(res.x, lattice)
        return lattice, res

    def _set_parameters(self, new_settings: Sequence[float], lattice: "Lattice"):
        """Set the new lattice settings."""
        for param, value in zip(self.free_parameters, new_settings):
            for i in lattice.search(param.element):
                setattr(lattice[i], param.attribute, value)
                # print(lattice[i])
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

    def __repr__(self):
        return f"Constraints({repr(self._lattice)})"
