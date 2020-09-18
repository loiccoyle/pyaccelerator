"""Accelerator Beam"""
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from .sampling import bigaussian
from .utils import compute_twiss_clojure, to_twiss


class Beam:
    """Represents one beam.

    Args:
        energy (optional): Beam energy in GeV.
        mass (optional): Particle mass in kg.
        n_particles (optional): Number of particles in the beam.
        emittance (optional): Normalized beam emittance in meters, to specify
            horizontal and vertical emittances use a tuple.
        sampling (optional): distribution sampling method.

    Examples:
        Beam with even emittances:

            >>> Beam(n_particles=100, emittance=2.5e-6)

        Beam with uneven emittances:

            >>> Beam(n_particles=100, emittance=(3.5e-6, 2.5e-6))

        Compute the phase space ellipse:

            >>> beam = Beam()
            >>> x, x_prime = beam.ellipse([1, 2, 5], plane="h")

        Match a distribution to twiss parameters:

            >>> beam = Beam()
            >>> x, x_prime = beam.match([1, 2, 5], plane="h")
    """

    _sampling_map = {"bigaussian": bigaussian}

    def __init__(
        self,
        energy: float = 6500.0,
        mass: float = m_p,
        n_particles: int = 1000,
        emittance: Union[Tuple[float, float], float] = 3.5e-6,
        sampling: str = "bigaussian",
    ):
        if not isinstance(emittance, tuple):
            emittance = (emittance, emittance)
        self.energy = energy
        self.mass = mass
        self.gamma_relativistic = self.energy * 1e9 * e / (self.mass * c ** 2)
        self.beta_relativistic = np.sqrt(1.0 - 1.0 / self.gamma_relativistic ** 2)
        self.emittance_h = emittance[0]
        self.emittance_v = emittance[1]
        self.n_particles = n_particles
        self.sampling = self._sampling_map[sampling]
        self._sampling_str = sampling

    @property
    def geo_emittance_h(self):
        return self.emittance_h / (self.beta_relativistic * self.gamma_relativistic)

    @property
    def geo_emittance_v(self):
        return self.emittance_v / (self.beta_relativistic * self.gamma_relativistic)

    def ellipse(
        self,
        twiss: Sequence[float],
        plane: str = "h",
        closure_tol: float = 1e-9,
        n_angles: int = 1e3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the beam's phase space ellipse given the twiss parameters.

        Args:
            twiss: Twiss parameters, beta[m], alpha[rad], gamma[m^-1], one
                twiss parameter can be None.
            plane (optional): Plane of interest, either 'h' or 'v'.
            closure_tol (optional): Numerical tolerance on the twiss closure
                condition.
            n_angles (optional): Number of angles for which to compute the ellipse.

        Returns:
            Position and angle phase space coordrinates of the ellipse.
        """
        twiss = to_twiss(twiss)
        beta, alpha, _ = twiss.T[0]  # pylint: disable=unsubscriptable-object
        closure = compute_twiss_clojure(twiss)
        if not -closure_tol <= closure - 1 <= closure_tol:
            raise ValueError(
                f"Closure condition not met: beta * gamma - alpha**2 = {closure} != 1"
            )
        emit = getattr(self, "geo_emittance_" + plane.lower())
        angles = np.linspace(0, 2 * np.pi, int(n_angles))
        # TODO: make sure these equations are correct
        u = np.sqrt(emit * beta) * np.cos(angles)
        u_prime = -(alpha / beta) * u - np.sqrt(emit / beta) * np.sin(angles)
        return u, u_prime

    def match(
        self,
        twiss: Sequence[float],
        plane: str = "h",
        closure_tol: float = 1e-9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a matched beam phase space distribution to the provided
        `twiss` parameters.

        Args:
            twiss: Initial twiss parameters.
            plane (optional): Plane of interest, either "h" or "v".

        Returns:
            Position and angle phase space coordinates.
        """
        plane = plane.lower()
        twiss = to_twiss(twiss)
        beta, alpha, _ = twiss.T[0]  # pylint: disable=unsubscriptable-object
        closure = compute_twiss_clojure(twiss)
        if not -closure_tol <= closure - 1 <= closure_tol:
            raise ValueError(
                f"Closure condition not met: beta * gamma - alpha**2 = {closure} != 1"
            )
        emit = getattr(self, "geo_emittance_" + plane)
        u_pre, u_prime_pre = self.sampling(self.n_particles, (0, 0), emit)
        u = np.sqrt(beta) * u_pre
        u_prime = -(alpha / np.sqrt(beta)) * u_pre + (1.0 / np.sqrt(beta)) * u_prime_pre
        return u, u_prime

    def __repr__(self) -> str:
        args = {
            "energy": self.energy,
            "mass": self.mass,
            "n_particles": self.n_particles,
            "emittance": (self.emittance_h, self.emittance_v),
            "sampling": self._sampling_str,
        }
        arg_str = ",\n".join([f"{key}={repr(value)}" for key, value in args.items()])
        return f"Beam(\n{arg_str})"

    def plot(
        self, twiss: Sequence[float], *args, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot the particle distribution matched to the `twiss` parameters in
        both the horizontal and vertical planes.

        Args:
            twiss: Twiss parameters to match the distribution.
            args, kwargs: Passed to `plt.scatter`.

        Returns:
            The plotted `plt.Figure` and a `np.ndarray` of the `plt.Axes`.
        """
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("Unmatched phase space beam ditributions")
        axes[0].scatter(*self.match(twiss, plane="h"), *args, **kwargs)
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("x'")
        axes[0].set_aspect("equal")
        axes[1].scatter(*self.match(twiss, plane="v"), *args, **kwargs)
        axes[1].set_xlabel("y [m]")
        axes[1].set_ylabel("y'")
        axes[1].set_aspect("equal")
        return fig, axes
