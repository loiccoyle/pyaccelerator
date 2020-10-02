"""Accelerator Beam"""
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants

from .sampling import bigaussian
from .utils import compute_twiss_clojure, to_twiss


class Beam:
    """Represents one beam.

    Args:
        energy: Beam kinetic energy in MeV, defaults to 6500000.
        mass: Particle mass in MeV, defaults to proton mass.
        n_particles: Number of particles in the beam, defaults to 1000.
        emittance: Normalized beam emittance in meters, to specify horizontal
            and vertical emittances use a tuple, defaults to 3.5e-6.
        sigma_energy: Kinetic energy spread in MeV.
        sampling: distribution sampling method, defaults to "bigaussian".

    Examples:
        Beam with even emittances:

            >>> Beam(n_particles=100, emittance=2.5e-6)

        Beam with uneven emittances:

            >>> Beam(n_particles=100, emittance=(3.5e-6, 2.5e-6))

        Compute the phase space ellipse:

            >>> beam = Beam()
            >>> x, x_prime, dp = beam.ellipse([1, 2, 5], plane="h")

        Match a distribution to twiss parameters:

            >>> beam = Beam()
            >>> x, x_prime, dp = beam.match([1, 2, 5], plane="h")
    """

    _sampling_map = {"bigaussian": bigaussian}

    def __init__(
        self,
        energy: float = 6500000.0,
        mass: float = physical_constants["proton mass energy equivalent in MeV"][0],
        n_particles: int = 1000,
        emittance: Union[Tuple[float, float], float] = 3.5e-6,
        sigma_energy: float = 0,
        sampling: str = "bigaussian",
    ):
        if not isinstance(emittance, tuple):
            emittance = (emittance, emittance)
        self.energy = energy
        self.mass = mass
        self.emittance_h = emittance[0]
        self.emittance_v = emittance[1]
        self.sigma_energy = sigma_energy
        self.n_particles = n_particles
        self.sampling = self._sampling_map[sampling]
        self._sampling_str = sampling

    @property
    def gamma_relativistic(self):
        return self.energy / self.mass

    @property
    def beta_relativistic(self):
        return np.sqrt(1 - 1 / self.gamma_relativistic ** 2)

    @property
    def geo_emittance_h(self):
        return self.emittance_h / (self.beta_relativistic * self.gamma_relativistic)

    @property
    def geo_emittance_v(self):
        return self.emittance_v / (self.beta_relativistic * self.gamma_relativistic)

    @property
    def p(self):
        # in MeV/c
        return np.sqrt(self.energy ** 2 + 2 * self.energy * self.mass)

    @property
    def sigma_p(self):
        # in MeV/c
        return np.sqrt(self.sigma_energy ** 2 + 2 * self.sigma_energy * self.mass)

    def ellipse(
        self,
        twiss: Sequence[float],
        plane: str = "h",
        closure_tol: float = 1e-10,
        n_angles: int = 1e3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the beam's phase space ellipse given the twiss parameters.

        Args:
            twiss: Twiss parameters, beta[m], alpha[rad], gamma[m^-1], one
                twiss parameter can be None.
            plane: Plane of interest, either "h" or "v", defaults to "h".
            closure_tol: Numerical tolerance on the twiss closure condition,
                defaults to 1e-9.
            n_angles: Number of angles for which to compute the ellipse,
                defaults to 1e3.

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
        dp = np.zeros(u_prime.shape)
        return u, u_prime, dp

    def match(
        self,
        twiss: Sequence[float],
        plane: str = "h",
        closure_tol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a matched beam phase space distribution to the provided
        `twiss` parameters.

        Args:
            twiss: Initial twiss parameters.
            plane: Plane of interest, either "h" or "v", defaults to "h".
            closure_tol: Numerical tolerance on the twiss closure condition,
                defaults to 1e-10.

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
        u_pre, u_prime_pre, dp = self.sampling(
            self.n_particles, (0, 0, 0), emit, self.sigma_p
        )
        # match circle to the ellipse described by the twiss parameters
        u = np.sqrt(beta) * u_pre
        u_prime = -(alpha / np.sqrt(beta)) * u_pre + (1.0 / np.sqrt(beta)) * u_prime_pre
        # turn dp into dp/p
        dp /= self.p
        return u, u_prime, dp

    def __repr__(self) -> str:
        args = {
            "energy": self.energy,
            "mass": self.mass,
            "n_particles": self.n_particles,
            "emittance": (self.emittance_h, self.emittance_v),
            "sigma_energy": self.sigma_energy,
            "sampling": self._sampling_str,
        }
        arg_str = ",\n".join([f"{key}={repr(value)}" for key, value in args.items()])
        return f"Beam(\n{arg_str})"

    def plot(
        self,
        twiss_h: Optional[Sequence[float]] = None,
        twiss_v: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot the particle distribution matched to the `twiss` parameters in
        both the horizontal and vertical planes.

        Args:
            twiss_h (optional): Horizontal twiss parameters to match the distribution.
            twiss_v (optional): Vertical twiss parameters to match the distribution.
            **kwargs: Passed to ``plt.scatter``.

        Returns:
            The plotted ``plt.Figure`` and a ``np.ndarray`` of the ``plt.Axes``.
        """
        if twiss_h is None and twiss_v is None:
            raise ValueError("Both 'twiss_h' and 'twiss_h' are None.")
        n_plots = sum([twiss is not None for twiss in [twiss_h, twiss_v]])
        current_plot = 0
        fig, axes = plt.subplots(1, n_plots, sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        fig.suptitle("Phase space beam ditributions")
        if twiss_h is not None:
            x, x_prime, _ = self.match(twiss_h, plane="h")
            axes[current_plot].scatter(x, x_prime, **kwargs)
            axes[current_plot].set_xlabel("x [m]")
            axes[current_plot].set_ylabel("x'")
            axes[current_plot].set_aspect("equal")
            current_plot += 1
        if twiss_v is not None:
            y, y_prime, _ = self.match(twiss_v, plane="v")
            axes[current_plot].scatter(y, y_prime, **kwargs)
            axes[current_plot].set_xlabel("y [m]")
            axes[current_plot].set_ylabel("y'")
            axes[current_plot].set_aspect("equal")
        fig.tight_layout()
        return fig, axes
