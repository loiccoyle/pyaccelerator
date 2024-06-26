"""Accelerator Beam"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.constants import physical_constants

from .sampling import bigaussian
from .utils import PhasespaceDistribution, compute_twiss_clojure, to_twiss


class Beam:
    """Represents one beam.

    Args:
        energy: Beam kinetic energy in MeV, defaults to 6500000.
        mass: Particle mass in MeV, defaults to proton mass.
        n_particles: Number of particles in the beam, defaults to 1000.
        emittance: Normalized beam emittance in meters, to specify horizontal
            and vertical emittances use a tuple, defaults to 3.5e-6.
        sigma_energy: Relative Kinetic energy spread.
        sampling: Distribution sampling method, defaults to "bigaussian".

    Examples:
        Beam with even emittances:

            >>> Beam(n_particles=100, emittance=2.5e-6)

        Beam with uneven emittances:

            >>> Beam(n_particles=100, emittance=(3.5e-6, 2.5e-6))

        Compute the phase space ellipse:

            >>> beam = Beam()
            >>> ellipse = beam.ellipse([1, 2, 5])
            >>> ellipse.x
            ...
            >>> ellipse.x_prime
            ...
            >>> ellipse.y
            ...
            >>> ellipse.y_prime
            ...
            >>> ellipse.dp
            ...
            >>> ellipse.plot()
            ...

        Match a distribution to twiss parameters:

            >>> beam = Beam()
            >>> distrib = beam.match([1, 2, 5])
            >>> distrib.x
            ...
            >>> distrib.x_prime
            ...
            >>> distrib.y
            ...
            >>> distrib.y_prime
            ...
            >>> distrib.dp
            ...
            >>> distrib.plot()
            ...
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
        return (self.mass + self.energy) / self.mass

    @property
    def beta_relativistic(self):
        return np.sqrt(1 - 1 / self.gamma_relativistic**2)

    @property
    def geo_emittance_h(self):
        return self.emittance_h / (self.beta_relativistic * self.gamma_relativistic)

    @property
    def geo_emittance_v(self):
        return self.emittance_v / (self.beta_relativistic * self.gamma_relativistic)

    @property
    def p(self):
        # in MeV/c
        return np.sqrt(self.energy**2 + 2 * self.energy * self.mass)

    @property
    def sigma_p(self):
        absolute_sigma_e = self.sigma_energy * self.energy
        # in MeV/c
        return np.sqrt(absolute_sigma_e**2 + 2 * absolute_sigma_e * self.mass)

    def ellipse(
        self,
        twiss_h: Union[Sequence[float], np.ndarray],
        twiss_v: Optional[Union[Sequence[float], np.ndarray]] = None,
        closure_tol: float = 1e-10,
        n_angles: int = 1000,
    ) -> PhasespaceDistribution:
        """Compute the beam's phase space ellipse given the twiss parameters.

        Args:
            twiss_h: Horizontal twiss parameters, beta[m], alpha[rad], gamma[m^-1], one
                twiss parameter can be None.
            twiss_v: Vertical twiss parameters, beta[m], alpha[rad], gamma[m^-1], one
                twiss parameter can be None, if None assumes the same twiss
                values as `twiss_h`.
            closure_tol: Numerical tolerance on the twiss closure condition,
                defaults to 1e-10.
            n_angles: Number of angles for which to compute the ellipse,
                defaults to 1e3.

        Returns:
            Position, angle phase and dp/p space coordrinates of the ellipse.
                Note, dp/p will be set to 0.
        """
        twiss_h_ = to_twiss(twiss_h)
        twiss_v_ = twiss_h_ if twiss_v is None else to_twiss(twiss_v)
        beta_h, alpha_h, _ = twiss_h_.T[0]
        beta_v, alpha_v, _ = twiss_v_.T[0]
        # check the twiss parameters
        for twiss in (twiss_h_, twiss_v_):
            closure = compute_twiss_clojure(twiss)  # type: ignore
            if not -closure_tol <= closure - 1 <= closure_tol:
                raise ValueError(
                    f"Closure condition not met for {twiss}: beta * gamma - alpha**2 = {closure} != 1"
                )
        angles = np.linspace(0, 2 * np.pi, int(n_angles))
        # TODO: make sure these equations are correct
        # horizontal phase space ellipse
        x = np.sqrt(self.geo_emittance_h * beta_h) * np.cos(angles)
        x_prime = -(alpha_h / beta_h) * x - np.sqrt(
            self.geo_emittance_h / beta_h
        ) * np.sin(angles)
        # vetical phase space ellipse
        y = np.sqrt(self.geo_emittance_v * beta_v) * np.cos(angles)
        y_prime = -(alpha_v / beta_v) * y - np.sqrt(
            self.geo_emittance_v / beta_v
        ) * np.sin(angles)
        dp = np.zeros(x_prime.shape)
        return PhasespaceDistribution(x, x_prime, y, y_prime, dp)

    def match(
        self,
        twiss_h: Union[Sequence[float], np.ndarray],
        twiss_v: Optional[Union[Sequence[float], np.ndarray]] = None,
        closure_tol: float = 1e-10,
    ) -> PhasespaceDistribution:
        """Generate a matched beam phase space distribution to the provided
        `twiss` parameters.

        Args:
            twiss_h: Horizontal twiss parameters, beta[m], alpha[rad],
                gamma[m^-1], to which to match the distribution.
            twiss_v: Horizontal twiss parameters, beta[m], alpha[rad],
                gamma[m^-1], to which to match the distribution, if None will
                use the same values as `twiss_h`.
            closure_tol: Numerical tolerance on the twiss closure condition,
                defaults to 1e-10.

        Returns:
            Position, angle and dp/p phase space coordinates.
        """
        twiss_h_ = to_twiss(twiss_h)
        twiss_v_ = twiss_h_ if twiss_v is None else to_twiss(twiss_v)
        beta_h, alpha_h, _ = twiss_h_.T[0]
        beta_v, alpha_v, _ = twiss_v_.T[0]
        # check the twiss parameters
        for twiss in (twiss_h_, twiss_v_):
            closure = compute_twiss_clojure(twiss)  # type: ignore
            if not -closure_tol <= closure - 1 <= closure_tol:
                raise ValueError(
                    f"Closure condition not met: beta * gamma - alpha**2 = {closure} != 1"
                )
        x_pre, x_prime_pre, y_pre, y_prime_pre, dp = self.sampling(
            self.n_particles,
            (0, 0, 0, 0, 0),
            self.geo_emittance_h,
            self.geo_emittance_v,
            self.sigma_p,
        )
        # match circle to the ellipse described by the twiss parameters x plane
        x = np.sqrt(beta_h) * x_pre
        x_prime = (
            -(alpha_h / np.sqrt(beta_h)) * x_pre + (1.0 / np.sqrt(beta_h)) * x_prime_pre
        )
        # match circle to the ellipse described by the twiss parameters y plane
        y = np.sqrt(beta_v) * y_pre
        y_prime = (
            -(alpha_v / np.sqrt(beta_v)) * y_pre + (1.0 / np.sqrt(beta_v)) * y_prime_pre
        )
        # turn dp into dp/p
        dp /= self.p
        return PhasespaceDistribution(x, x_prime, y, y_prime, dp)

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
