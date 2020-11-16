from collections import namedtuple
from functools import reduce
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# where the x, x' and y, y' are located in the phase space coord vector
PLANE_INDICES = {"h": [0, 1], "v": [2, 3]}
PLANE_SLICES = {"h": slice(0, 2), "v": slice(2, 4)}

# some named tuples to make the interface friendlier
PhasespaceDistribution = namedtuple(
    "PhasespaceDistribution", ["x", "x_prime", "y", "y_prime", "dp"]
)
TransportedTwiss = namedtuple("TransportedTwiss", ["s", "beta", "alpha", "gamma"])
TransportedPhasespace = namedtuple(
    "TransportedPhasespace", ["s", "x", "x_prime", "y", "y_prime", "dp"]
)


def plot_twiss(self) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the evolution of twiss parameters through the lattice.

    Return:
        The plotted ``plt.Figure`` and ``plt.Axes``.
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(self.s, self.beta, label="beta")
    ax.plot(self.s, self.alpha, label="alpha")
    ax.plot(self.s, self.gamma, label="gamma")
    ax.set_xlabel("s [m]")
    ax.legend()
    return fig, ax


def plot_phasespace(
    self, plane: str = "both", add_legend=False
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the evolution of phase space coordinates through the lattice.

    Return:
        The plotted ``plt.Figure`` and ``plt.Axes``.
    """
    plane = plane.lower()
    if plane not in ("h", "v", "both"):
        raise ValueError("'plane' should be either 'h', 'v' or 'both'.")
    if plane == "both":
        n_plots = 2
    else:
        n_plots = 1
    fig, ax = plt.subplots(1, n_plots, sharex=True, sharey=True)

    def plot_plane(ax, u, u_prime, plane, add_legend=False):
        if all([coord.ndim > 1 for coord in (u, u_prime, self.dp)]):
            # if the data contains multiple particles, plot in phase space
            # TODO: this could be split into multiple subplots like what Michael did
            # in the notebook.
            lines = ax.plot(u, u_prime, linewidth=0, marker=".")
            ax.set_xlabel(f"{plane} [m]")
            ax.set_ylabel(f"{plane}'")
            ax.set_aspect("equal")
            if add_legend:
                fig.subplots_adjust(right=0.8)
                ax.legend(
                    lines,
                    [f"s={s}" for s in self.s],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
        else:
            # else plot the evolution of the particle coordinates
            ax.plot(self.s, u, label=f"{plane}")
            ax.plot(self.s, u_prime, label=f"{plane}_prime")
            ax.plot(self.s, self.dp, label="dp")
            ax.set_xlabel("s [m]")
            ax.set_ylabel(f"{plane} [m]")
            if add_legend:
                ax.legend()

    if plane == "h":
        plot_plane(ax, self.x, self.x_prime, "x", add_legend=add_legend)
    elif plane == "v":
        plot_plane(ax, self.y, self.y_prime, "y", add_legend=add_legend)
    elif plane == "both":
        plot_plane(ax[0], self.x, self.x_prime, "x", add_legend=False)
        plot_plane(ax[1], self.y, self.y_prime, "y", add_legend=add_legend)

    return fig, ax


def plot_phasespace_distribution(
    self, plane: str = "both"
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the phase space coordinates distribution.

    Return:
        The plotted ``plt.Figure`` and ``plt.Axes``.
    """
    plane = plane.lower()
    if plane not in ("h", "v", "both"):
        raise ValueError("'plane' should be either 'h', 'v' or 'both'.")
    if plane == "both":
        n_plots = 2
    else:
        n_plots = 1
    fig, ax = plt.subplots(1, n_plots, sharex=True, sharey=True)

    def plot_plane(ax, u, u_prime, plane, add_cbar=False):
        if (self.dp == 0).all():
            # if all the dp/p are 0 then don't change color based on dp/p
            ax.scatter(u, u_prime, s=4)
        else:
            # else add colour and colour bar w.r.t. to dp/p
            scatter = ax.scatter(u, u_prime, c=self.dp, s=4)
            if add_cbar:
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = fig.colorbar(scatter, cax=cbar_ax)
                cbar.set_label("dp/p", rotation=90)
        ax.set_xlabel(f"{plane} [m]")
        ax.set_ylabel(f"{plane}'")
        ax.set_aspect("equal")

    if plane == "h":
        plot_plane(ax, self.x, self.x_prime, "x", add_cbar=True)
    elif plane == "v":
        plot_plane(ax, self.y, self.y_prime, "y", add_cbar=True)
    elif plane == "both":
        plot_plane(ax[0], self.x, self.x_prime, "x", add_cbar=False)
        plot_plane(ax[1], self.y, self.y_prime, "y", add_cbar=True)

    return fig, ax


PhasespaceDistribution.plot = plot_phasespace_distribution
TransportedPhasespace.plot = plot_phasespace
TransportedTwiss.plot = plot_twiss


def to_v_vec(vec: Sequence[float]) -> np.ndarray:
    """Helper function to create 1D vertical arrays.

    Args:
        vec: Vector to convert to vertical array.

    Returns:
        Vertical 1D ``np.ndarray``.
    """
    vec = np.array(vec)
    if np.squeeze(vec).ndim > 1:
        raise ValueError("'vec' is not 1D.")
    return np.array(vec).reshape(-1, 1)


def to_twiss(twiss: Sequence[Union[float, None]]) -> np.ndarray:
    """Helper function to create twiss vectors.

    Args:
        twiss: List of length 3, a single twiss parameter can be None.

    Returns:
        Completed vertical 1D twiss parameter ``np.ndarray``.
    """
    if len(twiss) != 3:
        raise ValueError("Length of 'twiss' != 3.")
    twiss = complete_twiss(*twiss)
    return to_v_vec(twiss)


def to_phase_coord(phase_coord: Sequence[float]) -> np.ndarray:
    """Helper function to create phase space coordinate vectors.

    Args:
        phase_coord: List of length 3.

    Returns:
        Vertical 1D ``np.ndarray``.
    """
    if len(phase_coord) != 3:
        raise ValueError("Length of 'phase_coord' != 3, u, u_prime, dp/p.")
    return to_v_vec(phase_coord)


def complete_twiss(
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Given 2 twiss parameters, compute the third.

    Args:
        beta (optional): Beta function in meters.
        alpha (optional): Twiss alpha in radians.
        gamma (optional): Twiss gamma in meter^-1.

    Returns:
        Tuple of completes twiss parameters, (beta, alpha, gamma).
    """

    number_of_none = sum([param is None for param in (beta, alpha, gamma)])
    if number_of_none == 0:
        return (beta, alpha, gamma)
    if number_of_none != 1:
        raise ValueError("Only one twiss parameter can be omitted.")
    if beta is None:
        beta = (1 + alpha ** 2) / gamma
    elif alpha is None:
        alpha = np.sqrt(beta * gamma - 1)
    elif gamma is None:
        gamma = (1 + alpha ** 2) / beta
    return (beta, alpha, gamma)


def compute_one_turn(list_of_m: Sequence[np.array]) -> np.array:
    """Iteratively compute the matrix multiplictions of the arrays in the
    provided sequence.

    Args:
        list_of_m: Sequence of transfer matrices.

    Returns:
        Result of the iterative matrix multiplication of the matrices.
    """
    # matrix multiply all the elements.
    return reduce(lambda x, y: np.dot(y, x), list_of_m)


def compute_twiss_clojure(twiss: Sequence[float]) -> float:
    """Compute twiss clojure condition:

    beta * gamma - alpha^2 = 1

    Args:
        twiss: List of twiss parameters, [beta[m], alpha[rad], gamma[m]]

    Returns:
        beta * gamma - alpha^2
    """
    return twiss[0] * twiss[2] - twiss[1] ** 2


def compute_m_twiss(m: np.array) -> np.array:
    """Compute the twiss transfer matrix from a 2x2 phase space transfer
    matrix.

    Args:
        m: Phase space transfer matrix.

    Returns:
        Twiss parameter transfer matrix, (3, 3), beta, alpha, gamma.
    """
    m_twiss = np.zeros((3, 3))
    m_twiss[0][0] = m[0][0] ** 2
    m_twiss[0][1] = -2 * m[0][0] * m[0][1]
    m_twiss[0][2] = m[0][1] ** 2

    m_twiss[1][0] = -m[0][0] * m[1][0]
    m_twiss[1][1] = 1 + 2 * m[0][1] * m[1][0]
    m_twiss[1][2] = -m[0][1] * m[1][1]

    m_twiss[2][0] = m[1][0] ** 2
    m_twiss[2][1] = -2 * m[1][0] * m[1][1]
    m_twiss[2][2] = m[1][1] ** 2
    return m_twiss


def compute_twiss_solution(transfer_m: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Compute the periodic twiss solution for the provided transfer matrix.

    Args:
        transfer_m: 3x3 transfer matrix.
        tol: Numerical tolerance.

    Returns:
        Twiss vector, beta, alpha, gamma.
    """
    denom = (
        1
        - transfer_m[0, 0] ** 2
        - 2 * transfer_m[0, 1] * transfer_m[1, 0]
        - transfer_m[1, 1] ** 2
        + np.linalg.det(transfer_m[:2, :2])
    )
    if denom < 0 + tol:
        raise ValueError("Matrix has no prediodic twiss solution.")
    denom = np.sqrt(denom)
    beta = 2 * transfer_m[0, 1] / denom
    alpha = (transfer_m[0, 0] - transfer_m[1, 1]) / denom
    gamma = (1 + alpha ** 2) / beta
    out = to_v_vec((beta, alpha, gamma))
    if out[0, 0] < 0:
        out *= -1
    return out
