from functools import reduce
from typing import Optional, Sequence, Tuple, Union

import numpy as np


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
    """Compute the twiss transfer matrix from a (2, 2) phase space transfer
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


def compute_dispersion_solution(transfer_m: np.ndarray) -> np.ndarray:
    # TODO: add numerical tolerances on the == 0 conditions
    disp_prime_denom = 1 - transfer_m[0, 0] - transfer_m[1, 1] + np.linalg.det(transfer_m[:2, :2])
    disp_denom = (1 - transfer_m[0, 0])
    if disp_prime_denom == 0 or disp_denom == 0:
        raise ValueError("Matrix has no periodic dispersion solution.")
    disp_prime = (transfer_m[1, 0] * transfer_m[0, 2] + transfer_m[1, 2] * (1 - transfer_m[0, 0])) / disp_prime_denom
    disp = (transfer_m[0, 1] * disp_prime + transfer_m[0, 2]) / disp_denom
    return to_v_vec((disp, disp_prime, 1))


def compute_twiss_solution(transfer_m: np.ndarray) -> np.ndarray:
    # TODO: add numerical tolerances on the < 0 condition
    denom = 1 - transfer_m[0, 0] ** 2 - 2 * transfer_m[0, 1] * transfer_m[1, 0] - transfer_m[1, 1]**2 + np.linalg.det(transfer_m[:2, :2])
    if denom < 0:
        raise ValueError("Matrix has no prediodic twiss solution.")
    denom = np.sqrt(denom)
    beta = 2 * transfer_m[0, 1] / denom
    alpha = (transfer_m[0, 0] - transfer_m[1, 1]) / denom
    gamma = (1 + alpha ** 2) / beta
    out = to_v_vec((beta, alpha, gamma))
    if out[0, 0] < 0:
        out *= -1
    return out
