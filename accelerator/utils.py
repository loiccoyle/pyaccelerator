from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np


def to_v_vec(vec: List[float]) -> np.ndarray:
    """Helper function to create 1D vertical arrays.
    """
    vec = np.array(vec)
    if np.squeeze(vec).ndim > 1:
        raise ValueError("'vec' is not 1D.")
    return np.array(vec).reshape(-1, 1)


def to_twiss(twiss: List[Union[float, None]]) -> np.ndarray:
    """Helper function to create twiss vectors.

    Args:
        twiss: list of length 3.
    """
    if len(twiss) != 3:
        raise ValueError(f"Length of 'twiss' != 3.")
    twiss = complete_twiss(*twiss)
    return np.array(twiss).reshape(3, 1)


def to_phase_coord(phase_coord: List[float]) -> np.ndarray:
    """Helper function to create phase space coordinate vectors.

    Args:
        twiss: list of length 2.
    """
    if len(phase_coord) != 2:
        raise ValueError(f"Length of 'phase_coord' != 2.")
    return np.array(phase_coord).reshape(2, 1)


def complete_twiss(
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Given 2 twiss parameters, compute the third.

    Args:
        beta (optional): beta function in meters.
        alpha (optional): twiss alpha in radians.
        gamma (optional): twiss gamma in meter^-1.

    Returns:
        tuple of completes twiss parameters, (beta, alpha, gamma).
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


def compute_one_turn(list_of_m: List[np.array]) -> np.array:
    # matrix multiply all the elements.
    return reduce(lambda x, y: np.dot(y, x), list_of_m)


def compute_twiss_clojure(twiss: List[float]) -> float:
    """Compute twiss clojure condition:

    beta * gamma - alpha^2 = 1

    Args:
        twiss: list of twiss parameters, [beta[m], alpha[rad], gamma[m]]

    Returns:
        beta * gamma - alpha^2
    """
    return twiss[0] * twiss[2] - twiss[1] ** 2


def compute_m_twiss(m: np.array) -> np.array:
    """Compute the twiss transfer matrix from a (2, 2) phase space transfer
    matrix.

    Args:
        m: phase space transfer matrix.

    Returns:
        twiss parameter transfer matrix, (3, 3), beta, alpha, gamma.
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


def compute_invariant(transfer_matrix: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    eig_values, eig_vectors = np.linalg.eig(transfer_matrix)
    mask = (eig_values < 1 + tol) & (eig_values > 1 - tol)
    if sum(mask) == 0:
        raise ValueError("'transfer_matrix' does not have any invariant points.")
    return eig_vectors[:, mask]


def compute_twiss_invariant(
    twiss_transfer_matrix: np.ndarray, tol: float = 1e-14
) -> np.ndarray:
    """Find twiss parameters which are invariant under the provided transfer matrix.

    Args:
        twiss_transfer_matrix: (3, 3) transfer matrix.
        tol (optional): numerical tolerance.

    Returns:
        invariant twiss parameters.
    """
    if (
        twiss_transfer_matrix.shape[0] != 3
        or twiss_transfer_matrix.shape[0] != twiss_transfer_matrix.shape[1]
    ):
        raise ValueError("'twiss_transfer_matrix' is not of shape (3, 3).")
    invariants = compute_invariant(twiss_transfer_matrix)
    potential_twiss = np.apply_along_axis(compute_twiss_clojure, 0, invariants) > tol
    if sum(potential_twiss) == 0:
        raise ValueError(
            "No eigen vectors are compatible with twiss clojure condition."
        )
    index = np.where(potential_twiss)[0][0]
    twiss = to_twiss(invariants[:, index].real)
    clojure = compute_twiss_clojure(twiss)
    if clojure != 1:
        twiss /= np.sqrt(clojure)
    if twiss[0][0] < 0:
        twiss *= -1
    return twiss
