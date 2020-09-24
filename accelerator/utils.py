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
        phase_coord: List of length 2.

    Returns:
        Vertical 1D ``np.ndarray``.
    """
    if len(phase_coord) != 2:
        raise ValueError("Length of 'phase_coord' != 2.")
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


def compute_invariant(transfer_matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Computes the invariant vector(s) for a given transformation matrix.

    Args:
        transfer_matrix: Transformation matrix.
        tol: Numerical tolerance, defaults to 1e-10.

    Returns:
        ``np.ndarray`` of invariant vectors, each column is a vector.
    """
    eig_values, eig_vectors = np.linalg.eig(transfer_matrix)
    mask = (eig_values < 1 + tol) & (eig_values > 1 - tol)
    if sum(mask) == 0:
        raise ValueError("'transfer_matrix' does not have any invariant points.")
    return eig_vectors[:, mask]


def compute_twiss_invariant(
    twiss_transfer_matrix: np.ndarray, tol: float = 1e-10
) -> np.ndarray:
    """Find twiss parameters which are invariant under the provided transfer matrix.

    Args:
        twiss_transfer_matrix: (3, 3) transfer matrix.
        tol: Numerical tolerance, defaults to 1e-10.

    Returns:
        Invariant twiss parameters.
    """
    if twiss_transfer_matrix.shape[0] != 3 or twiss_transfer_matrix.shape[1] != 3:
        raise ValueError("'twiss_transfer_matrix' is not of shape (3, 3).")
    invariants = compute_invariant(twiss_transfer_matrix, tol=tol)
    twiss_clojures = np.apply_along_axis(compute_twiss_clojure, 0, invariants)
    potential_twiss = (twiss_clojures > tol)
    if not any(potential_twiss):
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
