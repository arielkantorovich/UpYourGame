"""
Energy Game -- Common Utilities
===============================
Shared mathematical functions for the energy consumption game:
  - Simplex projection
  - Nash equilibrium (NE) gradient
  - Residual gradient for the global objective

@author: Ariel Kantorovich
"""

import numpy as np


def project_onto_simplex(V: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Project each (L, N) slice of *V* onto the simplex {x >= 0, sum(x) <= z}.

    Only vectors whose components sum to more than *z* are modified;
    the rest are returned unchanged.

    Parameters
    ----------
    V : np.ndarray, shape (L, N, K)
        Action matrix to project.
    z : float
        Radius (budget) of the simplex.

    Returns
    -------
    np.ndarray, shape (L, N, K)
    """
    L, N, K = V.shape
    sum_V = np.sum(V, axis=-1, keepdims=True)

    mask = sum_V > z

    U = np.sort(V, axis=-1)[:, :, ::-1]
    cssv = np.cumsum(U, axis=-1) - z
    ind = np.arange(1, K + 1)
    cond = (U - cssv / ind) > 0
    rho = np.count_nonzero(cond, axis=-1, keepdims=True)

    theta = np.take_along_axis(cssv, rho - 1, axis=-1) / rho
    projection = np.maximum(V - theta, 0)

    return np.where(mask, projection, V)


def calculate_NE_gradient(
    const_V: float,
    Xn_k: np.ndarray,
    A_k: np.ndarray,
    B_k: np.ndarray,
    Sk: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Nash-equilibrium gradient of player reward r_{n,k}.

    Parameters
    ----------
    const_V : float
        Scaling constant for the log-utility V = const_V * ln(1 + x).
    Xn_k : np.ndarray, shape (L, N, K)
    A_k  : np.ndarray, shape (L, N, K)
    B_k  : np.ndarray, shape (L, N, K)
    Sk   : np.ndarray, shape (L, N, K)

    Returns
    -------
    np.ndarray, shape (L, N, K)
    """
    grad_Pk = A_k * (Sk ** 2 + 2 * Sk * Xn_k) + B_k * (Sk + Xn_k)
    grad_vk = const_V / (1 + Xn_k)
    return grad_vk - grad_Pk


def calculate_residual_gradient(
    Xn_k: np.ndarray,
    A_k: np.ndarray,
    B_k: np.ndarray,
    Sk: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    Residual gradient that bridges the NE gradient to the global-optimal gradient.

    Parameters
    ----------
    Xn_k : np.ndarray, shape (L, N, K)
    A_k  : np.ndarray, shape (L, N, K)
    B_k  : np.ndarray, shape (L, N, K)
    Sk   : np.ndarray, shape (L, N, K)
    N    : int
        Number of players.

    Returns
    -------
    np.ndarray, shape (L, N, K)
    """
    temp = -(2 * A_k * Xn_k * Sk + B_k * Xn_k)
    grad_sum = np.sum(temp, axis=1)
    grad_sum = np.repeat(grad_sum[:, np.newaxis, :], N, axis=1)
    return grad_sum - temp
