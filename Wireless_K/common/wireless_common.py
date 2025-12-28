"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
from scipy.stats import truncnorm


def generate_gain_channel(
        N: int = 5,
        L: int = 200,
        K: int = 1,
        alpha: float = 10e-3,
        R_link: float = 0.1,
        distribution: str = "uniform",
        eps: float = 1e-12,
        output_layout: str = "LNNK",
) -> np.ndarray:
    """
    Generate gain tensor for L trials:
      - N transmitters (players)
      - N receivers (one per player)
      - K channel realizations (K different receiver offsets per trial)

    Returns:
      output_layout="LNNK": g shape (L, N, N, K)
      output_layout="LKN N": g shape (L, K, N, N)
    """

    # Tx locations: (L, N, 2)
    if distribution == "uniform":
        Transceivers = np.random.rand(L, N, 2) * 2 - 1
    elif distribution == "normal":
        lower, upper = -1, 1
        mu, sigma = 0, 0.5
        Transceivers = truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
        ).rvs((L, N, 2))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # For each k, make a different Rx position around each Tx:
    # ReceiverOffsets: (L, N, K, 2)
    ReceiverOffsets = R_link * (np.random.rand(L, N, K, 2) * 2 - 1)

    # Receivers: (L, N, K, 2)
    Receivers = Transceivers[:, :, None, :] + ReceiverOffsets

    # Pairwise distances from Tx_i to Rx_j(k):
    # Transceivers: (L, N, 1, 1, 2)
    # Receivers:     (L, 1, N, K, 2)
    # diff -> (L, N, N, K, 2) -> norm -> (L, N, N, K)
    diff = Transceivers[:, :, None, None, :] - Receivers[:, None, :, :, :]
    distances = np.linalg.norm(diff, axis=-1)  # (L, N, N, K)

    g = alpha / (distances**2 + eps)  # (L, N, N, K)

    if output_layout.upper() == "LKN N".replace(" ", ""):  # "LKNN"
        return np.transpose(g, (0, 3, 1, 2))  # (L, K, N, N)
    elif output_layout.upper() == "LNNK":
        return g
    else:
        raise ValueError("output_layout must be 'LNNK' or 'LKNN'")




def calc_local_gradient(
    g_diag: np.ndarray,
    In: np.ndarray,
    N0: float | np.ndarray,
    P: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Calculate the local gradient ("greedy part") for multi-channel case.

    Parameters
    ----------
    g_diag : ndarray, shape (L, N, K)
        Direct channel gains (diagonal terms) per trial/player/channel.
    In : ndarray, shape (L, N, K)
        Interference power per trial/player/channel (excluding desired signal).
    N0 : float or ndarray
        Noise power. Can be scalar, (K,), (1,1,K), or (L,N,K).
    P : ndarray, shape (L, N, K)
        Transmit powers per trial/player/channel.
    eps : float
        Small constant for numerical stability (avoid division by 0).

    Returns
    -------
    gradients_local : ndarray, shape (L, N, K)
        Local gradient values per trial/player/channel.
    """
    if g_diag.shape != In.shape or g_diag.shape != P.shape:
        raise ValueError(f"Shape mismatch: g_diag{g_diag.shape}, In{In.shape}, P{P.shape}")

    denom = In + N0
    numerator = g_diag / (denom + eps)          # (L, N, K)
    SNR = numerator * P                          # (L, N, K)
    gradients_local = numerator / (1.0 + SNR + eps)  # (L, N, K)

    return gradients_local


def calc_residual_gradient(g_diag: np.ndarray,
                           g_zero: np.ndarray,
                           In: np.ndarray,
                           N0: float | np.ndarray,
                           P: np.ndarray,
                           grad_local: np.ndarray,
                           eps: float = 1e-12
                           ) -> np.ndarray:
    """
    Calculate the residual gradient the original prior gradient
         results[l, j, k] = - sum_i g0[l, i, j, k] * temp[l, i, k]
    :param g_diag:  shape (L, N, K), Direct channel gains (diagonal terms) per trial/player/channel.
    :param g_zero: shape (L, N, N, K) g where the diagonal is zero.
    :param In: shape (L, N, K), Interference power per trial/player/channel (excluding desired signal).
    :param N0: float, White gaussian Noise power
    :param P: shape (L, N, K), Transmit powers per trial/player/channel.
    :param grad_local: shape (L, N, K), Local gradient values per trial/player/channel.
    :param eps: float, Small constant for numerical stability (avoid division by 0).
    :return: prior gradient (residual gradient) (L, N, K) shape.
    """
    if g_diag.shape != In.shape or g_diag.shape != P.shape:
        raise ValueError(f"Shape mismatch: g_diag{g_diag.shape}, In{In.shape}, P{P.shape}")
    denom = In + N0
    numerator = g_diag / (denom + eps)  # (L, N, K)
    SNR = numerator * P  # (L, N, K)
    temp = grad_local * SNR / (g_diag + eps)  # (L, N, K)
    results = -np.einsum('lijk,lik->ljk', g_zero, temp)  # (L, N, K)
    return results



def extract_g_diag(g: np.ndarray) -> np.ndarray:
    """
    Extract diagonal gains g_{i,i,k} for each l,k.
    g: (L, N, N, K) -> (L, N, K)
    """
    diag = np.diagonal(g, axis1=1, axis2=2)   # (L, K, N)
    return np.moveaxis(diag, -1, 1)

def compute_total_rx_power(g: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Calculate  total[l, j, k] = sum_i g[l, i, j, k] * P[l, i, k]
    :param g: (L,N,N,K) - channel gain {gmn}
    :param P: (L,N,K) - transmission power
    :return (L, N, K) - total power sum.
    """
    return np.einsum('lijk,lik->ljk', g, P)

def compute_interference(g: np.ndarray, P: np.ndarray, g_diag: np.ndarray) -> np.ndarray:
    """
    The following function calculate the interference of player n using vectorization
    In[l, j, k] = sum_{i != j} g[l, i, j, k] * P[l, i, k]
               = total - g_diag * P
    :param g: (L,N,N,K) - channel gain {gmn}
    :param P: (L,N,K) - transmission power
    :param g_diag: (L,K, N) - channel gain between transmitter and is indeed receiver
    :return In: (L, N, K)
    """
    total = compute_total_rx_power(g, P)     # (L, N, K)
    return total - g_diag * P                # (L, N, K)


def project_box(
    P: np.ndarray,
    low: float | np.ndarray,
    high: float | np.ndarray
) -> None:
    """
    In-place Project P onto box constraints [low, high].

    Parameters
    ----------
    P : ndarray
        Input array.
    low : float or ndarray
        Lower bound (scalar or broadcastable).
    high : float or ndarray
        Upper bound (scalar or broadcastable).

    Returns
    -------
    ndarray
        Projected array (new array).
    """
    np.clip(P, low, high, out=P)

def compute_objective(In: np.ndarray,
                      g_diag: np.ndarray,
                      P: np.ndarray,
                      N0: float,
                      eps: float = 1e-12) -> float:
    """
    Calculate objective.
    :param obj: float is obj[t]
    :return: float, put in the objective the relevant results
    """
    denom = In + N0
    numerator = g_diag / (denom + eps)  # (L, N, K)
    SNR = numerator * P  # (L, N, K)
    temp = np.log(1 + SNR) # (L N K)
    total = np.sum(temp, axis=2) # Sum on K
    total = np.sum(total, axis=1) # Sum On N
    obj = np.mean(total) # Calculate Average on L trilas
    return obj












def main():
    g = generate_gain_channel(N=5, L=200, K=10, output_layout="LNNK")
    print(g.shape)  # (200, 5, 5, 10)
    g2 = generate_gain_channel(N=5, L=200, K=10, output_layout="LKNN")
    print(g2.shape) # (200, 10, 5, 5)



if __name__ == '__main__':
    main()