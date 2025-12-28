"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
from scipy.stats import truncnorm
from dataclasses import dataclass
import argparse

@dataclass(slots=True)
class SimConfig:
    """
    Simulation Parameters
    """
    L: int = 200
    N: int = 5
    K: int = 14
    Rlink: float = 0.1
    T: int = 2000
    dist: str = "uniform"
    isPlot: bool = False
    alpha: float = 10e-3
    Border_floor: float = 0.0
    Border_ceil: float = 1.0
    N0: float = 0.001
    seed: int | None = None


@dataclass(slots=True)
class SimRecord:
    # arrays you want to log over time
    P: np.ndarray            # (T, L, N, K) or (L, N, K) depending on what you store
    obj: np.ndarray          # (T,) or (T,L) etc.
    grad_norm: np.ndarray    # (T,) etc.

    @classmethod
    def create(cls, cfg: "SimConfig") -> "SimRecord":
        """
        Refers to the class itself, Used in class methods.
        :param cfg:
        :return:
        """
        # choose shapes that match what you want to record
        P = np.zeros((cfg.T, cfg.L, cfg.N, cfg.K), dtype=np.float64)
        obj = np.zeros((cfg.T,), dtype=np.float64)
        grad_norm = np.zeros((cfg.T,), dtype=np.float64)
        return cls(P=P, obj=obj, grad_norm=grad_norm)


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Wireless naive K simulation")

    p.add_argument("--L", type=int, default=200)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--K", type=int, default=14)
    p.add_argument("--Rlink", type=float, default=0.1)
    p.add_argument("--N0", type=float, default=0.001)
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--dist", type=str, default="uniform", choices=["uniform", "normal"])
    p.add_argument("--plot", action="store_true", help="Enable plotting")
    p.add_argument("--seed", type=int, default=None)

    # optional extras
    p.add_argument("--alpha", type=float, default=10e-3)

    a = p.parse_args()
    return SimConfig(
        L=a.L, N=a.N, K=a.K, Rlink=a.Rlink, T=a.T,
        dist=a.dist, isPlot=a.plot, seed=a.seed, alpha=a.alpha
    )

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


def extract_g_diag(g: np.ndarray) -> np.ndarray:
    """
    Extract diagonal gains g_{i,i,k} for each l,k.
    g: (L, N, N, K) -> (L, N, K)
    """
    return np.diagonal(g, axis1=1, axis2=2)  # (L, N, K)

def compute_total_rx_power(g: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    total[l, j, k] = sum_i g[l, i, j, k] * P[l, i, k]
    g: (L,N,N,K), P: (L,N,K) -> total: (L,N,K)
    """
    return np.einsum('lijk,lik->ljk', g, P)

def compute_interference(g: np.ndarray, P: np.ndarray, g_diag: np.ndarray) -> np.ndarray:
    """
    In[l, j, k] = sum_{i != j} g[l, i, j, k] * P[l, i, k]
               = total - g_diag * P
    """
    total = compute_total_rx_power(g, P)     # (L, N, K)
    return total - g_diag * P                # (L, N, K)











def main():
    g = generate_gain_channel(N=5, L=200, K=10, output_layout="LNNK")
    print(g.shape)  # (200, 5, 5, 10)
    g2 = generate_gain_channel(N=5, L=200, K=10, output_layout="LKNN")
    print(g2.shape) # (200, 10, 5, 5)



if __name__ == '__main__':
    main()