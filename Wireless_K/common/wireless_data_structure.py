"""
Created on : ------

@author: Ariel_Kantorovich
"""

from dataclasses import dataclass
import argparse
import numpy as np
from enum import IntEnum

@dataclass(slots=True)
class SimConfig:
    """
    Simulation Parameters
    """
    L: int = 200
    N: int = 5
    K: int = 14
    Rlink: float = 0.1
    lr_c: float = 0.002
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
        :Note: cls is SimRecord
        :param cfg:
        :return:
        """
        # choose shapes that match what you want to record
        P = np.zeros((cfg.T, cfg.L, cfg.N, cfg.K), dtype=np.float64)
        obj = np.zeros((cfg.T,), dtype=np.float64)
        grad_norm = np.zeros((cfg.T,), dtype=np.float64)
        return cls(P=P, obj=obj, grad_norm=grad_norm)


class GradMode(IntEnum):
    NAIVE_NASH = 0
    OPTIMAL = 1
    PRIOR_APPROXIMATION = 2


@dataclass(frozen=True)
class Sim_G:
    g: np.ndarray
    g_diag: np.ndarray
    g_zero: np.ndarray

def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Wireless naive K simulation")

    p.add_argument("--L", type=int, default=200)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--K", type=int, default=14)
    p.add_argument("--Rlink", type=float, default=0.1)
    p.add_argument("--N0", type=float, default=0.001)
    p.add_argument("--lr_c", type=float, default=0.002, help="Learning rate coefficient for the gradient update (default: 0.002)")
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--dist", type=str, default="uniform", choices=["uniform", "normal"])
    p.add_argument("--plot", action="store_true", help="Enable plotting")
    p.add_argument("--seed", type=int, default=None)

    # optional extras
    p.add_argument("--alpha", type=float, default=10e-3)

    a = p.parse_args()
    return SimConfig(
        L=a.L, N=a.N, K=a.K, Rlink=a.Rlink, lr_c=a.lr_c, T=a.T,
        dist=a.dist, isPlot=a.plot, seed=a.seed, alpha=a.alpha
    )