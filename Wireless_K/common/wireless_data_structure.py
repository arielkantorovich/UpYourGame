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
    SaveToTrain: bool = False
    isValid: bool = False
    alpha: float = 10e-3
    Border_floor: float = 0.0
    Border_ceil: float = 1.0
    N0: float = 0.001
    train_path: str = None
    seed: int | None = None


@dataclass(slots=True)
class SimRecord:
    """
    Simulation Record
    P size (T, L, N, K)
    obj size (T,)
    grad_norm_prior size (T, L, N, K)
    In size (T, L, N, K)
    """
    P: np.ndarray
    obj: np.ndarray
    grad_norm_prior: np.ndarray
    In: np.ndarray

    @classmethod
    def create(cls, cfg: "SimConfig") -> "SimRecord":
        """
        Refers to the class itself, Used in class methods.
        :Note: cls is SimRecord
        :param cfg:
        :return:
        """
        # choose shapes that match what you want to record
        P = np.zeros((cfg.T, cfg.L, cfg.N, cfg.K), dtype=np.float32)
        obj = np.zeros((cfg.T,), dtype=np.float32)
        grad_norm = np.zeros((cfg.T, cfg.L, cfg.N, cfg.K), dtype=np.float32)
        In = np.zeros((cfg.T, cfg.L, cfg.N, cfg.K), dtype=np.float32)
        return cls(P=P, obj=obj, grad_norm_prior=grad_norm, In=In)


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
    p.add_argument("--p_max", type=float, default=1.0, help="border_ceill maximum power projection")
    p.add_argument("--lr_c", type=float, default=0.002, help="Learning rate coefficient for the gradient update (default: 0.002)")
    p.add_argument("--T", type=int, default=2000)
    p.add_argument("--dist", type=str, default="uniform", choices=["uniform", "normal"])
    p.add_argument("--plot", action="store_true", help="Enable plotting")
    p.add_argument("--save_train", action="store_true", help="Enable saving data to train DCPA")
    p.add_argument("--valid", action="store_true", help="Enable saving data to train DCPA")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train_path", type=str, default=None, help="Path to training data e.g Training_data/results")

    # optional extras
    p.add_argument("--alpha", type=float, default=10e-3)

    a = p.parse_args()
    return SimConfig(
        L=a.L, N=a.N, K=a.K, Rlink=a.Rlink, lr_c=a.lr_c, T=a.T,
        dist=a.dist, isPlot=a.plot, SaveToTrain=a.save_train, isValid=a.valid, seed=a.seed, alpha=a.alpha, Border_ceil=a.p_max,
        train_path=a.train_path
    )