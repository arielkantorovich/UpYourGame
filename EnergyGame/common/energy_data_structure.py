"""
Energy Game -- Data Structures & CLI
=====================================
SimConfig dataclass and argparse helper, following the same pattern
as ``QuadraticGames/utils/data_structure.py`` and
``Wireless_K/common/wireless_data_structure.py``.

@author: Ariel Kantorovich
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import argparse
import numpy as np


class GradMode(IntEnum):
    NAIVE_NASH = 0
    OPTIMAL = 1
    PRIOR_APPROXIMATION = 2


@dataclass(slots=True)
class SimConfig:
    """All simulation parameters for the energy consumption game."""
    L: int = 100                     # Number of game instances
    N: int = 5                       # Number of players
    K: int = 24                      # Number of resources (e.g. hours in a day)
    T: int = 300                     # Number of gradient-ascent iterations
    const_V: float = 25.0            # Scaling constant for log-utility
    alpha: float = 1.5               # Scaling factor for A_k
    beta: float = 0.97               # Scaling factor for B_k
    gamma_n_k: float = 7.35          # Per-resource energy upper bound
    gamma_n: float = 15.55           # Total energy budget (simplex radius)
    lr: float = 0.0005               # Learning rate
    dist: str = "uniform"            # Distribution: "uniform" or "exponential"
    T_exploration: int = 5           # Exploration half-width (actual range: -T to T)
    T_loss: int = 200                # Path-loss trajectory length
    isPlot: bool = False             # Enable plotting
    debug: bool = False              # Debug mode
    model_path: Optional[str] = None # Path to trained .pth weights
    hyper_path: Optional[str] = None # Path to normalisation stats folder
    weights_dir: str = "Weights"     # Directory containing .pth files
    hyper_dir: str = "hyperparameters"  # Directory containing normalisation .npy


@dataclass(slots=True)
class SimRecord:
    """Stores simulation outputs for one gradient mode."""
    reward: np.ndarray       # (T, N) mean reward per player
    std: np.ndarray          # (T, N) std of reward
    grad: np.ndarray         # (T, N) mean gradient (debug)
    final_x: np.ndarray      # (L, N, K) final actions

    @classmethod
    def zeros(cls, L: int, T: int, N: int, K: int) -> "SimRecord":
        return cls(
            reward=np.zeros((T, N)),
            std=np.zeros((T, N)),
            grad=np.zeros((T, N)),
            final_x=np.zeros((L, N, K)),
        )


def parse_args() -> SimConfig:
    """Parse command-line arguments and return a ``SimConfig``."""
    p = argparse.ArgumentParser(description="Energy Consumption Game Simulation")

    p.add_argument("--L", type=int, default=100,
                   help="Number of game instances (default: 100)")
    p.add_argument("--N", type=int, default=5,
                   help="Number of players (default: 5)")
    p.add_argument("--K", type=int, default=24,
                   help="Number of resources (default: 24)")
    p.add_argument("--T", type=int, default=300,
                   help="Number of gradient-ascent iterations (default: 300)")
    p.add_argument("--lr", type=float, default=0.0005,
                   help="Learning rate (default: 0.0005)")
    p.add_argument("--dist", type=str, default="uniform",
                   choices=["uniform", "exponential"],
                   help="Parameter sampling distribution (default: uniform)")
    p.add_argument("--T_exploration", type=int, default=5,
                   help="Exploration half-width, range is [-T, T] (default: 5)")
    p.add_argument("--T_loss", type=int, default=200,
                   help="Path-loss trajectory length (default: 200)")
    p.add_argument("--model_path", type=str, default=None,
                   help="Filename of the trained .pth weights")
    p.add_argument("--hyper_path", type=str, default=None,
                   help="Folder name under hyperparameters/ with X_mean.npy and X_std.npy")
    p.add_argument("--plot", action="store_true",
                   help="Enable plotting")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode (NE and Global only, skip DCPA)")

    a = p.parse_args()

    return SimConfig(
        L=a.L,
        N=a.N,
        K=a.K,
        T=a.T,
        lr=a.lr,
        dist=a.dist,
        T_exploration=a.T_exploration,
        T_loss=a.T_loss,
        model_path=a.model_path,
        hyper_path=a.hyper_path,
        isPlot=a.plot,
        debug=a.debug,
    )
