"""
Created on : ------

@author: Ariel_Kantorovich
"""

from dataclasses import dataclass
import argparse
from enum import IntEnum
import numpy as np



@dataclass(slots=True)
class SimConfig:
    """
    Simulation Parameters
    """
    L: int = 800
    N: int = 5
    lr: float = 0.01
    T: int = 1000
    alpha: float = 1.0
    beta: float = 0.2
    T_exploration: int = 2000
    qnn_low: float = 1.2
    qnn_high: float = 2.2
    action_project_low: float = -20.0
    action_project_high: float = 20.0
    isPlot: bool = False
    debug: bool = False


class GradMode(IntEnum):
    NAIVE_NASH = 0
    OPTIMAL = 1
    PRIOR_APPROXIMATION = 2


@dataclass(slots=True)
class SimRecord:
    """
    Stores simulation outputs for one gradient mode.
    """
    cost_record: np.ndarray
    grad_record: np.ndarray
    sum_cost: np.ndarray
    mean_cost: np.ndarray
    mean_grad: np.ndarray
    final_x: np.ndarray

    @classmethod
    def zeros(cls, L: int, T: int, N: int) -> "SimRecord":
        return cls(
            cost_record=np.zeros((L, T, N)),
            grad_record=np.zeros((L, T, N)),
            sum_cost=np.zeros((L, T)),
            mean_cost=np.zeros(T),
            mean_grad=np.zeros((T, N)),
            final_x=np.zeros((L, N, 1)),
        )


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Quadratic Game Parameters Simulation")

    p.add_argument("--L", type=int, default=800)
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--T_exploration", type=int, default=2000)
    p.add_argument("--qnn_low", type=float, default=1.2)
    p.add_argument("--qnn_high", type=float, default=2.2)
    p.add_argument("--plot", action="store_true", help="Enable plotting")
    p.add_argument("--debug", action="store_true", help="Plot only Nash and Optimal curves")
    p.add_argument("--action_project_low", type=float, default=-20.0)
    p.add_argument("--action_project_high", type=float, default=20.0)
    a = p.parse_args()

    return SimConfig(
        L=a.L,
        N=a.N,
        lr=a.lr,
        T=a.T,
        alpha=a.alpha,
        beta=a.beta,
        T_exploration=a.T_exploration,
        qnn_low=a.qnn_low,
        qnn_high=a.qnn_high,
        action_project_low=a.action_project_low,
        action_project_high=a.action_project_high,
        isPlot=a.plot,
        debug=a.debug,
    )
