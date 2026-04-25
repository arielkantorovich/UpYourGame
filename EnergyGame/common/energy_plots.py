"""
Energy Game -- Plotting Utilities
=================================

@author: Ariel Kantorovich
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_energy_results(
    reward_NE: np.ndarray,
    reward_global: np.ndarray,
    reward_dcpa: np.ndarray,
    N: int,
    std_NE: np.ndarray | None = None,
    plot_std: bool = False,
    debug: bool = False,
) -> None:
    """
    Plot total reward over iterations for Nash, Global, and DCPA.

    Parameters
    ----------
    reward_NE     : (T, N)  mean per-player reward for NE.
    reward_global : (T, N)  mean per-player reward for Global.
    reward_dcpa   : (T, N)  mean per-player reward for DCPA.
    N             : int     number of players.
    std_NE        : (T, N)  optional standard deviation for NE.
    plot_std      : bool    whether to draw std bands.
    debug         : bool    if True, skip DCPA curve.
    """
    T = reward_NE.shape[0]
    t = np.arange(T)

    total_NE = np.sum(reward_NE, axis=1)
    total_global = np.sum(reward_global, axis=1)
    total_dcpa = np.sum(reward_dcpa, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(t, total_NE, color="b", label="Nash")
    if plot_std and std_NE is not None:
        total_std = np.mean(std_NE, axis=1)
        plt.fill_between(
            t, total_NE - total_std, total_NE + total_std,
            color="b", alpha=0.2,
        )
    if not debug:
        plt.plot(t, total_dcpa, color="gold", linestyle="-", label="DCPA")
    plt.plot(t, total_global, "--k", label="Global")
    plt.ylim(np.max(total_NE) - 200, np.max(total_global) + 100)
    plt.xlabel("Iteration")
    plt.ylabel("Total Reward")
    plt.title(f"Energy Consumption Game (N={N})")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
