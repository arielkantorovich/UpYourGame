"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_NE_opt_GAP(T: int, obj_NE: np.ndarray, obj_opt: np.ndarray) -> None:
    """
    The following function plot simulation results and gap between NE and OPT
    :param T:
    :param obj_NE: (T, ) nash objective record
    :param obj_opt: (T, ) optimal objective record
    :return: None
    """
    final_iter = T - 1
    final_opt = obj_opt[final_iter]
    final_ne = obj_NE[final_iter]

    eps = 1e-12
    diff_percentage = 100.0 * (final_opt - final_ne) / (final_opt + eps)

    t = np.arange(T)

    plt.figure(figsize=(8, 5))
    plt.plot(t, obj_opt, label="OPT")
    plt.plot(t, obj_NE, label="NE")

    # Vertical line BETWEEN NE[T-1] and OPT[T-1]
    plt.vlines(
        x=final_iter,
        ymin=min(final_opt, final_ne),
        ymax=max(final_opt, final_ne),
        colors="black",
        linestyles="dashed",
        linewidth=2
    )

    # Put text next to the vertical line
    y_mid = 0.5 * (final_opt + final_ne)
    x_offset = 100
    plt.text(
        final_iter - x_offset,
        y_mid,
        f"{diff_percentage:.2f}%",
        ha="left",
        va="center",
        color="black"
    )

    plt.xlabel("# Iteration")
    plt.ylabel("# Objective")
    plt.title("Final Nash Gap vs Optimal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_GAP_N(N_list: list, diff_prec: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(N_list, diff_prec, linestyle='--', marker='o')
    plt.xlabel("# Number of agents")
    plt.ylabel("# Gap (%)")
    plt.title("Nash Gap vs Optimal")
    plt.show()