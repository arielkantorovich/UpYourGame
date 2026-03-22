import numpy as np
import matplotlib.pyplot as plt


def plot_quadratic_mean_cost(
    ne_mean_cost: np.ndarray,
    optimal_mean_cost: np.ndarray,
    dcpa_mean_cost: np.ndarray,
    optimal_std_cost: np.ndarray,
    ne_std_cost: np.ndarray,
    dcpa_std_cost: np.ndarray,
    num_players: int,
    asymmetry_pct: float = 0.0,
    non_symmetric: bool = False,
    plot_std: bool = False,
    debug: bool = False,
    y_min: float = 0.0,
    y_max: float = 0.0,
) -> None:
    """
    Plot the mean cost curves of the quadratic simulation.

    Parameters
    ----------
    ne_mean_cost : np.ndarray
        Mean cost curve of the Nash equilibrium dynamics.
    optimal_mean_cost : np.ndarray
        Mean cost curve of the optimal dynamics.
    dcpa_mean_cost : np.ndarray
        Mean cost curve of the DCPA dynamics.
    optimal_std_cost : np.ndarray
        Standard deviation of the total optimal cost across games at each iteration.
    ne_std_cost : np.ndarray
        Standard deviation of the total NE cost across games at each iteration.
    dcpa_std_cost : np.ndarray
        Standard deviation of the total DCPA cost across games at each iteration.
    num_players : int
        Number of players in the simulation.
    asymmetry_pct : float, optional
        Relative percentage distance from symmetry, shown as ``Delta`` in the title.
    non_symmetric : bool, optional
        If True, label the plot as non-symmetric.
    plot_std : bool, optional
        If True, draw shaded standard deviation bands around the mean curves.
    debug : bool, optional
        If True, plot only Nash and Optimal.
    y_min : float, optional
        Minimum y-axis value. If 0, the lower bound is left unchanged.
    y_max : float, optional
        Maximum y-axis value. If 0, the upper bound is left unchanged.
    """
    t = np.arange(ne_mean_cost.shape[0])
    symmetry_label = "Non-Symmetric" if non_symmetric else "Symmetric"
    plt.figure()
    plt.plot(t, optimal_mean_cost, "--k", label="Optimal")
    if plot_std:
        plt.fill_between(
            t,
            optimal_mean_cost - optimal_std_cost,
            optimal_mean_cost + optimal_std_cost,
            color="k",
            alpha=0.12,
            linewidth=0,
        )
    plt.plot(t, ne_mean_cost, "r", label="NE")
    if plot_std:
        plt.fill_between(
            t,
            ne_mean_cost - ne_std_cost,
            ne_mean_cost + ne_std_cost,
            color="r",
            alpha=0.18,
            linewidth=0,
        )
    if not debug:
        plt.plot(t, dcpa_mean_cost, "b", label="DCPA")
        if plot_std:
            plt.fill_between(
                t,
                dcpa_mean_cost - dcpa_std_cost,
                dcpa_mean_cost + dcpa_std_cost,
                color="b",
                alpha=0.18,
                linewidth=0,
            )
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(
        rf"Quadratic Game (N={num_players}, {symmetry_label}, $\Delta={asymmetry_pct:.2f}\%$)"
    )
    current_ymin, current_ymax = plt.ylim()
    plt.ylim(
        bottom=y_min if y_min != 0.0 else current_ymin,
        top=y_max if y_max != 0.0 else current_ymax,
    )
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
