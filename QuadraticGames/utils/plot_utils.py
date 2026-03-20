import numpy as np
import matplotlib.pyplot as plt


def plot_quadratic_mean_cost(
    ne_mean_cost: np.ndarray,
    optimal_mean_cost: np.ndarray,
    dcpa_mean_cost: np.ndarray,
    num_players: int,
    non_symmetric: bool = False,
    debug: bool = False,
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
    num_players : int
        Number of players in the simulation.
    non_symmetric : bool, optional
        If True, label the plot as non-symmetric.
    debug : bool, optional
        If True, plot only Nash and Optimal.
    """
    t = np.arange(ne_mean_cost.shape[0])
    symmetry_label = "Non-Symmetric" if non_symmetric else "Symmetric"
    plt.figure()
    plt.plot(t, optimal_mean_cost, "--k", label="Optimal")
    plt.plot(t, ne_mean_cost, "r", label="NE")
    if not debug:
        plt.plot(t, dcpa_mean_cost, "b", label="DCPA")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Quadratic Game (N={num_players}, {symmetry_label})")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
