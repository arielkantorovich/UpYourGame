"""
Created on : ------

@author: Ariel_Kantorovich
"""

from .wireless_common import *
from .wireless_data_structure import *
from .wireless_plots import *


def main_loop(cfg: SimConfig, rec: SimRecord, G: Sim_G, P: np.ndarray, lr: np.ndarray, grad_mode: GradMode) -> None:
    """
    Main simulation entry point.

    Parameters
    ----------
    P : (L, N, K) Initialize power transmission
    lr: (T, ) learning rate factor
    grad_mode: Enum gradient indicator (0 - naive nash approach,
                                            1 - Optimal solution
                                            2 - Prior Approximation from DNN)

    G: Sim_G
        Include constant channel gain matrices
        - g: array in size (L, N, N, K) or (L, K, N, N)
        - g_diag: diagonal of g
        - g_zero: zero diagonal of g


    rec: SimRecord
        Simulation record Parameters
        - P: array storing powers over time
        - obj: objective values over time
        - grad_norm: gradient norms over time

    cfg : SimConfig
        Simulation configuration containing:
        - L : number of trials
        - N : number of players
        - K : number of frequency channels
        - Rlink : receiver link radius
        - T : number of iterations
        - N0 : White Gaussian Noise parameter
        - dist : channel distribution
        - alpha : path loss constant
        - seed : RNG seed (optional)
    """

    # Define Init condition in gradients
    gradients_residual = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)
    gradients_ml = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)

    for t in range(cfg.T):
        In = compute_interference(G.g, P, G.g_diag)
        gradients_local = calc_local_gradient(G.g_diag, In, cfg.N0, P)
        if grad_mode == GradMode.NAIVE_NASH:
            pass
        elif grad_mode == GradMode.OPTIMAL:
            gradients_residual = calc_residual_gradient(G.g_diag, G.g_zero, In, cfg.N0, P, gradients_local)
        elif grad_mode == GradMode.PRIOR_APPROXIMATION:
            gradients_ml.fill(0.0) # TO DO
        else:
            raise ValueError(f"Grad Mode {grad_mode} not recognized.")

        P = P + lr[t] * (gradients_local + gradients_residual + gradients_ml)
        project_box(P, cfg.Border_floor, cfg.Border_ceil)

        rec.P[t] = P
        rec.obj[t] = compute_objective(In, G.g_diag, P, cfg.N0)