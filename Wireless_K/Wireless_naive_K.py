"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np

from common.wireless_common import *
from common.wireless_data_structure import *
from common.wireless_plots import *


def main(cfg: SimConfig, rec: SimRecord, G: Sim_G, P: np.ndarray, lr: np.ndarray, grad_mode: GradMode) -> None:
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





if __name__ == '__main__':
    # Defin Basic Parameters
    cfg = parse_args()
    rec_NE = SimRecord.create(cfg)
    rec_opt = SimRecord.create(cfg)
    lr = np.ones((cfg.T, )) *  cfg.lr_c

    print("================================Running main Wireless_naive_K ===========================")
    print(f"Parameters:")
    print(cfg)

    # Define Constant Var's
    g = generate_gain_channel(N=cfg.N, L=cfg.L, K=cfg.K, alpha=cfg.alpha, R_link=cfg.Rlink, distribution=cfg.dist)
    g.setflags(write=False) # Constant array

    g_diag = extract_g_diag(g)
    g_diag.setflags(write=False)

    eye = np.eye(cfg.N, dtype=bool)[None, :, :, None]  # (1, N, N, 1)
    g_zero = np.where(eye, 0.0, g)
    g_zero.setflags(write=False)

    g_struct = Sim_G(g=g, g_diag=g_diag, g_zero=g_zero)

    # Define Initialize condition
    P_init = cfg.Border_ceil * np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)
    P_NE = P_init.copy()
    P_opt = P_init.copy()

    # Run Main loops
    main(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    main(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        plot_NE_opt_GAP(cfg.T, rec_NE.obj, rec_opt.obj)
