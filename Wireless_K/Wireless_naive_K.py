"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np

from common.wireless_common import *
from common.wireless_data_structure import *
import matplotlib.pyplot as plt


def main(cfg: SimConfig, rec: SimRecord, P: np.ndarray, lr: np.ndarray, grad_mode: int) -> None:
    """
    Main simulation entry point.

    Parameters
    ----------
    P : (L, N, K) Initialize power transmission
    lr: (T, ) learning rate factor
    grad_mode: int gradient indicator (0 - naive nash approach,
                                            1 - Optimal solution
                                            2 - Prior Approximation from DNN)

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
        - alpha : pathloss constant
        - seed : RNG seed (optional)
    """
    # Define Constant Var's
    g = generate_gain_channel(N=cfg.N, L=cfg.L, K=cfg.K, alpha=cfg.alpha, R_link=cfg.Rlink, distribution=cfg.dist)
    g.setflags(write=False) # Constant array

    g_diag = extract_g_diag(g)
    g_diag.setflags(write=False)

    eye = np.eye(cfg.N, dtype=bool)[None, :, :, None]  # (1, N, N, 1)
    g_zero = np.where(eye, 0.0, g)
    g_zero.setflags(write=False)

    # Define Init condition in gradients
    gradients_local = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)
    gradients_residual = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)
    gradients_ml = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)

    for t in range(cfg.T):
        In = compute_interference(g, P, g_diag)
        gradients_local = calc_local_gradient(g_diag, In, cfg.N0, P)
        if grad_mode == GradMode.NAIVE_NASH:
            pass
        elif grad_mode == GradMode.OPTIMAL:
            gradients_residual = calc_residual_gradient(g_diag, g_zero, In, cfg.N0, P, gradients_local)
        elif grad_mode == GradMode.PRIOR_APPROXIMATION:
            gradients_ml = 0.0
        else:
            raise ValueError(f"Grad Mode {grad_mode} not recognized.")

        P = P + lr[t] * (gradients_local + gradients_residual + gradients_ml)
        project_box(P, cfg.Border_floor, cfg.Border_ceil)

        rec.P[t] = P
        rec.obj[t] = compute_objective(In, g_diag, P, cfg.N0)





if __name__ == '__main__':
    # Defin Basic Parameters
    cfg = parse_args()
    rec_NE = SimRecord.create(cfg)
    rec_opt = SimRecord.create(cfg)
    lr = np.ones((cfg.T, )) *  0.002

    print("================================Running main Wireless_naive_K ===========================")
    print(f"Parameters:")
    print(cfg)
    print(f"Record parameters:\nP.shape{rec_opt.P.shape}\nobj.shape{rec_opt.obj.shape}\ngrad_norm.shape{rec_opt.grad_norm.shape}")

    # Define Initialize condition
    P_init = np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)
    P_NE = P_init.copy()
    P_opt = P_init.copy()

    print("===== Call Main Loop For Nash ====")
    main(cfg, rec_NE, P_NE, lr, GradMode.NAIVE_NASH)
    print("===== Finsh Main====")

    print("===== Call Main Loop For Nash ====")
    main(cfg, rec_opt, P_opt, lr, GradMode.OPTIMAL)
    print("===== Finsh Main====")

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        t = np.arange(cfg.T)
        plt.figure(1)
        plt.plot(t, rec_opt.obj, label="opt")
        plt.plot(t, rec_NE.obj, label="NE")
        plt.legend()
        plt.xlabel("# Iteration")
        plt.ylabel("# Obj")
        plt.show()

        print("============= Finish Visualize ==================")

    print(f"===================Finsh Wireless K = {cfg.K} Simulation ==================")


