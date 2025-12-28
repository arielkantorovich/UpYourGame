"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
from wireless_common import *




def main(cfg: SimConfig, P: np.ndarray, lr: np.ndarray):
    """
    Main simulation entry point.

    Parameters
    ----------
    P : (L, N, K) Initialize power transmission
    lr: (T, ) learning rate factor
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

    for t in range(cfg.T):
        In = compute_interference(g, P, g_diag)
        gradients_local = calc_local_gradient(g_diag, In, cfg.N0, P)





if __name__ == '__main__':
    # Defin Basic Parameters
    cfg = parse_args()

    print("================================Running main Wireless_naive_K ===========================")
    print(f"Parameters:")
    print(cfg)

    # Define Initialize condition
    P_init = np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)
    P_NE = P_init.copy()
    P_opt = P_init.copy()

    print("===== Call Main Loop For Nash ====")
    main(cfg, P_NE)
    print("===== Finsh Main====")

    print("===== Call Main Loop For Nash ====")
    main(cfg, P_opt)
    print("===== Finsh Main====")


