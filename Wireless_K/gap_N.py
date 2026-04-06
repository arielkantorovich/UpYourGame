"""
Created on : ------
@brief: The following code simulate the gap between optimal and NE when N grows
@author: Ariel_Kantorovich
"""

from common.wireless_main import *


def main(N_list: list, cfg: SimConfig, learning_rate: np.ndarray) -> None:
    """
    Main simulation entry point.
    :param N_list:
    :param cfg: SimConfig
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
    :param learning_rate:
    :return:
    """
    diff_prec_fun_N = np.zeros((len(N_list), ))

    for i, N in enumerate(N_list):
        # Define Constant Var's
        cfg.N = int(N)
        print(f"N={cfg.N}")
        rec_NE = SimRecord.create(cfg)
        rec_opt = SimRecord.create(cfg)

        g_struct = set_g_struct(cfg=cfg)

        # Define Initialize condition
        P_init = cfg.Border_ceil * np.random.rand(cfg.L, cfg.N, cfg.K)
        P_init.setflags(write=False)
        P_NE = P_init.copy()
        P_opt = P_init.copy()

        # Run Main loops
        main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
        main_loop(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)

        # Calculate the main results
        final_iter = cfg.T - 1
        final_opt = rec_opt.obj[final_iter]
        final_ne = rec_NE.obj[final_iter]
        eps = 1e-12
        diff_percentage = 100.0 * (final_opt - final_ne) / (final_opt + eps)

        diff_prec_fun_N[i] = diff_percentage

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        plot_GAP_N(N_list, diff_prec_fun_N)
    print("============= Finsh Run gap_N simulation===============")


if __name__ == "__main__":
    # Defin Basic Parameters
    N_list = [5, 15, 30, 50, 80]
    cfg = parse_args()
    print(cfg)
    lr = np.ones((cfg.T,)) * cfg.lr_c

    main(N_list, cfg, lr)