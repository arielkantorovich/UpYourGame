"""
Created on : ------

@author: Ariel_Kantorovich
"""
from common.wireless_main import *
from common.wireless_common import *



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
    g_struct = set_g_struct(cfg=cfg)

    # Define Initialize condition
    P_init = cfg.Border_ceil * np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)
    P_NE = P_init.copy()
    P_opt = P_init.copy()
    P_DCPA = P_init.copy()

    # Exploration process
    # cfg_exp = SimConfig()
    # main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    # alpha_k, beta_k = wireless_NN_predict()

    # Run Main loops
    main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    main_loop(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        plot_NE_opt_GAP(cfg.T, rec_NE.obj, rec_opt.obj)
