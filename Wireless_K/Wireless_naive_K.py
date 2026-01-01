"""
Created on : ------

@author: Ariel_Kantorovich
"""
from common.wireless_main import *




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
    main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    main_loop(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        plot_NE_opt_GAP(cfg.T, rec_NE.obj, rec_opt.obj)
