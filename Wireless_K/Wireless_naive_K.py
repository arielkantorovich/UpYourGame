"""
Created on : ------

@author: Ariel_Kantorovich
"""
from common.wireless_main import *
from common.wireless_common import *
from DNN_common.NN_common import *


if __name__ == '__main__':
    # Defin Basic Parameters
    cfg = parse_args()
    rec_NE = SimRecord.create(cfg)
    rec_opt = SimRecord.create(cfg)
    rec_DCPA = SimRecord.create(cfg)
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

    P_exp = P_init.copy()
    P_DCPA = P_init.copy()

    # Exploration process
    cfg_exp = parse_args()
    cfg_exp.T = 100
    lr_exp = np.ones((cfg_exp.T,)) * cfg_exp.lr_c
    rec_exp = SimRecord.create(cfg_exp)
    input_pre = Exploration_Process(cfg_exp, rec_exp, g_struct, P_exp, lr)
    alpha_k, beta_k = wireless_NN_predict(L=cfg_exp.L, N=cfg_exp.N, input_path=cfg.train_path, cfg_name="train_config.yaml",
                                          device="cpu", inputs_pre=input_pre) # (L, N, K)
    alpha_k.setflags(write=False)
    beta_k.setflags(write=False)

    # Run Main loops
    main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    main_loop(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)
    main_loop(cfg, rec_DCPA, g_struct, P_DCPA, lr, GradMode.PRIOR_APPROXIMATION, alpha_k, beta_k)

    if cfg.isPlot:
        print("============= Visualize Results ==================")
        plot_NE_opt_GAP(cfg.T, rec_NE.obj, rec_opt.obj, rec_DCPA.obj)
