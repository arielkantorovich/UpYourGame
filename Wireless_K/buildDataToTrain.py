"""
Created on : ------

@author: Ariel_Kantorovich
"""
from common.wireless_main import *
from Wireless_K.DNN_common.wireless_paths import *
from Wireless_K.DNN_common.wireless_dataset import *

def loop(cfg: SimConfig) -> Tuple[SimRecord, SimRecord]:
    """
    :param cfg:
    :return: tuple [SimRecord, SimRecord]
    """
    # Defin Basic Parameters
    rec_NE = SimRecord.create(cfg)
    rec_opt = SimRecord.create(cfg)
    lr = np.ones((cfg.T,)) * cfg.lr_c

    g_struct = set_g_struct(cfg=cfg)

    # Define Initialize condition
    P_init = cfg.Border_ceil * np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)
    P_NE = P_init.copy()
    P_opt = P_init.copy()

    # Run Main loops
    main_loop(cfg, rec_NE, g_struct, P_NE, lr, GradMode.NAIVE_NASH)
    main_loop(cfg, rec_opt, g_struct, P_opt, lr, GradMode.OPTIMAL)

    return rec_NE, rec_opt




if __name__ == '__main__':
    # Defin Basic Parameters
    config = parse_args()

    # Train Loo[
    Train_rec_NE, Train_rec_opt = loop(config)

    # Valid Loop
    config.L = int(config.L * 0.2)
    Valid_rec_NE, Valid_rec_opt = loop(config)


    if config.SaveToTrain:
        print("============= Save Data Results ====================")
        train_dir, valid_dir = make_dataset_dirs("Training_Data", N=config.N, K=config.K)

        X_train, Y_train, _ = build_player_subset_dataset(P=Train_rec_NE.P, In=Train_rec_NE.In, grad=Train_rec_opt.grad_norm_prior, N_subset=5)
        X_valid, Y_valid, _ = build_player_subset_dataset(P=Valid_rec_NE.P, In=Valid_rec_NE.In, grad=Valid_rec_opt.grad_norm_prior, N_subset=5)

        Z_train, _, _ = build_player_subset_dataset(P=Train_rec_opt.P, In=Train_rec_opt.In, grad=Train_rec_opt.grad_norm_prior, N_subset=5)
        Z_valid, _, _ = build_player_subset_dataset(P=Valid_rec_opt.P, In=Valid_rec_opt.In, grad=Valid_rec_opt.grad_norm_prior, N_subset=5)

        save_split_npz(train_dir, X=X_train, z=Z_train, y=Y_train, prefix="data")
        save_split_npz(valid_dir, X=X_valid, z=Z_valid, y=Y_valid, prefix="data")

        print("Saved to:")
        print(" train:", train_dir)
        print(" valid:", valid_dir)

