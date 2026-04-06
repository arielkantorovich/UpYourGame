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


def generate_and_save_dataset_in_batches(
    cfg: SimConfig,
    out_dir: str,
    *,
    L_total: int,
    L_batch: int = 500,
    N_subset: int = 5,
    prefix: str = "data",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ensure we actually record P/In/grad in SimRecord
    # (main_loop only fills rec.In / rec.grad_norm_prior if SaveToTrain is True in your code)
    cfg_base = cfg.copy()
    cfg_base.SaveToTrain = True

    shard = 0
    for start in range(0, L_total, L_batch):
        cfg_b = cfg_base.copy()
        cfg_b.L = min(L_batch, L_total - start)

        rec_NE, rec_opt = loop(cfg_b)

        X, y, _ = build_player_subset_dataset(
            P=rec_NE.P, In=rec_NE.In,
            grad=rec_opt.grad_norm_prior,
            N_subset=N_subset,
        )

        z, _, _ = build_player_subset_dataset(
            P=rec_opt.P, In=rec_opt.In,
            grad=rec_opt.grad_norm_prior,
            N_subset=N_subset,
        )

        np.savez(out_dir / f"{prefix}_{shard:05d}.npz", X=X, z=z, y=y)
        shard += 1

        del rec_NE, rec_opt, X, y, z


if __name__ == '__main__':
    # Defin Basic Parameters
    config = parse_args()

    if config.SaveToTrain:
        print("============= Save Data Results ====================")
        train_dir, valid_dir = make_dataset_dirs("Training_Data", N=config.N, K=config.K)

        # TRAIN: generate L samples in batches
        generate_and_save_dataset_in_batches(
            config,
            train_dir,
            L_total=config.L,
            L_batch=500,      # tune
            N_subset=config.N_sub,
            prefix="data",
        )

        # VALID: generate 20% of L in batches (do NOT overwrite config.L globally)
        L_valid = int(config.L * 0.2)
        generate_and_save_dataset_in_batches(
            config,
            valid_dir,
            L_total=L_valid,
            L_batch=500,
            N_subset=config.N_sub,
            prefix="data",
        )

        print("Saved shards to:")
        print(" train:", train_dir)
        print(" valid:", valid_dir)


