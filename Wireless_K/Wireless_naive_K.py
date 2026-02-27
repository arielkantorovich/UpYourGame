"""
Created on : ------

@author: Ariel_Kantorovich
"""
from common.wireless_main import *
from common.wireless_common import *
from DNN_common.NN_common import *
from dataclasses import fields
import yaml


def _save_run_config(cfg: SimConfig) -> None:
    """Save current simulation config to <train_path>/config_run.yaml."""
    if cfg.train_path is None:
        return
    out_path = Path(cfg.train_path) / "config_run.yaml"
    cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(SimConfig)}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"sim_config": cfg_dict}, f, sort_keys=False)


if __name__ == "__main__":

    # ------------------------
    # 1) Parse config & setup
    # ------------------------
    cfg = parse_args()
    print("================================ Running Wireless Simulation ===============================")
    print(cfg)

    if not cfg.isDebug:
        _save_run_config(cfg)

    g_struct = set_g_struct(cfg)
    lr = np.full(cfg.T, cfg.lr_c)
    lr_AlphaBeta = np.full(cfg.T, cfg.lr_c/cfg.lr_dec_factor)

    # Initial powers (shared)
    P_init = cfg.Border_ceil * np.random.rand(cfg.L, cfg.N, cfg.K)
    P_init.setflags(write=False)

    # ------------------------
    # 2) Exploration + NN
    # ------------------------
    if not cfg.isDebug:
        print("================================ Running Exploration =======================================")

        cfg_exp = parse_args()
        cfg_exp.T = 100
        lr_exp = np.full(cfg_exp.T, cfg_exp.lr_c)

        rec_exp = SimRecord.create(cfg_exp)
        P_exp = P_init.copy()

        inputs_pre = Exploration_Process(cfg_exp, rec_exp, g_struct, P_exp, lr_exp)

        alpha_k, beta_k = wireless_NN_predict(
            input_path=cfg.train_path,
            isAlphaBeta=True,
            cfg_name="train_config.yaml",
            device="cpu",
            inputs_pre=inputs_pre,
            L=cfg_exp.L,
            N=cfg_exp.N,
        )

        alpha_k.setflags(write=False)
        beta_k.setflags(write=False)

        only_alpha_k, _ = wireless_NN_predict(
            input_path=cfg.train_path,
            isAlphaBeta=False,
            cfg_name="train_config.yaml",
            device="cpu",
            inputs_pre=inputs_pre,
            L=cfg_exp.L,
            N=cfg_exp.N,
        )
    # ------------------------
    # 3) Main simulation runs
    # ------------------------
    print("================================ Running Main Simulations ==================================")
    runs = {
        "NE": {"grad_mode": GradMode.NAIVE_NASH, "lr": lr, "alpha": None, "beta": None},
        "OPT": {"grad_mode": GradMode.OPTIMAL, "lr": lr, "alpha": None, "beta": None},
    }

    if not cfg.isDebug:
        runs.update({
            "DCPA-alphaBeta": {
                "grad_mode": GradMode.PRIOR_APPROXIMATION,
                "lr": lr_AlphaBeta,
                "alpha": alpha_k,
                "beta": beta_k,
            },
            "DCPA-alpha": {
                "grad_mode": GradMode.PRIOR_APPROXIMATION,
                "lr": lr,
                "alpha": only_alpha_k,
                "beta": np.zeros_like(only_alpha_k),
            }
        })

    results = {}

    for name, params in runs.items():
        print(f"--- Running {name} ---")
        results[name] = run_simulation(
            cfg=cfg,
            g_struct=g_struct,
            P_init=P_init,
            lr=params["lr"],
            grad_mode=params["grad_mode"],
            alpha_k=params["alpha"],
            beta_k=params["beta"],
        )

    # ------------------------
    # 4) Visualization
    # ------------------------
    if cfg.isPlot:
        print("================================ Visualizing Results ======================================")
        plot_NE_opt_GAP(
            cfg.T,
            cfg.isDebug,
            results["NE"].obj,
            results["OPT"].obj,
            np.zeros_like(results["NE"].obj) if cfg.isDebug else results["DCPA-alphaBeta"].obj,
            np.zeros_like(results["NE"].obj) if cfg.isDebug else results["DCPA-alpha"].obj,
        )

    # --------------------------------------------------------------
    # 5) Save Final results in percentages from optimal solution
    # --------------------------------------------------------------
        if not cfg.isDebug:
            # Calculate the main results
            final_iter = cfg.T - 1
            final_opt = results["OPT"].obj[final_iter]

            # Map output keys -> results dict keys
            methods = {
                "NE": "NE",
                "alpha": "DCPA-alpha",
                "alphaBeta": "DCPA-alphaBeta",
            }

            gaps = {
                out_key: gap_percent(final_opt, results[res_key].obj[final_iter])
                for out_key, res_key in methods.items()
            }

            out_path = Path(cfg.train_path) / "inference_results.npz"
            np.savez(out_path, L=cfg.L, N=cfg.N, K=cfg.K, **gaps)

            print("================================ Save Results ======================================")
