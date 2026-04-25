"""
Energy Consumption Game -- Online Simulation
=============================================
Compares three gradient modes:
  1. **Nash** -- selfish NE gradient only.
  2. **DCPA** -- NE gradient + NN-estimated residual (our method).
  3. **Global** -- NE gradient + true residual (oracle upper bound).

Usage
-----
    python Energy_sim.py --model_path Energy_NetPath(N=5).pth \\
                         --hyper_path N=5_energey_game_uniform \\
                         --N 5 --L 100 --plot

@author: Ariel Kantorovich
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Allow running from both repo root  (`python EnergyGame/Energy_sim.py`)
# and from inside EnergyGame/        (`python Energy_sim.py`)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EnergyGame.common.energy_common import (
    project_onto_simplex,
    calculate_NE_gradient,
    calculate_residual_gradient,
)
from EnergyGame.common.energy_data_structure import parse_args, SimConfig
from EnergyGame.common.energy_plots import plot_energy_results
from EnergyGame.dnn_utils.energy_nn import Nash_Filter_NN


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_parameters(L: int, N: int, K: int, alpha: float, beta: float,
                      dist: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample game parameters A_k, B_k.

    Returns
    -------
    A_k, B_k : each (L, N, K)
    """
    if dist == "uniform":
        A_k = alpha * np.random.uniform(low=0.1, high=1.8, size=(L, K))
        B_k = beta * np.random.uniform(low=0.0, high=5.0, size=(L, K))
    elif dist == "exponential":
        mean_a = (0.1 + 1.8) / 2
        mean_b = 2.5
        A_k_raw = np.random.exponential(scale=mean_a, size=(L, K))
        A_k = alpha * np.clip(A_k_raw, 0.1, 1.8)
        B_k_raw = np.random.exponential(scale=mean_b, size=(L, K))
        B_k = beta * np.clip(B_k_raw, 0.0, 5.0)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    A_k = np.repeat(A_k[:, np.newaxis, :], N, axis=1)
    B_k = np.repeat(B_k[:, np.newaxis, :], N, axis=1)
    return A_k, B_k


# ---------------------------------------------------------------------------
# Game loops
# ---------------------------------------------------------------------------

def main_loop(cfg: SimConfig, Xn_k: np.ndarray,
              A_k: np.ndarray, B_k: np.ndarray,
              is_global: bool = False):
    """
    Run gradient ascent for Nash or Global mode.

    Returns
    -------
    final_x, reward, grad, std
    """
    L, N, K = Xn_k.shape
    T = cfg.T
    lr = cfg.lr * np.ones((T,))
    reward = np.zeros((T, N))
    std = np.zeros((T, N))
    grad_record = np.zeros((T, N))
    grad_global = np.zeros_like(Xn_k)

    for t in range(T):
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)

        V = cfg.const_V * np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        r_n = np.sum(V - P, axis=2)
        reward[t] = np.mean(r_n, axis=0)
        std[t] = np.std(r_n, axis=0)

        grad_NE = calculate_NE_gradient(cfg.const_V, Xn_k, A_k, B_k, Sk)
        if is_global:
            grad_global = calculate_residual_gradient(Xn_k, A_k, B_k, Sk, N)

        Xn_k = Xn_k + lr[t] * (grad_NE + grad_global)
        Xn_k = np.clip(Xn_k, a_min=0, a_max=cfg.gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=cfg.gamma_n)

    return Xn_k, reward, grad_record, std


def main_loop_dcpa(cfg: SimConfig, Xn_k: np.ndarray,
                   A_k: np.ndarray, B_k: np.ndarray):
    """
    Run gradient ascent using the NN-estimated residual (DCPA).

    Returns
    -------
    final_x, reward
    """
    L, N, K = Xn_k.shape
    T = cfg.T
    lr = cfg.lr * np.ones((T,))
    reward = np.zeros((T, N))

    # --- Load NN and run exploration / estimation --------------------------
    weights_dir = os.path.join(_SCRIPT_DIR, cfg.weights_dir)
    hyper_dir = os.path.join(_SCRIPT_DIR, cfg.hyper_dir)

    nf = Nash_Filter_NN(L, N, K, A_k[:, 0, :], B_k[:, 0, :], cfg.const_V)
    nf.Load_NN_Model(cfg.model_path, weights_dir=weights_dir,
                     input_size=23, output_size=2)
    x_train = nf.Exploration_process(cfg.T_exploration,
                                     pathN=cfg.hyper_path,
                                     hyper_dir=hyper_dir)
    Ak_est, Bk_est = nf.Estimate_ak_bk(x_train)

    print(
        f"MSE: Ak_error = "
        f"{np.sum((Ak_est - A_k) ** 2) / (L * N * K):.6f}    "
        f"Bk_error = "
        f"{np.sum((Bk_est - B_k) ** 2) / (L * N * K):.6f}"
    )

    # --- Game loop ---------------------------------------------------------
    for t in range(T):
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)

        V = cfg.const_V * np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        r_n = np.sum(V - P, axis=2)
        reward[t] = np.mean(r_n, axis=0)

        Sk_est = nf.Get_sk(Ak_est, Bk_est, Xn_k, P)
        grad_NE = calculate_NE_gradient(cfg.const_V, Xn_k, Ak_est, Bk_est, Sk_est)
        grad_res = calculate_residual_gradient(Xn_k, Ak_est, Bk_est, Sk_est, N)

        Xn_k = Xn_k + lr[t] * (grad_NE + grad_res)
        Xn_k = np.clip(Xn_k, a_min=0, a_max=cfg.gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=cfg.gamma_n)

    return Xn_k, reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    A_k, B_k = sample_parameters(
        cfg.L, cfg.N, cfg.K, cfg.alpha, cfg.beta, cfg.dist
    )
    Xn_k_init = np.random.uniform(low=0.0, high=1.0, size=(cfg.L, cfg.N, cfg.K))

    # ---- DCPA (our method) ------------------------------------------------
    if cfg.model_path is not None and cfg.hyper_path is not None:
        _, reward_dcpa = main_loop_dcpa(cfg, Xn_k_init.copy(), A_k, B_k)
    else:
        reward_dcpa = np.zeros((cfg.T, cfg.N))
        print("[WARN] --model_path and --hyper_path not provided; skipping DCPA.")

    # ---- Nash (selfish) ---------------------------------------------------
    _, reward_NE, _, std_NE = main_loop(
        cfg, Xn_k_init.copy(), A_k, B_k, is_global=False
    )

    # ---- Global (oracle) --------------------------------------------------
    _, reward_global, _, _ = main_loop(
        cfg, Xn_k_init.copy(), A_k, B_k, is_global=True
    )

    # ---- Plot -------------------------------------------------------------
    if cfg.isPlot:
        plot_energy_results(
            reward_NE, reward_global, reward_dcpa,
            N=cfg.N, std_NE=std_NE, debug=cfg.debug,
        )

    print("Energy game simulation finished.")


if __name__ == "__main__":
    main()
