"""
Energy Consumption Game -- Build Training Data (Offline Stage)
==============================================================
Generates exploration features (X) and optimal-gradient-descent
path data (Z, Y) for training the DCPA neural network.

Usage
-----
    # Build training data (30 000 games)
    python build_data_to_train.py --N 5 --K 24 --dist 0 --isValid 0

    # Build validation data (3 000 games)
    python build_data_to_train.py --N 5 --K 24 --dist 0 --isValid 1

@author: Ariel Kantorovich
"""

import sys
import os
import numpy as np
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EnergyGame.common.energy_common import (
    project_onto_simplex,
    calculate_NE_gradient,
    calculate_residual_gradient,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build training data for the Energy Consumption Game."
    )
    p.add_argument("--N", type=int, default=5,
                   help="Number of players (default: 5)")
    p.add_argument("--K", type=int, default=24,
                   help="Number of resources (default: 24)")
    p.add_argument("--dist", type=int, default=0,
                   help="Distribution: 0 = uniform, 1 = exponential (default: 0)")
    p.add_argument("--isValid", type=int, default=1,
                   help="1 = validation set (3 000 games), 0 = training set (30 000 games)")
    p.add_argument("--T_exp", type=int, default=5,
                   help="Exploration half-width; range is [-T, T] (default: 5)")
    p.add_argument("--T_loss", type=int, default=200,
                   help="Path-loss trajectory length (default: 200)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: Training_Data/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exploration feature builder
# ---------------------------------------------------------------------------

def build_exploration_features(
    x_init: np.ndarray,
    T_exploration: int,
    L: int,
    N: int,
    K: int,
    A_k: np.ndarray,
    B_k: np.ndarray,
    file_x: str,
    file_y_sanity: str,
):
    """
    Record exploration observations (P, x) while perturbing actions
    by a linear ramp, then save X (features) and Y_sanity (true a_k, b_k).
    """
    Xn_k = x_init.copy()
    X_train = []
    Y_train = [A_k[:, 0, 0].reshape(L, 1), B_k[:, 0, 0].reshape(L, 1)]

    for t in range(-T_exploration, T_exploration + 1):
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        X_train.append(P[:, 0, 0].reshape(L, 1))
        X_train.append(Xn_k[:, 0, 0].reshape(L, 1))
        Xn_k = Xn_k + (1 / N) * t

    X_train = np.concatenate(X_train, axis=-1)
    X_train = np.append(X_train, N * np.ones((L, 1)), axis=1)
    Y_train = np.concatenate(Y_train, axis=-1)
    np.save(file_x, X_train)
    np.save(file_y_sanity, Y_train)
    print(f"Saved exploration features  -> {file_x}")
    print(f"Saved sanity labels         -> {file_y_sanity}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    const_V = 25
    K = args.K
    N = args.N
    T_loss = args.T_loss
    T_exploration = args.T_exp
    isValid = args.isValid
    dist_flag = args.dist

    # Number of games
    if isValid:
        L = 3_000
        state = "valid"
    else:
        L = 30_000
        state = "train"

    # Game parameter sampling
    alpha = 1.5
    beta = 0.97
    gamma_n_k = 7.35
    gamma_n = 15.55

    if dist_flag == 0:
        dist_name = "uniform"
        A_k = alpha * np.random.uniform(low=0.1, high=1.8, size=(L, K))
        B_k = beta * np.random.uniform(low=0.0, high=5.0, size=(L, K))
    elif dist_flag == 1:
        dist_name = "exponential"
        mean_a = (0.1 + 1.8) / 2
        mean_b = 2.5
        A_k_raw = np.random.exponential(scale=mean_a, size=(L, K))
        A_k = alpha * np.clip(A_k_raw, 0.1, 1.8)
        B_k_raw = np.random.exponential(scale=mean_b, size=(L, K))
        B_k = beta * np.clip(B_k_raw, 0.0, 5.0)
    else:
        raise ValueError(f"Unknown dist flag: {dist_flag}")

    A_k = np.repeat(A_k[:, np.newaxis, :], N, axis=1)
    B_k = np.repeat(B_k[:, np.newaxis, :], N, axis=1)

    # Output paths
    pathN = f"N={N}_energy_game_{dist_name}"
    if args.output_dir:
        folder_path = os.path.join(args.output_dir, pathN)
    else:
        folder_path = os.path.join(_SCRIPT_DIR, "Training_Data", pathN)
    os.makedirs(folder_path, exist_ok=True)

    file_x = os.path.join(folder_path, f"X_{state}.npy")
    file_y_sanity = os.path.join(folder_path, f"Y_{state}_sanity.npy")
    file_z = os.path.join(folder_path, f"Z_{state}.npy")
    file_y = os.path.join(folder_path, f"Y_{state}.npy")

    # --- Phase 1: exploration features -------------------------------------
    x_init = np.random.uniform(low=3.1, high=7.0, size=(L, N, K))
    build_exploration_features(
        x_init, T_exploration, L, N, K, A_k, B_k, file_x, file_y_sanity
    )

    # --- Phase 2: optimal-gradient path data -------------------------------
    learning_rate = 0.0005 * np.ones((T_loss,))
    Z_train = np.zeros((L, 2 * T_loss))
    Y_train = np.zeros((L, T_loss))
    Xn_k = x_init.copy()

    for t in range(T_loss):
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)

        grad_NE = calculate_NE_gradient(const_V, Xn_k, A_k, B_k, Sk)
        grad_global = calculate_residual_gradient(Xn_k, A_k, B_k, Sk, N)

        Z_train[:, 2 * t] = Xn_k[:, 0, 0]
        Z_train[:, 2 * t + 1] = Sk[:, 0, 0]
        Y_train[:, t] = grad_global[:, 0, 0]

        Xn_k = Xn_k + learning_rate[t] * (grad_NE + grad_global)
        Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=gamma_n)

    np.save(file_z, Z_train)
    np.save(file_y, Y_train)
    print(f"Saved path features         -> {file_z}")
    print(f"Saved gradient labels       -> {file_y}")
    print("Build data finished.")


if __name__ == "__main__":
    main()
