"""
Energy Game -- Neural Network & Nash Filters
=============================================
Contains:
  - ``Energy_naive``:    Fully-connected MLP for predicting a_k, b_k.
  - ``Nash_Filter``:     Least-squares parameter estimator.
  - ``Nash_Filter_NN``:  NN-based parameter estimator (extends Nash_Filter).

@author: Ariel Kantorovich
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class Energy_naive(nn.Module):
    """4-hidden-layer FC MLP with Kaiming initialisation."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc0 = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_size)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Least-squares Nash filter
# ---------------------------------------------------------------------------
class Nash_Filter:
    """Estimate game parameters a_k, b_k via least-squares on exploration data."""

    def __init__(self, L: int, N: int, K: int,
                 A_k: np.ndarray, B_k: np.ndarray, const_V: float):
        self.L = L
        self.N = N
        self.K = K
        self.A_k = A_k
        self.B_k = B_k
        self.const_V = const_V

    def Estimate_ak_bk(self, P_list: np.ndarray, X_list: np.ndarray):
        """
        Least-squares estimation of a_k and b_k from exploration data.

        Parameters
        ----------
        P_list : (T_exploration, L, K)
        X_list : (T_exploration, L, K)

        Returns
        -------
        ak, bk : each (L, K)
        """
        T_exploration, L, K = X_list.shape
        X_flat = X_list.reshape(T_exploration, -1)
        P_flat = P_list.reshape(T_exploration, -1)

        A = np.stack(
            [(self.N ** 2) * (X_flat ** 3), self.N * (X_flat ** 2)], axis=2
        )
        A = A.transpose(1, 0, 2)

        coefficients = np.array([
            np.linalg.lstsq(A[i], P_flat[:, i], rcond=None)[0]
            for i in range(A.shape[0])
        ])

        ak = coefficients[:, 0].reshape(L, K)
        bk = coefficients[:, 1].reshape(L, K)
        return ak, bk

    def Exploration_process(self, K: int = 24, explor_factor: int = 2,
                            gamma_n_k: float = 7.35):
        """
        Generate exploration trajectories (random constant actions per step).

        Returns
        -------
        P, x : each (T_explor, L, K)
        """
        if explor_factor < 2:
            raise ValueError("Exploration factor must be >= 2")
        domain_sample = np.arange(0.2, gamma_n_k, 0.1)
        if len(domain_sample) < 2 * K:
            raise ValueError(
                "More unknowns than domain samples -- check delta (default 0.1)"
            )
        T_explor = K * explor_factor
        x = np.random.choice(domain_sample, size=T_explor, replace=False)
        x = np.repeat(x[:, np.newaxis, np.newaxis], self.L, axis=1)
        x = np.repeat(x, K, axis=2)
        Sk = self.N * x
        P = self.A_k * x * (Sk ** 2) + self.B_k * Sk * x
        return P, x

    def Get_sk(self, Ak: np.ndarray, Bk: np.ndarray,
               Xnk: np.ndarray, Pnk: np.ndarray) -> np.ndarray:
        """Recover S_k from the quadratic payment equation."""
        coeffs_a = Ak * Xnk
        coeffs_b = Bk * Xnk
        coeffs_c = -Pnk
        delta = np.sqrt(coeffs_b ** 2 - 4 * coeffs_a * coeffs_c)
        S1 = (-coeffs_b + delta) / (2 * coeffs_a + 1e-6)
        S2 = (-coeffs_b - delta) / (2 * coeffs_a + 1e-6)
        return np.where(S1 > 0, S1, S2)


# ---------------------------------------------------------------------------
# NN-based Nash filter
# ---------------------------------------------------------------------------
class Nash_Filter_NN(Nash_Filter):
    """Extends ``Nash_Filter`` with a trained neural network for a_k, b_k."""

    def __init__(self, L: int, N: int, K: int,
                 A_k: np.ndarray, B_k: np.ndarray, const_V: float):
        super().__init__(L, N, K, A_k, B_k, const_V)
        self.model: Energy_naive | None = None

    # -- model I/O ----------------------------------------------------------
    def Load_NN_Model(self, path_weights: str,
                      weights_dir: str = "Weights",
                      input_size: int = 23, output_size: int = 2):
        file_path = os.path.join(weights_dir, path_weights)
        self.model = Energy_naive(input_size, output_size)
        self.model.load_state_dict(
            torch.load(file_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.model(X_t).numpy()

    # -- NN exploration & estimation ----------------------------------------
    def Exploration_process(self, T_exploration: int = 5,     # type: ignore[override]
                            pathN: str = "N=5_energey_game",
                            hyper_dir: str = "hyperparameters"):
        """
        Build the feature tensor for NN inference from exploration data.

        Returns
        -------
        X_train : (L, N, K, input_size)
        """
        low_bound = ((3 * T_exploration) / self.N) + 0.1
        Xn_k = np.random.uniform(low=low_bound, high=7.0,
                                 size=(self.L, self.N, self.K))
        X_train = []
        for t in range(-T_exploration, T_exploration + 1):
            Sk = np.sum(Xn_k, axis=1)
            Sk = np.repeat(Sk[:, np.newaxis, :], self.N, axis=1)
            P = self.A_k[:, np.newaxis, :] * Xn_k * (Sk ** 2) \
                + self.B_k[:, np.newaxis, :] * Xn_k * Sk
            X_train.append(P)
            X_train.append(Xn_k)
            Xn_k = Xn_k + (1 / self.N) * t

        X_train = np.stack(X_train, axis=-1)
        X_train = np.append(
            X_train, self.N * np.ones((self.L, self.N, self.K, 1)), axis=-1
        )

        # Z-score normalisation
        X_mean = np.load(os.path.join(hyper_dir, pathN, "X_mean.npy"))
        X_std = np.load(os.path.join(hyper_dir, pathN, "X_std.npy"))
        epsilon = 0.01
        X_train = (X_train - X_mean) / (X_std + epsilon)
        return X_train

    def Estimate_ak_bk(self, X_train: np.ndarray):       # type: ignore[override]
        """
        Predict a_k, b_k via the neural network.

        Parameters
        ----------
        X_train : (L, N, K, input_size)

        Returns
        -------
        ak, bk : each (L, N, K)
        """
        L, N, K, input_size = X_train.shape
        X_flat = X_train.reshape(-1, input_size)
        predictions = self.forward_pass(X_flat)
        ak = predictions[:, 0].reshape(L, N, K)
        bk = predictions[:, 1].reshape(L, N, K)
        return ak, bk
