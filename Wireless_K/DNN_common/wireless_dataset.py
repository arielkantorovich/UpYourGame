"""
Created on : ------

@author: Ariel_Kantorovich
"""

import numpy as np
from typing import Optional, Tuple

def build_player_subset_dataset(
    P: np.ndarray,
    In: np.ndarray,
    grad: np.ndarray,
    N_subset: int,
    player_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param player_idx:
    :param P: (T, L, N, K)
    :param In: (T, L, N, K)
    :param grad: (T, L, N, K)
    :param N_subset: scalar e.g 3 so N_sub = [0, 1, 2]
    :param player_idx: which indices take the subset optional parameter
    :return:
    """

    if P.shape != In.shape or P.shape != grad.shape:
        raise ValueError(f"Shape mismatch: P{P.shape}, In{In.shape}, grad{grad.shape}")

    T, L, N, K = P.shape

    if player_idx is None:
        if N_subset > N:
            raise ValueError(f"N_subset={N_subset} cannot be > N={N}")
        player_idx = np.arange(N_subset, dtype=int)
    else:
        player_idx = np.asarray(player_idx, dtype=int)
        if player_idx.ndim != 1:
            raise ValueError("player_idx must be 1D")
        if len(player_idx) != N_subset:
            raise ValueError(f"player_idx length {len(player_idx)} != N_subset {N_subset}")
        if np.any(player_idx < 0) or np.any(player_idx >= N):
            raise ValueError("player_idx contains out-of-range indices")

    P_sub = P[:, :, player_idx, :]      # (T, L, N_sub, K)
    In_sub = In[:, :, player_idx, :]    # (T, L, N_sub, K)
    grad_sub = grad[:, :, player_idx, :]# (T, L, N_sub, K)

    X_tln = np.concatenate([P_sub, In_sub], axis=-1)  # (T, L, N_sub, 2K)
    X_lnt = np.transpose(X_tln, (1, 2, 0, 3))         # (L, N_sub, T, 2K)
    X = X_lnt.reshape(L * N_subset, T * 2 * K)         # (L*N_sub, T*2K) for FC Input


    Y_lnkT = np.transpose(grad_sub, (1, 2, 3, 0))     # (L, N_sub, K, T)
    Y = Y_lnkT.reshape(L * N_subset, K, T)            # (L*N_sub, K, T)

    l_idx = np.repeat(np.arange(L), N_subset)
    n_idx = np.tile(player_idx, L)
    meta = np.stack([l_idx, n_idx], axis=1)

    return X, Y, meta



if __name__ == '__main__':
    pass
    # X, Y, meta = build_player_subset_dataset()
