"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np

from .wireless_common import *
from .wireless_data_structure import *
from .wireless_plots import *
import torch
from typing import  Optional

def main_loop(cfg: SimConfig, rec: SimRecord, G: Sim_G, P: np.ndarray, lr: np.ndarray,
              grad_mode: GradMode, alpha_k: Optional[np.ndarray] = None, beta_k: Optional[np.ndarray] = None) -> None:
    """
    Main simulation entry point.

    Parameters
    ----------
    P : (L, N, K) Initialize power transmission
    lr: (T, ) learning rate factor
    grad_mode: Enum gradient indicator (0 - naive nash approach,
                                            1 - Optimal solution
                                            2 - Prior Approximation from DNN)

    G: Sim_G
        Include constant channel gain matrices
        - g: array in size (L, N, N, K) or (L, K, N, N)
        - g_diag: diagonal of g
        - g_zero: zero diagonal of g


    rec: SimRecord
        Simulation record Parameters
        - P: array storing powers over time
        - obj: objective values over time
        - grad_norm: gradient norms over time

    cfg : SimConfig
        Simulation configuration containing:
        - L : number of trials
        - N : number of players
        - K : number of frequency channels
        - Rlink : receiver link radius
        - T : number of iterations
        - N0 : White Gaussian Noise parameter
        - dist : channel distribution
        - alpha : path loss constant
        - seed : RNG seed (optional)
    alpha_k: alpha parameter from NN size (L, N, K)
    beta_k: beta parameter from NN size (L, N, K)
    """

    # Define Init condition in gradients
    gradients_residual = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)
    gradients_ml = np.zeros((cfg.L, cfg.N, cfg.K), dtype=float)

    for t in range(cfg.T):
        In = compute_interference(G.g, P, G.g_diag)
        gradients_local = calc_local_gradient(G.g_diag, In, cfg.N0, P)
        if grad_mode == GradMode.NAIVE_NASH:
            pass
        elif grad_mode == GradMode.OPTIMAL:
            gradients_residual = calc_residual_gradient(G.g_diag, G.g_zero, In, cfg.N0, P, gradients_local)
        elif grad_mode == GradMode.PRIOR_APPROXIMATION:
            gradients_ml = alpha_k * P + beta_k * In
        else:
            raise ValueError(f"Grad Mode {grad_mode} not recognized.")

        P = P + lr[t] * (gradients_local + gradients_residual + gradients_ml)
        # project_box(P, cfg.Border_floor, cfg.Border_ceil)
        P = project_onto_simplex(P, cfg.Border_ceil)

        rec.P[t] = P
        rec.obj[t] = compute_objective(In, G.g_diag, P, cfg.N0)

        if cfg.SaveToTrain or cfg.isDebug:
            rec.grad_norm_prior[t] = gradients_residual
            rec.In[t] = In


def build_inputs_for_nn_from_rec(
    rec: SimRecord,
    cfg: SimConfig,
) -> torch.Tensor:
    """
    Build NN input for all games and all players.

    Expects:
      rec.P:  (T, L, N, K)
      rec.In: (T, L, N, K)

    Returns:
      inputs_pre_torch: torch.FloatTensor on CPU, shape (L*N, T*2K)
      meta: np.ndarray on CPU, shape (L*N, 2) with columns [l_idx, n_idx]
    """
    P = rec.P
    In = rec.In

    if P is None or In is None:
        raise ValueError("rec.P and rec.In must be populated (did you set cfg.SaveToTrain=True?)")

    if P.shape != In.shape:
        raise ValueError(f"Shape mismatch: rec.P {P.shape} vs rec.In {In.shape}")

    T, L, N, K = P.shape
    if T != cfg.T or L != cfg.L or N != cfg.N or K != cfg.K:
        # Not strictly required, but helps catch bugs
        pass

    # (T, L, N, 2K) where last dim is [P_k..., In_k...]
    X_tln = np.concatenate([P, In], axis=-1)

    # Reorder so each sample is (l, n) with a time sequence: (L, N, T, 2K)
    X_lnt = np.transpose(X_tln, (1, 2, 0, 3))

    # Flatten time to match FC input: (L*N, T*2K)
    X = X_lnt.reshape(L * N, T * 2 * K)

    # Return torch on CPU
    inputs_pre = torch.from_numpy(X).to(dtype=torch.float32, device="cpu")
    return inputs_pre


def Exploration_Process(
    cfg: SimConfig,
    rec: SimRecord,
    G: Sim_G,
    P: np.ndarray,
    lr: np.ndarray,
) -> torch.Tensor:
    """
    Runs naive exploration and returns NN inputs_pre on CPU as torch tensor.
    """
    # Must record rec.P and rec.In during simulation
    cfg.SaveToTrain = True

    main_loop(cfg, rec, G, P, lr, GradMode.NAIVE_NASH)

    inputs_pre = build_inputs_for_nn_from_rec(rec, cfg)
    # If you don't need meta, just return inputs_pre
    return inputs_pre



def run_simulation(
    cfg: SimConfig,
    g_struct: Sim_G,
    P_init: np.ndarray,
    lr: np.ndarray,
    grad_mode: GradMode,
    *,
    alpha_k=None,
    beta_k=None,
) -> SimRecord:
    """Run a single simulation mode and return its record."""
    rec = SimRecord.create(cfg)
    P = P_init.copy()

    main_loop(
        cfg=cfg,
        rec=rec,
        G=g_struct,
        P=P,
        lr=lr,
        grad_mode=grad_mode,
        alpha_k=alpha_k,
        beta_k=beta_k,
    )
    return rec
