"""
Created on : ------

@author: Ariel_Kantorovich
"""

from utils.quad_utils import *
from utils.data_structure import *

def main(cfg: SimConfig) -> None:
    Q, B = generate_Q_B(
        N=cfg.N,
        L=cfg.L,
        alpha=cfg.alpha,
        beta=cfg.beta,
        low=cfg.qnn_low,
        high=cfg.qnn_high,
    )

    x_init = np.random.uniform(0.1, 1.1, size=(cfg.L, cfg.N, 1))
    ne_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    optimal_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    prior_record = SimRecord.zeros(cfg.L, cfg.T, cfg.N)
    diagonals_est = np.expand_dims(np.diagonal(Q, axis1=1, axis2=2), axis=-1)

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B,
        x=x_init,
        sim_record=ne_record,
        grad_mode=GradMode.NAIVE_NASH,
    )

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B,
        x=x_init,
        sim_record=optimal_record,
        grad_mode=GradMode.OPTIMAL,
    )

    IterationLoop(
        cfg=cfg,
        Q=Q,
        B=B,
        x=x_init,
        sim_record=prior_record,
        grad_mode=GradMode.PRIOR_APPROXIMATION,
        diagonals_est=diagonals_est,
    )

    print("Quadratic simulation finished.")








if '__main__' == __name__:
    cfg = parse_args()
    print("================================ Running Quadratic Simulation ===============================")
    print(cfg)
    main(cfg)
