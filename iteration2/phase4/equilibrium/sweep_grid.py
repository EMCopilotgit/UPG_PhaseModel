import numpy as np
from dataclasses import dataclass
from iteration2.phase4.engine.run_phase4_geom import run_phase4_geom
from .equilibrium_test import compute_T_star


@dataclass
class SweepConfig:
    etas: np.ndarray
    lambdas: np.ndarray
    max_episodes: int
    n_seeds: int
    equilibrium_window: int
    equilibrium_tol: float


def sweep_grid(cfg: SweepConfig):
    n_eta = len(cfg.etas)
    n_lam = len(cfg.lambdas)

    T_star = np.full((n_eta, n_lam), np.nan)
    eq_flag = np.zeros((n_eta, n_lam), dtype=int)

    for i, eta in enumerate(cfg.etas):
        for j, lam in enumerate(cfg.lambdas):
            T_values = []

            for seed in range(cfg.n_seeds):
                logs, _, _ = run_phase4_geom(
                    episodes=cfg.max_episodes,
                    eta=eta,
                    lam=lam,
                )

                observable = np.array([ep["w_mean"] for ep in logs["episodes"]])
                T_seed = compute_T_star(observable, cfg.equilibrium_window, cfg.equilibrium_tol)

                if T_seed is not None:
                    T_values.append(T_seed)

            if T_values:
                eq_flag[i, j] = 1
                T_star[i, j] = np.mean(T_values)

            print(f"(eta={eta:.4f}, lam={lam:.4f}) → eq={bool(eq_flag[i,j])}, T*={T_star[i,j]}")

    return T_star, eq_flag
