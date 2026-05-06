import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from iteration2.phase4.run_phase4_geom import run_phase4_geom

RESULTS_DIR_PHASE4 = "iteration2/results/phase4"
RESULTS_DIR_PHASE5 = "iteration2/results/phase5"
os.makedirs(RESULTS_DIR_PHASE5, exist_ok=True)


@dataclass
class Phase5Config:
    etas: np.ndarray
    lambdas: np.ndarray
    max_episodes: int = 400
    T_steps: int = 200
    Nx: int = 64
    Ny: int = 64
    dt: float = 0.01
    kappa0: float = 0.2
    alpha: float = 0.1


def load_phase4_results() -> Dict[str, Any]:
    T_star = np.load(os.path.join(RESULTS_DIR_PHASE4, "T_star.npy"))
    eq_flag = np.load(os.path.join(RESULTS_DIR_PHASE4, "eq_flag.npy"))
    lambda_star = np.load(os.path.join(RESULTS_DIR_PHASE4, "lambda_star.npy"))
    etas_eff = np.load(os.path.join(RESULTS_DIR_PHASE4, "etas_eff.npy"))

    # Reconstruct eta / lambda grids from shapes
    # (Assumes same grids as default_config in Phase 4)
    etas = np.linspace(0.001, 0.010, T_star.shape[0])
    lambdas = np.linspace(0.02, 0.20, T_star.shape[1])

    return {
        "T_star": T_star,
        "eq_flag": eq_flag,
        "lambda_star": lambda_star,
        "etas_eff": etas_eff,
        "etas": etas,
        "lambdas": lambdas,
    }


def run_single_trajectory(cfg: Phase5Config, eta: float, lam: float) -> Dict[str, np.ndarray]:
    logs, final_state, w_history = run_phase4_geom(
        episodes=cfg.max_episodes,
        T_steps=cfg.T_steps,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        dt=cfg.dt,
        kappa0=cfg.kappa0,
        alpha=cfg.alpha,
        eta=eta,
        lam=lam,
    )
    w_mean = np.array([ep["w_mean"] for ep in logs["episodes"]])
    w_std = np.array([ep["w_std"] for ep in logs["episodes"]])
    episodes = np.arange(len(w_mean))

    return {
        "episodes": episodes,
        "w_mean": w_mean,
        "w_std": w_std,
    }


def classify_regime(w_mean: np.ndarray,
                    window: int = 40,
                    rel_tol: float = 0.02) -> str:
    """
    Very simple dynamical classifier:
    - 'equilibrium' if tail is stationary (small relative std)
    - 'drift' if mean keeps moving
    - 'oscillatory' if large variance around a mean
    """
    if len(w_mean) < window:
        return "insufficient_data"

    tail = w_mean[-window:]
    mean = np.mean(tail)
    std = np.std(tail)

    if np.abs(mean) < 1e-12:
        rel = std
    else:
        rel = std / np.abs(mean)

    if rel < rel_tol:
        return "equilibrium"

    # crude heuristic: look at overall trend
    slope = (w_mean[-1] - w_mean[0]) / len(w_mean)
    if np.abs(slope) > 0.001:
        return "drift"

    return "oscillatory"


def pick_representative_lambdas(etas: np.ndarray,
                                lambdas: np.ndarray,
                                target_eta: float,
                                lam_base: float,
                                lam_gap: float,
                                lam_island: float) -> Tuple[float, float, float, float]:
    """
    Helper to snap requested eta, lambda values to nearest grid points.
    Returns:
        eta_grid, lam_base_grid, lam_gap_grid, lam_island_grid
    """
    eta_idx = np.argmin(np.abs(etas - target_eta))
    eta_grid = etas[eta_idx]

    lam_base_idx = np.argmin(np.abs(lambdas - lam_base))
    lam_gap_idx = np.argmin(np.abs(lambdas - lam_gap))
    lam_island_idx = np.argmin(np.abs(lambdas - lam_island))

    return (
        eta_grid,
        lambdas[lam_base_idx],
        lambdas[lam_gap_idx],
        lambdas[lam_island_idx],
    )
