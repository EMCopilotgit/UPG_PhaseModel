import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from iteration2.phase5.phase5_dynamics import (
    RESULTS_DIR_PHASE5,
    Phase5Config,
    run_single_trajectory,
    classify_regime,
)


def plot_three_regimes_at_eta(cfg: Phase5Config,
                              eta: float,
                              lam_base: float,
                              lam_gap: float,
                              lam_island: float,
                              outname: str = "w_mean_three_regimes.png"):
    """
    Plot w_mean vs episode for three lambdas at fixed eta:
    - base equilibrium band
    - non-equilibrium gap
    - re-entrant equilibrium island
    """
    traj_base = run_single_trajectory(cfg, eta, lam_base)
    traj_gap = run_single_trajectory(cfg, eta, lam_gap)
    traj_island = run_single_trajectory(cfg, eta, lam_island)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(traj_base["episodes"], traj_base["w_mean"],
            label=fr"$\lambda={lam_base:.3f}$ (base band)")
    ax.plot(traj_gap["episodes"], traj_gap["w_mean"],
            label=fr"$\lambda={lam_gap:.3f}$ (gap)")
    ax.plot(traj_island["episodes"], traj_island["w_mean"],
            label=fr"$\lambda={lam_island:.3f}$ (floating island)")

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\langle w \rangle$")
    ax.set_title(fr"$\langle w \rangle$ vs episode at $\eta={eta:.4f}$")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE5, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_single_trajectory(traj: Dict[str, np.ndarray],
                           eta: float,
                           lam: float,
                           outname: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(traj["episodes"], traj["w_mean"], label=r"$\langle w \rangle$")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\langle w \rangle$")
    ax.set_title(fr"$\langle w \rangle$ vs episode at $\eta={eta:.4f}, \lambda={lam:.4f}$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE5, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_three_regimes_with_labels(cfg: Phase5Config,
                                   eta: float,
                                   lam_base: float,
                                   lam_gap: float,
                                   lam_island: float,
                                   outname: str = "w_mean_three_regimes_labeled.png"):
    traj_base = run_single_trajectory(cfg, eta, lam_base)
    traj_gap = run_single_trajectory(cfg, eta, lam_gap)
    traj_island = run_single_trajectory(cfg, eta, lam_island)

    reg_base = classify_regime(traj_base["w_mean"])
    reg_gap = classify_regime(traj_gap["w_mean"])
    reg_island = classify_regime(traj_island["w_mean"])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(traj_base["episodes"], traj_base["w_mean"],
            label=fr"$\lambda={lam_base:.3f}$ ({reg_base})")
    ax.plot(traj_gap["episodes"], traj_gap["w_mean"],
            label=fr"$\lambda={lam_gap:.3f}$ ({reg_gap})")
    ax.plot(traj_island["episodes"], traj_island["w_mean"],
            label=fr"$\lambda={lam_island:.3f}$ ({reg_island})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\langle w \rangle$")
    ax.set_title(fr"Dynamical regimes at $\eta={eta:.4f}$")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE5, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
