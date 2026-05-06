import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from iteration2.phase5.phase5_dynamics import (
    RESULTS_DIR_PHASE4,
    RESULTS_DIR_PHASE5,
)


def load_phase4_arrays():
    T_star = np.load(os.path.join(RESULTS_DIR_PHASE4, "T_star.npy"))
    eq_flag = np.load(os.path.join(RESULTS_DIR_PHASE4, "eq_flag.npy"))
    lambda_star = np.load(os.path.join(RESULTS_DIR_PHASE4, "lambda_star.npy"))
    etas_eff = np.load(os.path.join(RESULTS_DIR_PHASE4, "etas_eff.npy"))

    etas = np.linspace(0.001, 0.010, T_star.shape[0])
    lambdas = np.linspace(0.02, 0.20, T_star.shape[1])

    return etas, lambdas, T_star, eq_flag, etas_eff, lambda_star


def plot_annotated_binary():
    etas, lambdas, T_star, eq_flag, etas_eff, lambda_star = load_phase4_arrays()

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        eq_flag.T,
        origin="lower",
        aspect="auto",
        extent=[etas[0], etas[-1], lambdas[0], lambdas[-1]],
        cmap="Greys",
    )

    if len(etas_eff) > 0:
        ax.plot(etas_eff, lambda_star, color="red", lw=2, label=r"$\lambda^*(\eta)$")

    # Example annotations (positions tuned by eye)
    ax.text(0.0015, 0.025, "base\nband", color="black", fontsize=9)
    ax.text(0.0060, 0.075, "floating\nisland", color="black", fontsize=9)
    ax.text(0.0032, 0.060, "stability\nfinger", color="black", fontsize=9)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("Binary phase diagram with annotations")
    ax.legend()

    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR_PHASE5, "binary_phase_diagram_annotated.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_annotated_T_star():
    etas, lambdas, T_star, eq_flag, etas_eff, lambda_star = load_phase4_arrays()

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        T_star.T,
        origin="lower",
        aspect="auto",
        extent=[etas[0], etas[-1], lambdas[0], lambdas[-1]],
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$T^*(\eta,\lambda)$ (episodes)")

    if len(etas_eff) > 0:
        ax.plot(etas_eff, lambda_star, color="red", lw=2, label=r"$\lambda^*(\eta)$")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"$T^*(\eta,\lambda)$ with phase boundary")

    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR_PHASE5, "Tstar_heatmap_annotated.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
