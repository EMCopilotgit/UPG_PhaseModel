import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "iteration2/results/phase4"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_T_star_heatmap(cfg, T_star):
    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        T_star.T,
        origin="lower",
        aspect="auto",
        extent=[cfg.etas[0], cfg.etas[-1], cfg.lambdas[0], cfg.lambdas[-1]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label=r"$T^*$")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"$T^*(\eta,\lambda)$ heatmap")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "Tstar_heatmap.png"), dpi=300)
    plt.close(fig)


def plot_binary_phase_diagram(cfg, eq_flag, lambda_star_curve):
    etas_eff, lambda_star = lambda_star_curve

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.imshow(
        eq_flag.T,
        origin="lower",
        aspect="auto",
        extent=[cfg.etas[0], cfg.etas[-1], cfg.lambdas[0], cfg.lambdas[-1]],
        cmap="Greys",
    )

    if len(etas_eff) > 0:
        ax.plot(etas_eff, lambda_star, color="red", lw=2)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("Binary phase diagram")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "binary_phase_diagram.png"), dpi=300)
    plt.close(fig)


def plot_T_star_surface(cfg, T_star):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    eta_grid, lam_grid = np.meshgrid(cfg.etas, cfg.lambdas, indexing="ij")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(eta_grid, lam_grid, T_star, cmap="viridis", edgecolor="none")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_zlabel(r"$T^*$")
    ax.set_title("3D surface of $T^*(\eta,\lambda)$")

    fig.colorbar(surf, ax=ax, shrink=0.6)
    fig.tight_layout()

    fig.savefig(os.path.join(RESULTS_DIR, "Tstar_surface.png"), dpi=300)
    plt.close(fig)
