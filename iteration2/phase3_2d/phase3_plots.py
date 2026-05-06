import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from iteration2.phase3_2d.phase3_analysis import (
    RESULTS_DIR_PHASE3,
    extract_Q_history,
    extract_defect_count_history,
)


def plot_q_field(q_field: np.ndarray, outname: str = "q_field.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(q_field, origin="lower", cmap="bwr")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$q(x,y)$")
    ax.set_title("Topological charge field")
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE3, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_Q_history(logs: Dict[str, any], outname: str = "Q_history.png"):
    Q_hist = extract_Q_history(logs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(len(Q_hist)), Q_hist, marker="o", ms=3)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$Q_{\mathrm{total}}$")
    ax.set_title(r"Topological charge history $Q_{\mathrm{total}}(t)$")
    ax.grid(True)
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE3, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_defect_count_history(
    logs: Dict[str, any],
    outname: str = "defect_count_history.png",
):
    dc_hist = extract_defect_count_history(logs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(len(dc_hist)), dc_hist, marker="o", ms=3)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Defect count")
    ax.set_title("Defect count history")
    ax.grid(True)
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE3, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_Q_vs_defects(
    Q_values: np.ndarray,
    defect_counts: np.ndarray,
    outname: str = "Q_vs_defects_scatter.png",
):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(Q_values, defect_counts, alpha=0.7)
    ax.set_xlabel(r"$Q_{\mathrm{total}}$")
    ax.set_ylabel("Defect count")
    ax.set_title(r"$Q_{\mathrm{total}}$ vs defect count")
    ax.grid(True)
    fig.tight_layout()

    outpath = os.path.join(RESULTS_DIR_PHASE3, outname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
