#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os


def load_phase3_analysis(path="phase3_analysis_output.npz"):
    return np.load(path, allow_pickle=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_defect_count(data, outdir="results/phase3"):
    defect_count = data["defect_count"]
    if defect_count is None:
        return

    ensure_dir(outdir)
    t = np.arange(len(defect_count))

    plt.figure(figsize=(6, 4))
    plt.plot(t, defect_count, "-k")
    plt.xlabel("Time step")
    plt.ylabel("Defect count")
    plt.title("Phase III: Defect count vs time")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase3_defect_count.png"), dpi=200)
    plt.close()


def plot_defect_spacetime(data, outdir="results/phase3"):
    defect_positions = data["defect_positions"]
    if defect_positions is None:
        return

    ensure_dir(outdir)
    T = len(defect_positions)
    xs = []
    ts = []
    for t, pos in enumerate(defect_positions):
        xs.extend(pos)
        ts.extend([t] * len(pos))

    if len(xs) == 0:
        return

    # Infer system size from max position if needed
    L = max(xs) + 1

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ts, s=2, c="k")
    plt.xlabel("Position")
    plt.ylabel("Time step")
    plt.title("Phase III: Defect spacetime raster")
    plt.xlim(0, L)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase3_defect_spacetime.png"), dpi=200)
    plt.close()


def plot_defect_charge_hist(data, outdir="results/phase3"):
    all_charges = data["all_charges"]
    if all_charges is None or len(all_charges) == 0:
        return

    ensure_dir(outdir)

    plt.figure(figsize=(5, 4))
    bins = np.arange(all_charges.min() - 0.5, all_charges.max() + 1.5, 1)
    plt.hist(all_charges, bins=bins, edgecolor="k")
    plt.xlabel("Defect charge")
    plt.ylabel("Count")
    plt.title("Phase III: Defect charge distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase3_defect_charges.png"), dpi=200)
    plt.close()


def plot_final_fields_with_defects(data, outdir="results/phase3"):
    R = data["R"]
    T = data["T"]
    F = data["F"]
    N = data["N"]
    defect_positions = data["defect_positions"]

    if defect_positions is None:
        return

    ensure_dir(outdir)

    # Use last time step's defects
    last_defects = np.array(defect_positions[-1], dtype=int)
    x_idx = np.arange(len(R))

    plt.figure(figsize=(8, 6))

    plt.subplot(4, 1, 1)
    plt.plot(x_idx, R, "-b", label="R")
    if len(last_defects) > 0:
        plt.scatter(last_defects, R[last_defects], c="r", s=10, label="defects")
    plt.ylabel("R")
    plt.legend(loc="upper right")

    plt.subplot(4, 1, 2)
    plt.plot(x_idx, T, "-g", label="T")
    if len(last_defects) > 0:
        plt.scatter(last_defects, T[last_defects], c="r", s=10)
    plt.ylabel("T")

    plt.subplot(4, 1, 3)
    plt.plot(x_idx, F, "-m", label="F")
    if len(last_defects) > 0:
        plt.scatter(last_defects, F[last_defects], c="r", s=10)
    plt.ylabel("F")

    plt.subplot(4, 1, 4)
    plt.plot(x_idx, N, "-k", label="N")
    if len(last_defects) > 0:
        plt.scatter(last_defects, N[last_defects], c="r", s=10)
    plt.ylabel("N")
    plt.xlabel("Index")

    plt.suptitle("Phase III: Final fields with defect markers")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(outdir, "phase3_fields_with_defects.png"), dpi=200)
    plt.close()


def main():
    data = load_phase3_analysis("phase3_analysis_output.npz")

    plot_defect_count(data)
    plot_defect_spacetime(data)
    plot_defect_charge_hist(data)
    plot_final_fields_with_defects(data)


if __name__ == "__main__":
    main()
