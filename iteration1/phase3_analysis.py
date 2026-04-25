#!/usr/bin/env python3
import numpy as np


def load_phase3_output(path="phase3_output.npz"):
    data = np.load(path, allow_pickle=True)
    return data


def compute_defect_count(defect_positions):
    # defect_positions: object array, length T, each entry: array of positions
    counts = np.array([len(pos) for pos in defect_positions])
    return counts


def compute_field_means(R, T, F, N):
    R_mean = np.mean(R)
    T_mean = np.mean(T)
    F_mean = np.mean(F)
    N_mean = np.mean(N)
    return R_mean, T_mean, F_mean, N_mean


def compute_field_means_time_series(R_log=None, T_log=None, F_log=None, N_log=None):
    """
    Optional: if you later log full time series of fields, you can extend this.
    For now, we just return None placeholders.
    """
    return None, None, None, None


def flatten_defect_charges(defect_charges):
    if defect_charges is None:
        return None
    if len(defect_charges) == 0:
        return np.array([])
    return np.concatenate(defect_charges)


def main():
    data = load_phase3_output("phase3_output.npz")

    R = data["R"]
    T = data["T"]
    F = data["F"]
    N = data["N"]
    x = data["x"]

    defect_positions = data.get("defect_positions", None)
    defect_charges = data.get("defect_charges", None)
    defect_types = data.get("defect_types", None)

    # Compute defect count vs time
    if defect_positions is not None:
        defect_count = compute_defect_count(defect_positions)
    else:
        defect_count = None

    # Flatten charges for histogram
    if defect_charges is not None:
        all_charges = flatten_defect_charges(defect_charges)
    else:
        all_charges = None

    # Simple field means (final-time fields)
    R_mean, T_mean, F_mean, N_mean = compute_field_means(R, T, F, N)

    analysis = {
        "R": R,
        "T": T,
        "F": F,
        "N": N,
        "x": x,
        "defect_positions": defect_positions,
        "defect_charges": defect_charges,
        "defect_types": defect_types,
        "defect_count": defect_count,
        "all_charges": all_charges,
        "R_mean": R_mean,
        "T_mean": T_mean,
        "F_mean": F_mean,
        "N_mean": N_mean,
    }

    # Save as a single npz for plotting
    np.savez(
        "phase3_analysis_output.npz",
        R=analysis["R"],
        T=analysis["T"],
        F=analysis["F"],
        N=analysis["N"],
        x=analysis["x"],
        defect_positions=analysis["defect_positions"],
        defect_charges=analysis["defect_charges"],
        defect_types=analysis["defect_types"],
        defect_count=analysis["defect_count"],
        all_charges=analysis["all_charges"],
        R_mean=analysis["R_mean"],
        T_mean=analysis["T_mean"],
        F_mean=analysis["F_mean"],
        N_mean=analysis["N_mean"],
    )


if __name__ == "__main__":
    main()
