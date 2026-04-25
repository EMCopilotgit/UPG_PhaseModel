import os
import json
import numpy as np
import matplotlib.pyplot as plt


class Phase4Plotter:
    def __init__(self, results_dir="results/phase4"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Figure 1: Stability decay curve
    # ---------------------------------------------------------
    def plot_stability_history(self, history):
        plt.figure(figsize=(8, 4))
        plt.plot(history, lw=2)
        plt.xlabel("Time step")
        plt.ylabel("Stability score")
        plt.title("Phase IV Stability Decay")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "stability_decay.png"))
        plt.close()

    # ---------------------------------------------------------
    # Figure 2: Drift norms
    # ---------------------------------------------------------
    def plot_drift(self, drift):
        labels = ["R", "T", "F", "N"]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, drift, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        plt.ylabel("Norm drift")
        plt.title("Phase IV Drift Norms")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "drift_norms.png"))
        plt.close()

    # ---------------------------------------------------------
    # Figure 3: Field snapshots (before vs after)
    # ---------------------------------------------------------
    def plot_field_snapshots(self, R0, T0, F0, N0, R, T, F, N):
        fields0 = [R0, T0, F0, N0]
        fields1 = [R, T, F, N]
        names = ["R", "T", "F", "N"]

        plt.figure(figsize=(12, 8))
        for i, name in enumerate(names):
            plt.subplot(2, 2, i + 1)
            plt.plot(fields0[i], label=f"{name} (initial)", alpha=0.7)
            plt.plot(fields1[i], label=f"{name} (final)", alpha=0.7)
            plt.title(f"{name} Field Evolution")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "field_snapshots.png"))
        plt.close()

    # ---------------------------------------------------------
    # Figure 4: Optional perturbation response
    # ---------------------------------------------------------
    def plot_perturbation_response(self, R_before, R_after, R_final):
        plt.figure(figsize=(8, 4))
        plt.plot(R_before, label="Before perturbation", alpha=0.7)
        plt.plot(R_after, label="After perturbation", alpha=0.7)
        plt.plot(R_final, label="Final (recovered)", alpha=0.7)
        plt.title("Perturbation Response (R field)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "perturbation_response.png"))
        plt.close()
    # ---------------------------------------------------------
    # Extra 1: Perturbation footprint (R_after - R_before)
    # ---------------------------------------------------------
    def plot_perturbation_footprint(self, R_before, R_after):
        diff = R_after - R_before

        plt.figure(figsize=(8, 4))
        plt.plot(diff, label="R_after - R_before", color="#c44e52")
        plt.axhline(0.0, color="k", lw=1, alpha=0.5)
        plt.title("Perturbation Footprint (R field)")
        plt.xlabel("Index")
        plt.ylabel("ΔR")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "perturbation_footprint_R.png"))
        plt.close()

    # ---------------------------------------------------------
    # Extra 2: Recovery trajectory after perturbation
    # ---------------------------------------------------------
    def plot_recovery_trajectory(self, R_traj, R_before):
        # R_traj: list of R fields from perturbation onward
        norms = [np.linalg.norm(R - R_before) for R in R_traj]

        plt.figure(figsize=(8, 4))
        plt.plot(norms, lw=2)
        plt.title("Recovery Trajectory (‖R(t) - R_before‖)")
        plt.xlabel("Steps since perturbation")
        plt.ylabel("L2 distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "recovery_trajectory_R.png"))
        plt.close()

    # ---------------------------------------------------------
    # Extra 3: Multi-field snapshot at perturbation
    # ---------------------------------------------------------
    def plot_perturbation_multifield(self, Rb, Tb, Fb, Nb, Ra, Ta, Fa, Na):
        fields_before = [Rb, Tb, Fb, Nb]
        fields_after = [Ra, Ta, Fa, Na]
        names = ["R", "T", "F", "N"]

        plt.figure(figsize=(12, 8))
        for i, name in enumerate(names):
            plt.subplot(2, 2, i + 1)
            plt.plot(fields_before[i], label=f"{name} before", alpha=0.7)
            plt.plot(fields_after[i], label=f"{name} after", alpha=0.7)
            plt.title(f"{name} Perturbation Snapshot")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "perturbation_multifield.png"))
        plt.close()
