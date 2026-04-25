import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "results/phase7/phase7_diagnostics.npz"
OUTDIR = "results/phase7/plots/"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
#  LOAD DIAGNOSTICS
# ============================================================

def load_phase7_diagnostics(path=DATA_PATH):
    data = np.load(path, allow_pickle=True)
    diag = {
        "velocities": data["velocities"],
        "vacf_lags": data["vacf_lags"],
        "vacf": data["vacf"],
        "vacf_counts": data["vacf_counts"],
        "occ_centers": data["occ_centers"],
        "occ_counts": data["occ_counts"],
        "occ_probs": data["occ_probs"],
    }
    return diag


# ============================================================
#  PLOTS
# ============================================================

def plot_velocity_hist(v):
    valid = v[np.isfinite(v)]
    if len(valid) == 0:
        print("No valid velocities to plot.")
        return

    plt.figure(figsize=(8,5))
    plt.hist(valid, bins=200, color="steelblue", alpha=0.85)
    plt.xlabel("Velocity v = x(t+1) - x(t)")
    plt.ylabel("Count")
    plt.title("Phase VII: Velocity Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "velocity_hist.png"))
    plt.close()


def plot_vacf(lags, vacf):
    valid = np.isfinite(vacf)
    if not np.any(valid):
        print("No valid VACF values to plot.")
        return

    plt.figure(figsize=(8,5))
    plt.plot(lags[valid], vacf[valid], lw=2, color="darkred")
    plt.xlabel("Lag τ")
    plt.ylabel("C_v(τ)")
    plt.title("Phase VII: Velocity Autocorrelation Function")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "vacf.png"))
    plt.close()


def plot_vacf_loglog(lags, vacf):
    valid = (lags > 0) & np.isfinite(vacf)
    if not np.any(valid):
        print("No valid VACF values for log–log plot.")
        return

    plt.figure(figsize=(8,5))
    plt.loglog(lags[valid], np.abs(vacf[valid]), lw=2, color="darkgreen")
    plt.xlabel("Lag τ (log)")
    plt.ylabel("|C_v(τ)| (log)")
    plt.title("Phase VII: VACF (log–log)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "vacf_loglog.png"))
    plt.close()


def plot_spatial_occupancy(centers, counts, probs):
    if len(centers) == 0:
        print("No occupancy data to plot.")
        return

    plt.figure(figsize=(10,5))
    plt.bar(centers, counts, width=1.0, color="purple", alpha=0.8)
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.title("Phase VII: Spatial Occupancy Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "occupancy_hist.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(centers, probs, lw=2, color="black")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title("Phase VII: Spatial Occupancy Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "occupancy_prob.png"))
    plt.close()


# ============================================================
#  MAIN DRIVER
# ============================================================

def run_phase7_plots():
    print("Loading Phase VII diagnostics...")
    diag = load_phase7_diagnostics()

    v = diag["velocities"]
    lags = diag["vacf_lags"]
    vacf = diag["vacf"]
    centers = diag["occ_centers"]
    counts = diag["occ_counts"]
    probs = diag["occ_probs"]

    print("Plotting velocity histogram...")
    plot_velocity_hist(v)

    print("Plotting VACF...")
    plot_vacf(lags, vacf)
    plot_vacf_loglog(lags, vacf)

    print("Plotting spatial occupancy...")
    plot_spatial_occupancy(centers, counts, probs)

    print("Phase VII plotting complete.")


if __name__ == "__main__":
    run_phase7_plots()

