import os
import numpy as np
import matplotlib.pyplot as plt

GEOM_PATH = "results/phase6/phase6_geometry.npz"
OUTDIR = "results/phase6/plots/"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
#  LOAD GEOMETRY
# ============================================================

def load_phase6_geometry(path=GEOM_PATH):
    data = np.load(path, allow_pickle=True)
    geom = {
        "lags": data["lags"],
        "msd": data["msd"],
        "counts": data["counts"],
        "step_lengths": data["step_lengths"],
    }

    # Load displacement distributions
    disp = {}
    for key in data.files:
        if key.startswith("disp_lag_"):
            tau = int(key.split("_")[-1])
            disp[tau] = data[key]
    geom["disp"] = disp

    return geom


# ============================================================
#  PLOTS
# ============================================================

def plot_msd(lags, msd):
    plt.figure(figsize=(8,5))
    plt.plot(lags, msd, lw=2, color="darkblue")
    plt.xlabel("Lag τ")
    plt.ylabel("MSD(τ)")
    plt.title("Phase VI: Mean Squared Displacement")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "msd.png"))
    plt.close()


def plot_msd_loglog(lags, msd):
    plt.figure(figsize=(8,5))
    plt.loglog(lags, msd, lw=2, color="darkgreen")
    plt.xlabel("Lag τ (log)")
    plt.ylabel("MSD(τ) (log)")
    plt.title("Phase VI: MSD (log–log)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "msd_loglog.png"))
    plt.close()


def plot_step_length_hist(step_lengths):
    valid = step_lengths[np.isfinite(step_lengths)]
    if len(valid) == 0:
        print("No valid step lengths to plot.")
        return

    plt.figure(figsize=(8,5))
    plt.hist(valid, bins=200, color="purple", alpha=0.8)
    plt.xlabel("|Δx|")
    plt.ylabel("Count")
    plt.title("Phase VI: Step Length Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "step_length_hist.png"))
    plt.close()


def plot_displacement_distribution(disp_dict):
    for tau, arr in disp_dict.items():
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"No valid displacements for lag {tau}.")
            continue

        plt.figure(figsize=(8,5))
        plt.hist(valid, bins=200, color="orange", alpha=0.8)
        plt.xlabel(f"x(t+{tau}) - x(t)")
        plt.ylabel("Count")
        plt.title(f"Phase VI: Displacement Distribution (lag={tau})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"disp_lag_{tau}.png"))
        plt.close()


# ============================================================
#  MAIN DRIVER
# ============================================================

def run_phase6_plots():
    print("Loading Phase VI geometry...")
    geom = load_phase6_geometry()

    lags = geom["lags"]
    msd = geom["msd"]
    step_lengths = geom["step_lengths"]
    disp = geom["disp"]

    print("Plotting MSD...")
    plot_msd(lags, msd)
    plot_msd_loglog(lags, msd)

    print("Plotting step length histogram...")
    plot_step_length_hist(step_lengths)

    print("Plotting displacement distributions...")
    plot_displacement_distribution(disp)

    print("Phase VI plotting complete.")


if __name__ == "__main__":
    run_phase6_plots()
