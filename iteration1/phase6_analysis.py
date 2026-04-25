import os
import json
import numpy as np

CANONICAL_EVENTS = "phase5_events.npz"
OUTDIR = "results/phase6/"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
#  LOAD + RECONSTRUCT WORLDLINES / EVENTS (same catalogue format)
# ============================================================

def load_phase5_catalogue(path=CANONICAL_EVENTS):
    data = np.load(path, allow_pickle=True)

    catalogue = {
        "worldline_ids": data["worldline_ids"],
        "worldline_lengths": data["worldline_lengths"],
        "worldline_times": data["worldline_times"],
        "worldline_positions": data["worldline_positions"],
        "worldline_charges": data["worldline_charges"],
        "worldline_types": data["worldline_types"],
        "event_types": data["event_types"],
        "event_times": data["event_times"],
        "event_worldline_ids": data["event_worldline_ids"],
    }

    return catalogue


def reconstruct_worldlines(cat):
    worldlines = []
    for i in range(len(cat["worldline_ids"])):
        wl = {
            "id": int(cat["worldline_ids"][i]),
            "times": np.array(cat["worldline_times"][i]),
            "positions": np.array(cat["worldline_positions"][i]),
            "charges": np.array(cat["worldline_charges"][i]),
            "types": np.array(cat["worldline_types"][i]),
        }
        worldlines.append(wl)
    return worldlines


# ============================================================
#  GEOMETRIC DIAGNOSTICS
# ============================================================

def compute_step_lengths(worldlines):
    """Return concatenated array of |x_{t+1} - x_t| over all worldlines."""
    steps = []
    for wl in worldlines:
        x = wl["positions"]
        if len(x) > 1:
            dx = np.diff(x)
            steps.append(np.abs(dx))
    if not steps:
        return np.array([])
    return np.concatenate(steps)


def compute_msd(worldlines, max_lag=None):
    """
    Mean squared displacement as a function of lag τ.
    MSD(τ) = ⟨(x(t+τ) - x(t))^2⟩ over all valid t, all worldlines.
    """
    # Determine max lag if not given
    if max_lag is None:
        max_len = max(len(wl["positions"]) for wl in worldlines)
        max_lag = max_len - 1
        max_lag = max(1, max_lag)

    msd = np.zeros(max_lag, dtype=float)
    counts = np.zeros(max_lag, dtype=int)

    for wl in worldlines:
        x = wl["positions"]
        n = len(x)
        for tau in range(1, min(max_lag + 1, n)):
            disp = x[tau:] - x[:-tau]
            msd[tau - 1] += np.sum(disp * disp)
            counts[tau - 1] += len(disp)

    valid = counts > 0
    msd[valid] /= counts[valid]
    msd[~valid] = np.nan

    lags = np.arange(1, max_lag + 1)
    return lags, msd, counts


def compute_displacement_distribution(worldlines, lag_values):
    """
    For selected lags, collect all displacements x(t+τ) - x(t) across worldlines.
    Returns dict: lag -> np.array of displacements.
    """
    disp_by_lag = {tau: [] for tau in lag_values}

    for wl in worldlines:
        x = wl["positions"]
        n = len(x)
        for tau in lag_values:
            if tau >= n:
                continue
            disp = x[tau:] - x[:-tau]
            disp_by_lag[tau].append(disp)

    for tau in lag_values:
        if disp_by_lag[tau]:
            disp_by_lag[tau] = np.concatenate(disp_by_lag[tau])
        else:
            disp_by_lag[tau] = np.array([])

    return disp_by_lag


# ============================================================
#  SUMMARY + SAVE
# ============================================================

def save_phase6_results(lags, msd, counts, step_lengths, disp_by_lag):
    # Save main arrays to npz
    np.savez(
        os.path.join(OUTDIR, "phase6_geometry.npz"),
        lags=lags,
        msd=msd,
        counts=counts,
        step_lengths=step_lengths,
        **{f"disp_lag_{tau}": disp_by_lag[tau] for tau in disp_by_lag}
    )

    # ---- STEP LENGTH STATS (fully safe) ----
    if step_lengths is None or len(step_lengths) == 0:
        step_stats = {
            "count": 0,
            "mean": None,
            "median": None,
            "max": None,
        }
    else:
        try:
            step_mean = float(np.mean(step_lengths))
        except Exception:
            step_mean = None
        try:
            step_median = float(np.median(step_lengths))
        except Exception:
            step_median = None
        try:
            step_max = float(np.max(step_lengths))
        except Exception:
            step_max = None

        step_stats = {
            "count": int(len(step_lengths)),
            "mean": step_mean,
            "median": step_median,
            "max": step_max,
        }

    # ---- MSD STATS (fully safe) ----
    if msd is None or len(msd) == 0:
        msd_stats = {
            "msd_at_lag_1": None,
            "msd_at_lag_max": None,
        }
    else:
        v1 = msd[0] if not np.isnan(msd[0]) else None
        vmax = msd[-1] if not np.isnan(msd[-1]) else None
        msd_stats = {
            "msd_at_lag_1": float(v1) if v1 is not None else None,
            "msd_at_lag_max": float(vmax) if vmax is not None else None,
        }

    summary = {
        "num_lags": int(len(lags)) if lags is not None else 0,
        "max_lag": int(lags[-1]) if lags is not None and len(lags) > 0 else 0,
        "step_length_stats": step_stats,
        "msd_stats": msd_stats,
        "displacement_lags": [int(tau) for tau in disp_by_lag.keys()],
    }

    with open(os.path.join(OUTDIR, "phase6_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved Phase VI geometry to results/phase6/phase6_geometry.npz")
    print("Saved Phase VI summary to results/phase6/phase6_summary.json")


# ============================================================
#  MAIN DRIVER
# ============================================================

def run_phase6_analysis(max_lag=None, disp_lags=(1, 10, 100)):
    print("Loading Phase V catalogue...")
    cat = load_phase5_catalogue()

    print("Reconstructing worldlines...")
    worldlines = reconstruct_worldlines(cat)

    print("Computing step lengths...")
    step_lengths = compute_step_lengths(worldlines)

    print("Computing MSD...")
    lags, msd, counts = compute_msd(worldlines, max_lag=max_lag)

    print(f"Computing displacement distributions for lags: {disp_lags} ...")
    disp_by_lag = compute_displacement_distribution(worldlines, lag_values=disp_lags)

    print("Saving Phase VI results...")
    save_phase6_results(lags, msd, counts, step_lengths, disp_by_lag)

    print("Phase VI analysis complete.")


if __name__ == "__main__":
    run_phase6_analysis()
