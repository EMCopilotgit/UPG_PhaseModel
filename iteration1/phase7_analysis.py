import os
import json
import numpy as np

CANONICAL_EVENTS = "phase5_events.npz"
OUTDIR = "results/phase7/"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
#  LOAD + RECONSTRUCT WORLDLINES (same as Phase V/VI)
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
#  VELOCITY + VELOCITY AUTOCORRELATION
# ============================================================

def compute_velocities(worldlines):
    """Concatenate all step velocities v = x_{t+1} - x_t across worldlines."""
    v_list = []
    for wl in worldlines:
        x = wl["positions"]
        if len(x) > 1:
            v = np.diff(x)
            v_list.append(v)
    if not v_list:
        return np.array([])
    return np.concatenate(v_list)



def compute_velocity_autocorrelation(worldlines, max_lag=None):
    """
    Velocity autocorrelation function:
    C_v(τ) = ⟨ v(t) v(t+τ) ⟩ over all valid t, all worldlines.
    """
    # Determine max lag if not given
    max_len = 0
    for wl in worldlines:
        x = wl["positions"]
        if len(x) > 1:
            max_len = max(max_len, len(x) - 1)

    if max_len == 0:
        return np.array([]), np.array([]), np.array([])

    if max_lag is None:
        max_lag = max_len - 1
        max_lag = max(1, max_lag)

    # Allocate arrays of size max_lag + 1
    cav = np.zeros(max_lag + 1, dtype=float)
    counts = np.zeros(max_lag + 1, dtype=int)

    for wl in worldlines:
        x = wl["positions"]
        if len(x) <= 1:
            continue

        v = np.diff(x)
        n = len(v)

        # Only iterate up to min(max_lag, n-1)
        max_tau = min(max_lag, n - 1)

        for tau in range(0, max_tau + 1):
            prod = v[tau:] * v[:n - tau]
            cav[tau] += np.sum(prod)
            counts[tau] += len(prod)

    valid = counts > 0
    cav[valid] /= counts[valid]
    cav[~valid] = np.nan

    lags = np.arange(0, max_lag + 1)
    return lags, cav, counts

# ============================================================
#  SIGN CHANGES (1D "TURNING")
# ============================================================

def compute_sign_change_rate(worldlines):
    """
    For each worldline, compute fraction of steps where sign(v_{t+1}) != sign(v_t).
    Return global mean over all worldlines with length >= 3.
    """
    rates = []
    for wl in worldlines:
        x = wl["positions"]
        if len(x) < 3:
            continue
        v = np.diff(x)
        # ignore zero velocities in sign; treat them as 0
        s = np.sign(v)
        # sign changes where both neighbors are nonzero and differ
        s1 = s[:-1]
        s2 = s[1:]
        mask = (s1 != 0) & (s2 != 0)
        if np.any(mask):
            changes = np.sum((s1[mask] != s2[mask]))
            total = np.sum(mask)
            rates.append(changes / total)
    if not rates:
        return None
    return float(np.mean(rates))


# ============================================================
#  SPATIAL OCCUPANCY + ENTROPY
# ============================================================

def compute_spatial_occupancy(worldlines, nbins=None):
    """
    Histogram of positions across all worldlines.
    Returns bin_centers, counts, probabilities, entropy.
    """
    all_pos = []
    for wl in worldlines:
        x = wl["positions"]
        if len(x) > 0:
            all_pos.append(x)
    if not all_pos:
        return np.array([]), np.array([]), np.array([]), None

    all_pos = np.concatenate(all_pos)
    xmin = np.min(all_pos)
    xmax = np.max(all_pos)

    if nbins is None:
        nbins = int(xmax - xmin + 1)

    counts, edges = np.histogram(all_pos, bins=nbins, range=(xmin - 0.5, xmax + 0.5))
    total = np.sum(counts)

    if total == 0:
        probs = np.zeros_like(counts, dtype=float)
        entropy = None
    else:
        probs = counts / total
        mask = probs > 0
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))

    # Correct bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, counts, probs, float(entropy) if entropy is not None else None


# ============================================================
#  SAVE
# ============================================================

def save_phase7_results(
    v_all,
    vacf_lags,
    vacf,
    vacf_counts,
    sign_change_rate,
    occ_centers,
    occ_counts,
    occ_probs,
    occ_entropy,
):
    np.savez(
        os.path.join(OUTDIR, "phase7_diagnostics.npz"),
        velocities=v_all,
        vacf_lags=vacf_lags,
        vacf=vacf,
        vacf_counts=vacf_counts,
        occ_centers=occ_centers,
        occ_counts=occ_counts,
        occ_probs=occ_probs,
    )

    # Summary JSON
    if v_all is None or len(v_all) == 0:
        v_stats = {
            "count": 0,
            "mean": None,
            "std": None,
        }
    else:
        v_stats = {
            "count": int(len(v_all)),
            "mean": float(np.mean(v_all)),
            "std": float(np.std(v_all)),
        }

    if vacf is None or len(vacf) == 0:
        vacf_stats = {
            "c0": None,
            "c1": None,
            "c_maxlag": None,
        }
    else:
        c0 = vacf[0] if not np.isnan(vacf[0]) else None
        c1 = vacf[1] if len(vacf) > 1 and not np.isnan(vacf[1]) else None
        cmax = vacf[-1] if not np.isnan(vacf[-1]) else None
        vacf_stats = {
            "c0": float(c0) if c0 is not None else None,
            "c1": float(c1) if c1 is not None else None,
            "c_maxlag": float(cmax) if cmax is not None else None,
        }

    summary = {
        "velocity_stats": v_stats,
        "vacf_stats": vacf_stats,
        "sign_change_rate": sign_change_rate,
        "spatial_entropy": occ_entropy,
        "occupancy_bins": int(len(occ_centers)) if occ_centers is not None else 0,
    }

    with open(os.path.join(OUTDIR, "phase7_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved Phase VII diagnostics to results/phase7/phase7_diagnostics.npz")
    print("Saved Phase VII summary to results/phase7/phase7_summary.json")


# ============================================================
#  MAIN DRIVER
# ============================================================

def run_phase7_analysis(max_vacf_lag=None):
    print("Loading Phase V catalogue...")
    cat = load_phase5_catalogue()

    print("Reconstructing worldlines...")
    worldlines = reconstruct_worldlines(cat)

    print("Computing velocities...")
    v_all = compute_velocities(worldlines)

    print("Computing velocity autocorrelation...")
    vacf_lags, vacf, vacf_counts = compute_velocity_autocorrelation(
        worldlines, max_lag=max_vacf_lag
    )

    print("Computing sign-change rate...")
    scr = compute_sign_change_rate(worldlines)

    print("Computing spatial occupancy and entropy...")
    occ_centers, occ_counts, occ_probs, occ_entropy = compute_spatial_occupancy(worldlines)

    print("Saving Phase VII results...")
    save_phase7_results(
        v_all,
        vacf_lags,
        vacf,
        vacf_counts,
        scr,
        occ_centers,
        occ_counts,
        occ_probs,
        occ_entropy,
    )

    print("Phase VII analysis complete.")


if __name__ == "__main__":
    run_phase7_analysis()
