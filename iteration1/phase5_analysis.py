import numpy as np
import json
import os
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
#  SETUP
# ============================================================

CANONICAL_EVENTS = "phase5_events.npz"
OUTDIR = "results/phase5/"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
#  LOAD + RECONSTRUCT WORLDLINES / EVENTS
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


def reconstruct_events(cat):
    events = []
    for i in range(len(cat["event_types"])):
        ev = {
            "type": cat["event_types"][i],
            "time": int(cat["event_times"][i]),
            "worldline_ids": np.array(cat["event_worldline_ids"][i]),
        }
        events.append(ev)
    return events


# ============================================================
#  WORLDLINE LENGTHS
# ============================================================

def compute_worldline_lengths(worldlines):
    return np.array([len(wl["times"]) for wl in worldlines])


def plot_worldline_lengths(lengths):
    plt.figure(figsize=(8,5))
    plt.hist(lengths, bins=60, color="steelblue", alpha=0.85)
    plt.xlabel("Worldline length (timesteps)")
    plt.ylabel("Count")
    plt.title("Phase V: Worldline Length Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "worldline_lengths.png"))
    plt.close()


# ============================================================
#  EVENT TYPES
# ============================================================

def compute_event_type_counts(events):
    return Counter(ev["type"] for ev in events)


def plot_event_types(counts):
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(8,5))
    plt.bar(labels, values, color="darkred", alpha=0.85)
    plt.xlabel("Event type")
    plt.ylabel("Count")
    plt.title("Phase V: Event Type Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "event_types.png"))
    plt.close()


# ============================================================
#  SPATIAL DISTRIBUTION
# ============================================================

def compute_spatial_distribution(worldlines):
    xs = []
    for wl in worldlines:
        xs.extend(wl["positions"])
    return np.array(xs)


def plot_spatial_distribution(xs):
    plt.figure(figsize=(8,5))
    plt.hist(xs, bins=100, color="purple", alpha=0.75)
    plt.xlabel("Spatial position")
    plt.ylabel("Count")
    plt.title("Phase V: Spatial Distribution of Defects")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "spatial_distribution.png"))
    plt.close()


# ============================================================
#  LIFETIME HISTOGRAM
# ============================================================

def plot_lifetime_histogram(lengths):
    plt.figure(figsize=(8,5))
    plt.hist(lengths, bins=80, color="green", alpha=0.75)
    plt.xlabel("Lifetime (timesteps)")
    plt.ylabel("Count")
    plt.title("Phase V: Defect Lifetime Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "lifetime_histogram.png"))
    plt.close()


# ============================================================
#  ANNIHILATION PATTERNS
# ============================================================

def compute_annihilation_separations(events, worldlines):
    separations = []

    # Build lookup table for worldline positions by time
    wl_pos_by_time = {}
    for wl in worldlines:
        wl_pos_by_time[wl["id"]] = dict(zip(wl["times"], wl["positions"]))

    for ev in events:
        if ev["type"] != "annihilation":
            continue

        ids = ev["worldline_ids"]
        t = ev["time"]

        if len(ids) == 2:
            x1 = wl_pos_by_time[ids[0]].get(t, None)
            x2 = wl_pos_by_time[ids[1]].get(t, None)
            if x1 is not None and x2 is not None:
                separations.append(abs(x1 - x2))

    return np.array(separations)


def plot_annihilation_separations(separations):
    if len(separations) == 0:
        print("No annihilation separations to plot.")
        return

    plt.figure(figsize=(8,5))
    plt.hist(separations, bins=50, color="orange", alpha=0.8)
    plt.xlabel("Separation at annihilation")
    plt.ylabel("Count")
    plt.title("Phase V: Annihilation Separation Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "annihilation_separations.png"))
    plt.close()


# ============================================================
#  SUMMARY JSON
# ============================================================

def save_summary(worldlines, events, lengths, separations):
    event_counts = compute_event_type_counts(events)

    summary = {
        "num_worldlines": len(worldlines),
        "num_events": len(events),
        "event_counts": dict(event_counts),
        "lifetime_stats": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
        },
        "annihilation": {
            "num_annihilations": int(event_counts.get("annihilation", 0)),
            "mean_separation": float(np.mean(separations)) if len(separations) else None
        }
    }

    with open(os.path.join(OUTDIR, "phase5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary to results/phase5/phase5_summary.json")


# ============================================================
#  MAIN DRIVER
# ============================================================

def run_phase5_analysis():
    print("Loading Phase V catalogue...")
    cat = load_phase5_catalogue()

    print("Reconstructing worldlines and events...")
    worldlines = reconstruct_worldlines(cat)
    events = reconstruct_events(cat)

    print("Computing worldline lengths...")
    lengths = compute_worldline_lengths(worldlines)

    print("Computing spatial distribution...")
    xs = compute_spatial_distribution(worldlines)

    print("Computing annihilation separations...")
    separations = compute_annihilation_separations(events, worldlines)

    print("Generating plots...")
    plot_worldline_lengths(lengths)
    plot_event_types(compute_event_type_counts(events))
    plot_spatial_distribution(xs)
    plot_lifetime_histogram(lengths)
    plot_annihilation_separations(separations)

    print("Saving summary JSON...")
    save_summary(worldlines, events, lengths, separations)

    print("Phase V analysis complete.")


if __name__ == "__main__":
    run_phase5_analysis()
