# phase5/catalogue.py
import numpy as np

def build_event_catalogue(worldlines, events):
    """
    worldlines: list of worldline dicts
    events: list of event dicts

    Returns a dict ready to np.savez.
    """
    wl_ids = np.array([wl["id"] for wl in worldlines], dtype=int)
    wl_lengths = np.array([len(wl["times"]) for wl in worldlines], dtype=int)
    wl_times = np.array([np.array(wl["times"], dtype=int) for wl in worldlines], dtype=object)
    wl_positions = np.array([np.array(wl["positions"]) for wl in worldlines], dtype=object)
    wl_charges = np.array([np.array(wl["charges"], dtype=int) for wl in worldlines], dtype=object)
    wl_types = np.array([np.array(wl["types"], dtype=object) for wl in worldlines], dtype=object)

    event_types = np.array([e["type"] for e in events], dtype=object)
    event_times = np.array([e["time"] for e in events], dtype=int)
    event_wl_ids = np.array([np.array(e["worldline_ids"], dtype=int) for e in events], dtype=object)

    return {
        "worldline_ids": wl_ids,
        "worldline_lengths": wl_lengths,
        "worldline_times": wl_times,
        "worldline_positions": wl_positions,
        "worldline_charges": wl_charges,
        "worldline_types": wl_types,
        "event_types": event_types,
        "event_times": event_times,
        "event_worldline_ids": event_wl_ids,
        "version": "5.0",
    }

def save_phase5_catalogue(catalogue, path="phase5_events.npz"):
    """
    Saves the Phase V event catalogue to a .npz file.
    """
    np.savez(path, **catalogue)

