# phase5/main5.py

import numpy as np

from .io_phase5 import load_phase3_for_phase5
from .worldlines import match_worldlines
from .catalogue import build_event_catalogue, save_phase5_catalogue


def main():

    print("Loading Phase III output...")
    data = load_phase3_for_phase5("phase3_output.npz")

    defect_positions = data["defect_positions"]
    defect_charges   = data["defect_charges"]
    defect_types     = data["defect_types"]

    if defect_positions is None:
        raise RuntimeError("Phase V requires defect data from Phase III, but none was found.")

    print("Running worldline matching...")
    worldlines, events = match_worldlines(
        defect_positions=defect_positions,
        defect_charges=defect_charges,
        defect_types=defect_types,
        max_radius=3.0
    )

    print(f"Worldlines constructed: {len(worldlines)}")
    print(f"Events detected: {len(events)}")

    print("Building event catalogue...")
    catalogue = build_event_catalogue(worldlines, events)

    print("Saving Phase V results...")
    save_phase5_catalogue(catalogue, path="phase5_events.npz")

    print("Phase V complete. Results saved to phase5_events.npz")


if __name__ == "__main__":
    main()
