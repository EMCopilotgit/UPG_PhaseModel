# phase3/defects_phase3.py

import numpy as np

def detect_defects_1d(N_field):
    """
    Detect defects in a 1D unified field N_field based on sign changes.
    A defect occurs where N_field[i] * N_field[i+1] < 0.

    Position: midpoint between i and i+1 (index space).
    Charge: +1 if N_field[i+1] - N_field[i] > 0, else -1.
    Type: "zero_crossing".
    """
    N = N_field
    positions = []
    charges = []
    types = []

    for i in range(len(N) - 1):
        if N[i] == 0 or N[i+1] == 0:
            continue
        if N[i] * N[i+1] < 0:
            pos = 0.5 * (i + i + 1)
            slope = N[i+1] - N[i]
            charge = 1 if slope > 0 else -1
            positions.append([pos])
            charges.append(charge)
            types.append("zero_crossing")

    if len(positions) == 0:
        return (
            np.zeros((0, 1), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=object),
        )

    return (
        np.array(positions, dtype=float),
        np.array(charges, dtype=int),
        np.array(types, dtype=object),
    )
