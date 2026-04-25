# phase5/io_phase5.py
import numpy as np

def load_phase3_for_phase5(path="phase3_output.npz"):
    data = np.load(path, allow_pickle=True)

    R = data["R"]
    T = data["T"]
    F = data["F"]
    N = data["N"]
    x = data["x"]

    defect_positions = data.get("defect_positions", None)
    defect_charges   = data.get("defect_charges", None)
    defect_types     = data.get("defect_types", None)

    if defect_positions is not None:
        defect_positions = defect_positions.astype(object)
    if defect_charges is not None:
        defect_charges = defect_charges.astype(object)
    if defect_types is not None:
        defect_types = defect_types.astype(object)

    return {
        "R": R,
        "T": T,
        "F": F,
        "N": N,
        "x": x,
        "defect_positions": defect_positions,
        "defect_charges": defect_charges,
        "defect_types": defect_types,
    }
