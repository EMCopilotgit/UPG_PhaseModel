import numpy as np

def init_logs():
    return {
        "y_em": [],
        "eps": [],
        "N": [],
        "curvature": [],
        "torsion": [],
        "n_field": [],
        "x_field": [],
    }

def log_step(logs, y_em, eps, N_unified, curvature, torsion, n, x):
    logs["y_em"].append(float(y_em))
    logs["eps"].append(float(eps))
    logs["N"].append(N_unified.copy())
    logs["curvature"].append(curvature.copy())
    logs["torsion"].append(torsion.copy())
    logs["n_field"].append(n.copy())
    logs["x_field"].append(x.copy())
