import numpy as np
from iteration2.phase3_2d.defects_2d import plaquette_charge

def update_plasticity_global(w, n, Q_total, eta=0.01, lam=0.01):
    """
    Stabilized plasticity update with global topological signal:

        w_new = (1 - lam) * w - eta * |Q_total| * |q(x,y)|

    """
    q = plaquette_charge(n)
    gain = abs(Q_total)

    w_new = (1.0 - lam) * w - eta * gain * np.abs(q)
    return w_new, q
