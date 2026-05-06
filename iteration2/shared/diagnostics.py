import numpy as np

def check_normalization(n):
    """
    Returns max deviation from unit length.
    """
    norms = np.linalg.norm(n, axis=1)
    deviation = np.max(np.abs(norms - 1.0))
    return deviation


def compute_energy(n):
    """
    Simple energy functional:
    E = sum |dn|^2
    """
    dn = np.roll(n, -1, axis=0) - n
    return float(np.sum(np.linalg.norm(dn, axis=1)**2))



def detect_instability(n):
    norms = np.linalg.norm(n, axis=1)
    if np.any(norms < 0.9) or np.any(norms > 1.1):
        return True

    # gradient magnitude
    dn = np.roll(n, -1, axis=0) - n
    mag = np.linalg.norm(dn, axis=1)

    # adaptive threshold
    threshold = np.mean(mag) + 3*np.std(mag)

    return np.any(mag > threshold)


def diagnostics_summary(n):
    """
    Returns a dictionary of Phase I diagnostics.
    """
    return {
        "max_norm_deviation": check_normalization(n),
        "energy": compute_energy(n),
        "unstable": detect_instability(n)
    }
