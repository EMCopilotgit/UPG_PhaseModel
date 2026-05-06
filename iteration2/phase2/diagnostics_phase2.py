import numpy as np

def compute_coherence(n):
    """
    Coherence = mean dot product between neighbors.
    Range: [-1, 1]
    """
    n_shift = np.roll(n, -1, axis=0)
    dots = np.sum(n * n_shift, axis=1)
    return float(np.mean(dots))


def compute_curvature_stats(curvature):
    """
    Returns mean, std, max curvature.
    """
    return {
        "curv_mean": float(np.mean(curvature)),
        "curv_std": float(np.std(curvature)),
        "curv_max": float(np.max(curvature))
    }


def compute_torsion_stats(torsion):
    """
    Torsion magnitude statistics.
    """
    mag = np.linalg.norm(torsion, axis=1)
    return {
        "tors_mean": float(np.mean(mag)),
        "tors_std": float(np.std(mag)),
        "tors_max": float(np.max(mag))
    }


def compute_unified_stats(N_unified):
    """
    Unified field magnitude statistics.
    """
    return {
        "N_mean": float(np.mean(N_unified)),
        "N_std": float(np.std(N_unified)),
        "N_max": float(np.max(N_unified))
    }


def diagnostics_phase2(n, curvature, torsion, N_unified):
    """
    Full geometric diagnostics for Phase II.
    """
    return {
        "coherence": compute_coherence(n),
        **compute_curvature_stats(curvature),
        **compute_torsion_stats(torsion),
        **compute_unified_stats(N_unified)
    }
