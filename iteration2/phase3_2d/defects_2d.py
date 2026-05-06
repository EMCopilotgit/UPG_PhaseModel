import numpy as np

def plaquette_charge(n):
    """
    Compute the topological charge density q[i,j] on a 2D S^2 field n[i,j,3].

    Uses the oriented solid angle of each plaquette:
        q = (1 / 4π) * Ω_plaquette

    where Ω is the signed spherical area spanned by the four spins.
    """
    # Shifted fields
    n_i1 = np.roll(n, -1, axis=0)
    n_j1 = np.roll(n, -1, axis=1)
    n_i1j1 = np.roll(n_i1, -1, axis=1)

    # Compute oriented solid angle using the standard formula
    def solid_angle(a, b, c):
        num = np.einsum('...i,...i', a, np.cross(b, c))
        den = 1.0 + np.einsum('...i,...i', a, b) \
                  + np.einsum('...i,...i', b, c) \
                  + np.einsum('...i,...i', c, a)
        return 2.0 * np.arctan2(num, den)

    Ω1 = solid_angle(n, n_i1, n_j1)
    Ω2 = solid_angle(n_i1j1, n_j1, n_i1)

    q = (Ω1 + Ω2) / (4.0 * np.pi)
    return q


def defects_summary_2d(n):
    """
    Return:
        - Q_total
        - defect_count
        - positions (optional)
        - charges (optional)

    Phase 4 only needs Q_total and q_field.
    """
    q = plaquette_charge(n)
    Q_total = float(np.sum(q))

    # Simple defect detection: threshold on |q|
    threshold = 0.2 * np.max(np.abs(q))
    mask = np.abs(q) > threshold
    positions = np.argwhere(mask)
    charges = q[mask]

    return {
        "q_field": q,
        "Q_total": Q_total,
        "defect_count": len(positions),
        "positions": positions,
        "charges": charges,
    }
