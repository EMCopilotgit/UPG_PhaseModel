import numpy as np

def detect_defects(n):
    """
    Very simple defect detector:
    flags sites where |dn| is large.
    """
    dn = np.roll(n, -1, axis=0) - n
    mag = np.linalg.norm(dn, axis=1)

    threshold = np.mean(mag) + 2*np.std(mag)
    defect_positions = np.where(mag > threshold)[0]

    return defect_positions, mag[defect_positions]
