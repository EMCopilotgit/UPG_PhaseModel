import numpy as np

def init_plasticity_field(Nx, Ny, w0=0.0):
    """
    Initialize the plasticity/memory field w[i,j].
    """
    return np.full((Nx, Ny), float(w0))
