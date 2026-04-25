import numpy as np

def init_random_state(N, x_scale=0.1):
    x = x_scale * np.random.randn(N)
    theta = 2 * np.pi * np.random.rand(N)
    return x, theta

def init_state_for_input(x_in, N, jitter=0.01):
    # Initialize all agents near the input value
    x = x_in + jitter * np.random.randn(N)
    theta = 2 * np.pi * np.random.rand(N)
    return x, theta
