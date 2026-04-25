import numpy as np

# -----------------------------------------
# 1. Norm drift
# -----------------------------------------
def compute_norm_drift(R0, T0, F0, N0, R, T, F, N):
    return (
        np.linalg.norm(R - R0),
        np.linalg.norm(T - T0),
        np.linalg.norm(F - F0),
        np.linalg.norm(N - N0)
    )


# -----------------------------------------
# 2. Stability score
# -----------------------------------------
def stability_score(R, T, F, N):
    return (
        np.std(R) + np.std(T) + np.std(F) + np.std(N)
    )


# -----------------------------------------
# 3. Recovery metric
# -----------------------------------------
def recovery_metric(history):
    # history is a list of stability scores
    return history[-1] - history[0]
