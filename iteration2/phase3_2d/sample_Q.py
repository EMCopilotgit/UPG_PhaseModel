import numpy as np

from iteration2.phase3_2d.run_phase3_2d import run_phase3_2d


def sample_Q(num_runs=20, T_steps=500, Nx=64, Ny=64):
    Q_values = []
    defect_counts = []

    for k in range(num_runs):
        logs, _ = run_phase3_2d(T_steps=T_steps, Nx=Nx, Ny=Ny)
        Q = logs["defects"][-1]["Q_total"]
        dc = logs["defects"][-1]["defect_count"]

        Q_values.append(Q)
        defect_counts.append(dc)

        print(f"run {k+1}/{num_runs}: Q_total={Q}, defects={dc}")

    return np.array(Q_values), np.array(defect_counts)
