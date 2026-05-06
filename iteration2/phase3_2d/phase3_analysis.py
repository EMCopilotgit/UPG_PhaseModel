import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np

from iteration2.phase3_2d.run_phase3_2d import run_phase3_2d

RESULTS_DIR_PHASE3 = "iteration2/results/phase3_2d"
os.makedirs(RESULTS_DIR_PHASE3, exist_ok=True)

@dataclass
class Phase3Config:
    T_steps: int = 500
    Nx: int = 64
    Ny: int = 64


def run_single(cfg: Phase3Config) -> Dict[str, Any]:
    logs, final_state = run_phase3_2d(
        T_steps=cfg.T_steps,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
    )
    return {"logs": logs, "final_state": final_state}


def extract_Q_history(logs: Dict[str, Any]) -> np.ndarray:
    return np.array([d["Q_total"] for d in logs["defects"]])


def extract_defect_count_history(logs: Dict[str, Any]) -> np.ndarray:
    return np.array([d["defect_count"] for d in logs["defects"]])


def sample_Q_distribution(
    cfg: Phase3Config,
    num_runs: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    Q_values = []
    defect_counts = []

    for k in range(num_runs):
        logs, _ = run_phase3_2d(
            T_steps=cfg.T_steps,
            Nx=cfg.Nx,
            Ny=cfg.Ny,
        )
        Q = logs["defects"][-1]["Q_total"]
        dc = logs["defects"][-1]["defect_count"]

        Q_values.append(Q)
        defect_counts.append(dc)

        print(f"run {k+1}/{num_runs}: Q_total={Q}, defects={dc}")

    Q_values = np.array(Q_values)
    defect_counts = np.array(defect_counts)

    np.save(os.path.join(RESULTS_DIR_PHASE3, "Q_values.npy"), Q_values)
    np.save(os.path.join(RESULTS_DIR_PHASE3, "defect_counts.npy"), defect_counts)

    return Q_values, defect_counts
