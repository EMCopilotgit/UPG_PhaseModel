# phase3/main3.py

import numpy as np

from iteration1.phase3.dynamics_phase3 import run_phase3, default_params_phase3
from iteration1.phase3.diagnostics_phase3 import save_phase3_results, plot_phase3_single


def main():
    inputs = np.linspace(-2*np.pi, 2*np.pi, 12)
    params = default_params_phase3()

    results = run_phase3(
        inputs=inputs,
        T_steps=6000,
        N=512,
        params=params
    )

    save_phase3_results(results, results_dir="results/phase3", tag="continual")

    # Plot one example from each phase
    plot_phase3_single(results["sin1"][0], "sin1")
    plot_phase3_single(results["cos"][0], "cos")
    plot_phase3_single(results["sin2"][0], "sin2")
    
    # Extract final fields for Phase IV
    R = results["R"]
    T = results["T"]
    F = results["F"]
    N = results["N"]
    x = results["x"]

    # --- Unified Phase III save block (compatible with Phase IV & Phase V) ---

    # Extract defect arrays from results (if present)
    defect_positions = results.get("defect_positions", None)
    defect_charges   = results.get("defect_charges", None)
    defect_types     = results.get("defect_types", None)

    # Convert to object arrays if they exist
    if defect_positions is not None:
        defect_positions = np.array(defect_positions, dtype=object)
    if defect_charges is not None:
        defect_charges = np.array(defect_charges, dtype=object)
    if defect_types is not None:
        defect_types = np.array(defect_types, dtype=object)

    np.savez(
        "phase3_output.npz",

        # --- Phase IV fields (unchanged) ---
        R=R,
        T=T,
        F=F,
        N=N,
        x=x,

        # --- Phase V fields (added safely) ---
        defect_positions=defect_positions,
        defect_charges=defect_charges,
        defect_types=defect_types,

        # --- Optional metadata ---
        version="3.1"
    )
if __name__ == "__main__":
    main()
