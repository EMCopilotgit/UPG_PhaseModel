import os
import numpy as np

from iteration2.phase3_2d.phase3_analysis import (
    Phase3Config,
    run_single,
    sample_Q_distribution,
    RESULTS_DIR_PHASE3,
)

from iteration2.phase3_2d.phase3_plots import (
    plot_Q_history,
    plot_defect_count_history,
    plot_Q_vs_defects,
)


def main():
    # ------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------
    cfg = Phase3Config(
        T_steps=500,
        Nx=64,
        Ny=64,
    )

    print("Running Phase III (2D S² dynamics)...")

    # ------------------------------------------------------------
    # 2. Run a single Phase III simulation
    # ------------------------------------------------------------
    result = run_single(cfg)
    logs = result["logs"]
    final_state = result["final_state"]

    # Extract q(x,y) from final defects summary
   
    # ------------------------------------------------------------
    # 3. Save plots for the single run
    # ------------------------------------------------------------
    print("Saving Phase III diagnostic plots...")

    plot_Q_history(logs, outname="Q_history.png")
    plot_defect_count_history(logs, outname="defect_count_history.png")

    # ------------------------------------------------------------
    # 4. Run multi-sample Q_total distribution
    # ------------------------------------------------------------
    print("Sampling Q_total distribution across multiple runs...")
    Q_values, defect_counts = sample_Q_distribution(cfg, num_runs=20)

    # Scatter plot: Q_total vs defect count
    plot_Q_vs_defects(Q_values, defect_counts, outname="Q_vs_defects_scatter.png")

    # ------------------------------------------------------------
    # 5. Save raw arrays
    # ------------------------------------------------------------
    np.save(os.path.join(RESULTS_DIR_PHASE3, "Q_values.npy"), Q_values)
    np.save(os.path.join(RESULTS_DIR_PHASE3, "defect_counts.npy"), defect_counts)

    print("Phase III complete.")
    print(f"Results saved in: {RESULTS_DIR_PHASE3}")


if __name__ == "__main__":
    main()
