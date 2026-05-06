import os

import numpy as np

from iteration2.phase5.phase5_dynamics import (
    Phase5Config,
    load_phase4_results,
    pick_representative_lambdas,
)
from iteration2.phase5.phase5_plot_examples import (
    plot_three_regimes_at_eta,
    plot_three_regimes_with_labels,
)
from iteration2.phase5.phase5_annotation import (
    plot_annotated_binary,
    plot_annotated_T_star,
)
from iteration2.phase5.phase5_summary import generate_phase5_summary
from iteration2.phase5.phase5_dynamics import RESULTS_DIR_PHASE5

def generate_synthetic_logs():
    """
    Generate a tiny synthetic <w> trajectory for Phase 5 test mode.
    """
    logs = {
        "episodes": [
            {"w_mean": 0.1 + 0.02 * np.sin(0.3 * k)}
            for k in range(40)   # 40 episodes, fast
        ]
    }
    return logs

def main(mode=None):
    # ============================================================
    # TEST MODE (fully synthetic, no Phase 4 dependencies)
    # ============================================================
    if mode == "test":
        print("Phase 5 running in TEST MODE")

        # Synthetic logs
        logs = generate_synthetic_logs()

        # Tiny synthetic eta/lambda grid
        etas = np.array([0.001, 0.002])
        lambdas = np.array([0.02, 0.04])

        # Minimal synthetic summary
        summary = (
            "## Phase V (TEST MODE)\n"
            "Synthetic test run completed.\n"
            "No Phase IV data loaded.\n"
            "No real trajectories analyzed.\n"
            "This mode verifies that Phase V runs end‑to‑end.\n"
        )

        summary_path = os.path.join(RESULTS_DIR_PHASE5, "phase5_summary_test.md")
        with open(summary_path, "w") as f:
            f.write(summary)

        print("Phase 5 TEST MODE complete.")
        print(f"Synthetic results saved in: {RESULTS_DIR_PHASE5}")
        return
    else:
        phase4 = load_phase4_results()
        logs = phase4["logs"]
        etas = phase4["etas"]
        lambdas = phase4["lambdas"]

        cfg = Phase5Config(etas=etas, lambdas=lambdas)

    # Choose a representative eta near the floating island
    target_eta = 0.006
    lam_base = 0.024
    lam_gap = 0.033
    lam_island = 0.045

    eta_grid, lam_base_g, lam_gap_g, lam_island_g = pick_representative_lambdas(
        etas, lambdas, target_eta, lam_base, lam_gap, lam_island
    )

    # 1) Example trajectories
    plot_three_regimes_at_eta(
        cfg,
        eta_grid,
        lam_base_g,
        lam_gap_g,
        lam_island_g,
        outname="w_mean_three_regimes.png",
    )

    plot_three_regimes_with_labels(
        cfg,
        eta_grid,
        lam_base_g,
        lam_gap_g,
        lam_island_g,
        outname="w_mean_three_regimes_labeled.png",
    )

    # 2) Annotated phase diagrams
    plot_annotated_binary()
    plot_annotated_T_star()

    # 3) Summary text
    summary = generate_phase5_summary()
    summary_path = os.path.join(RESULTS_DIR_PHASE5, "phase5_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    print("Phase 5 complete.")
    print(f"Figures and summary saved in: {RESULTS_DIR_PHASE5}")


if __name__ == "__main__":
    main()
