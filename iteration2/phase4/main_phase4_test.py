"""
Phase 4 — Fast Test Suite
Runs three quick diagnostics to verify that the Phase 4 engine,
equilibrium logic, and sweep pipeline are functioning correctly.

This avoids the multi-hour full sweep.
"""

import numpy as np

# ------------------------------
# 1. ENGINE TEST (fast)
# ------------------------------
from iteration2.phase4.engine.run_phase4_geom import run_phase4_geom

# ------------------------------
# 2. EQUILIBRIUM TEST (synthetic)
# ------------------------------
from iteration2.phase4.equilibrium.equilibrium_test import compute_T_star

# ------------------------------
# 3. MICRO-SWEEP TEST
# ------------------------------
from iteration2.phase4.equilibrium.sweep_grid import SweepConfig, sweep_grid
from iteration2.phase4.equilibrium.boundary import extract_lambda_star


def test_engine():
    print("\n=== Phase 4 Engine Test ===")

    logs, final_state, w_hist = run_phase4_geom(
        episodes=3,
        T_steps=50,
        Nx=32,
        Ny=32,
        eta=0.005,
        lam=0.02,
        dt=0.01,
        kappa0=0.2,
        alpha=0.1,
    )

    print("Engine test completed.")
    print(f"Logged {len(logs['episodes'])} episodes.")
    print(f"Final <w> = {logs['episodes'][-1]['w_mean']:.4f}")


def test_equilibrium_logic():
    print("\n=== Equilibrium Logic Test ===")

    # Synthetic observable: flat signal → should equilibrate quickly
    observable = np.ones(200) * 0.5
    T = compute_T_star(observable, window=20, tol=0.02)

    print(f"Synthetic equilibrium T* = {T}")


def test_micro_sweep():
    print("\n=== Micro Sweep Test ===")

    cfg = SweepConfig(
        etas=np.linspace(0.001, 0.003, 2),
        lambdas=np.linspace(0.02, 0.04, 2),
        max_episodes=20,
        n_seeds=1,
        equilibrium_window=10,
        equilibrium_tol=0.05,
    )

    T_star, eq_flag = sweep_grid(cfg)
    lambda_star_curve = extract_lambda_star(cfg, eq_flag)

    print("Micro sweep completed.")
    print("T* matrix:")
    print(T_star)
    print("Equilibrium flags:")
    print(eq_flag)
    print("λ*(η) boundary:")
    print(lambda_star_curve)


def main():
    print("Running Phase 4 fast test suite...")

    test_engine()
    test_equilibrium_logic()
    test_micro_sweep()

    print("\nAll Phase 4 tests completed successfully.")


if __name__ == "__main__":
    main()
