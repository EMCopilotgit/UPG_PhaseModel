#!/usr/bin/env python3
"""
Iteration 2 — Full Pipeline Runner
Phase 3 → Phase 4 → Phase 5

Supports:
    - Full mode (default)
    - Fast test mode: python3 -m iteration2.run_all test
"""

import sys
import os
import numpy as np

# -----------------------------
# Phase 3
# -----------------------------
from iteration2.phase3_2d.main_phase3_2d import main as run_phase3

# -----------------------------
# Phase 4
# -----------------------------
from iteration2.phase4.equilibrium.main_phase4 import (
    main as run_phase4_equilibrium,
)

# -----------------------------
# Phase 5
# -----------------------------
from iteration2.phase5.main_phase5 import main as run_phase5

# ============================================================
# CONFIG HELPERS
# ============================================================

def is_test_mode():
    return len(sys.argv) > 1 and sys.argv[1].lower() == "test"


# ============================================================
# RUNNERS
# ============================================================

def run_phase3_block():
    print("\n==============================")
    print(" Running Phase 3 (2D S² model)")
    print("==============================\n")

    run_phase3()   # already supports fast execution


def run_phase4_block():
    print("\n==============================")
    print(" Running Phase 4 (Equilibrium)")
    print("==============================\n")

    if is_test_mode():
        print("Phase 4 running in TEST MODE")
        run_phase4_equilibrium("test")
    else:
        print("Phase 4 running in FULL MODE")
        run_phase4_equilibrium()


def run_phase5_block():
    print("\n==============================")
    print(" Running Phase 5 (Scaling laws)")
    print("==============================\n")

    # Phase 5 is already fast unless you request large sweeps
    logs, summary = run_phase5()

    print("\nPhase 5 summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")



def run_phase5_block():
    print("\n==============================")
    print(" Running Phase 5 (Dynamics)")
    print("==============================\n")

    if is_test_mode():
        run_phase5("test")
    else:
        run_phase5()

    print("\nPhase 5 complete.")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n=======================================")
    print(" Iteration 2 — Full Pipeline Execution")
    print("=======================================\n")

    if is_test_mode():
        print(">>> RUNNING IN TEST MODE (fast) <<<\n")
    else:
        print(">>> RUNNING IN FULL MODE <<<\n")

    # Ensure results directory exists
    os.makedirs("iteration2/results", exist_ok=True)

    # ---- Phase 3 ----
    run_phase3_block()

    # ---- Phase 4 ----
    run_phase4_block()

    # ---- Phase 5 ----
    run_phase5_block()

    print("\n=======================================")
    print(" Iteration 2 pipeline complete.")
    print("=======================================\n")


if __name__ == "__main__":
    main()
