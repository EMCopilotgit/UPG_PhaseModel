

# **README.md — UPG Phase Model (Iteration 2)**

```markdown
# UPG Phase Model — Iteration 2  
Geometric Learning Framework: Phase III → Phase IV → Phase V Diagnostics

This repository contains the full implementation of **Iteration 2** of the UPG Phase Model, extending the geometric learning framework introduced in Iteration 1. Iteration 2 introduces a multi‑phase diagnostic pipeline that characterizes geometric learning dynamics across the (η, λ) parameter plane.

A versioned archival snapshot of this iteration is preserved on Zenodo:

**Zenodo DOI:** (https://doi.org/10.5281/zenodo.20054630)  
*(replace with the DOI Zenodo assigns once you publish the release)*

---

## Overview

Iteration 2 adds three major components to the UPG Phase Model:

### **Phase 3 — 2D S² Dynamics**
A spatially extended S² model with:
- local geometric diffusion  
- defect creation/annihilation  
- global topological charge diagnostics  
- Q‑distribution sampling across multiple runs  

Outputs are saved in:
```
iteration2/results/phase3_2d/
```

---

### **Phase 4 — Equilibrium Detection Across (η, λ)**
A grid sweep over learning rate η and plasticity strength λ.  
For each point, the system runs multiple episodes and determines whether the plasticity field ⟨w⟩ equilibrates.

Outputs include:
- `T_star(η, λ)` heatmap  
- binary equilibrium diagram  
- λ\* stability curve  
- 3D surface plots  

Saved in:
```
iteration2/results/phase4/
```

---

### **Phase 5 — Dynamical Regime Characterization**
Phase 5 interprets the Phase 4 phase structure by examining full ⟨w⟩ trajectories at selected (η, λ).  
Dynamics are classified into:

- **equilibrium** — stationary tail, low variance  
- **drift** — persistent trend  
- **oscillatory** — large variance around a mean  

These regimes explain the Phase 4 features:
- base equilibrium band  
- instability gap  
- re‑entrant equilibrium island  

Outputs are saved in:
```
iteration2/results/phase5/
```

---

## Running the Pipeline

### **Full mode**
Runs the complete Phase 3 → Phase 4 → Phase 5 pipeline:

```bash
python3 -m iteration2.run_all
```

### **Test mode**
Fast execution using reduced sweeps and synthetic Phase 5 diagnostics:

```bash
python3 -m iteration2.run_all test
```

Test mode is designed for:
- quick verification  
- CI workflows  
- development iteration  

---

## Repository Structure

```
iteration2/
    phase3_2d/
        main_phase3_2d.py
        ...
    phase4/
        equilibrium/
            main_phase4.py
            ...
    phase5/
        main_phase5.py
        phase5_dynamics.py
        phase5_annotation.py
        phase5_plots.py
        phase5_summary.py
    run_all.py
    results/
        phase3_2d/
        phase4/
        phase5/
```

---

## Citation

If you use this code or its results, please cite the Zenodo archive:

```
Meisner, Edward (2026). UPG Phase Model — Iteration 2 (v2.0.0). Zenodo.
[https://doi.org/XXXXXXXXX](https://doi.org/10.5281/zenodo.20054630)
```

*(Replace with the DOI Zenodo assigns.)*

---

## License

MIT License. See `LICENSE` for details.

---

## Notes

Iteration 2 focuses on:
- refining the geometric learning model  
- mapping equilibrium structure  
- interpreting dynamical regimes  

A manuscript is not required for this iteration; the Zenodo archive and this README serve as the formal record of the computational results.
```
