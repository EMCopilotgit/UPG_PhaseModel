# **UPG Phase Model — Iteration 1**

A modular, phase‑structured computational framework for exploring geometric learning, defect dynamics, and multi‑phase diagnostics.  
This repository contains the complete codebase for **Iteration 1** of the UPG Phase Model, including all simulation phases, diagnostics, worldline tracking, and catalogue generation.

---

## **Overview**

The UPG Phase Model is organized into a sequence of computational phases, each responsible for a distinct transformation or diagnostic operation.  
Iteration 1 provides a reproducible pipeline from raw phase inputs through worldline extraction and event catalogue construction.

The project emphasizes:

- clear modular boundaries  
- reproducible numerical experiments  
- phase‑based geometric interpretation  
- clean separation of simulation, diagnostics, and visualization  

---

## **Repository Structure**

```
UPG_PhaseModel/
│
├── README.md
├── iteration1/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase7/
│   └── utils/
│
└── results/        (created automatically when running phases)
```

Each `phaseN/` directory contains:

- simulation code  
- diagnostics  
- plotting utilities  
- a `mainN.py` entry point for running that phase independently  

---

## **Running the Pipeline**

To run a specific phase:

```bash
python3 -m iteration1.phase3.main3
```

or from inside the project root:

```bash
python3 iteration1/phase3/main3.py
```

Each phase writes its outputs to:

```
results/phaseN/
```

Plots, `.npz` files, and catalogues are stored automatically.

---

## **Dependencies**

The core dependencies are:

- Python 3.10+
- NumPy
- Matplotlib
- SciPy (for selected diagnostics)
- tqdm (optional, for progress bars)

Install via:

```bash
pip install -r requirements.txt
```
---

## **Versioning**

This repository corresponds to:

**Iteration 1 — Stable Research Release**

Future iterations will be added as separate top‑level directories:

```
iteration2/
iteration3/
...
```

ensuring long‑term reproducibility and archival clarity.

---

## **Citation**

If you use this code in academic work, please cite the accompanying manuscript:

**“Multiphase Diagnostics for Geometric Learning”**  
(Preprint link / Zenodo DOI will be added here.)

---

## **License**

MIT License

Copyright (c) 2026 Edward X. Meisner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---

## **Contact**

For questions or collaboration inquiries, please reach out via GitHub Issues or Discussions.


