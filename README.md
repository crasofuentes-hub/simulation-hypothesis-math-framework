# Simulation Hypothesis Math Framework (SHMF)

A conservative, test-driven Python framework to formalize and analyze **Simulation Hypothesis** models as **explicit mathematical assumptions** plus **falsifiable statistical procedures**.

This repository focuses on three pillars:

1) **Nesting models** (how simulated worlds branch / nest)  
2) **Physical bounds** (conditional on known physics)  
3) **Artifact detection** (null models + multiple testing control + power scaffolding)

> Goal: keep every claim tied to a stated assumption, a reproducible computation, or a falsifiable observable.

---

## What’s implemented (v1.1+)

### M2 — Budgeted Branching (compute-constrained)
A compute-budget model where branching depth/width are constrained by a finite computational resource (per depth). Useful to reason about *branching vs. budget* trade-offs and implied observer-weight distributions.

### M3 — Stochastic Branching (priors + reproducible Monte Carlo)
A stochastic nesting model with explicit parameters and reproducible Monte Carlo simulation (seeded). Intended for sensitivity analysis: how distributions over depth/branching change under different priors.

### Artifact Detection (null model + BH-FDR + power utilities)
A minimal artifact detection stack:
- Null surrogates (Gaussian and phase-randomized)
- Empirical p-values with smoothing
- Benjamini–Hochberg FDR control
- Simple normal-approx power scaffolding (two-sample)

---

## Repository structure

- `src/shmf/` — library code
  - `mathematical_foundations/` — nesting models (M2/M3)
  - `simulation_detection/` — detection stats + artifact utilities
  - `physics_constraints/` — physical limits (conditional statements)
  - `epistemology/` — priors / Bayesian scaffolding
- `tests/` — unit tests (pytest)
- `docs/` — paper-grade documentation
  - `assumptions.md`
  - `mathematical_model.md`
  - `physical_bounds.md`
  - `falsifiability_observables.md`
  - `limitations.md`
  - `reproducibility.md`
  - `index.md`
- `REFERENCES.md` — canonical references (Bremermann / Margolus–Levitin / Landauer etc.)
- `CITATION.cff` — citation metadata
- `LICENSE` — license terms

---

## Scientific posture (what SHMF is / is not)

### SHMF is
- A **framework** for writing simulation-hypothesis arguments as:
  - stated assumptions,
  - explicit models,
  - reproducible computations,
  - falsifiable observables where possible.

### SHMF is not
- A proof that we are in a simulation.
- A substitute for experimental physics.
- A “pattern detector” without null models and error control.

---

## Quickstart

### 1) Install (editable + dev tools)
```bash
python -m pip install -U pip
pip install -e ".[dev]"
