# Limitations

This repository is intentionally conservative.

## L1. Model dependence

All probability estimates depend on:
- reference class definition,
- depth truncation,
- branching assumptions,
- compute-to-observer mapping.

Different reasonable assumptions can reverse conclusions.

## L2. Unknown physics

If the simulator's substrate obeys unknown physics, physical bounds may not apply. SHMF therefore treats
bounds as conditional statements: "under known physics".

## L3. Underdetermination by data

Many proposed "simulation artifacts" are not unique; conventional physics and instrument systematics
often explain the same patterns. This motivates strict null modeling and statistical discipline.

## L4. Epistemic priors

Bayesian updates depend on likelihood models and priors. Without explicit priors, arguments become rhetorical.
SHMF requires priors to be written down.

## L5. Practical falsifiability

Even falsifiable predictions may be practically unreachable due to:
- insufficient measurement precision,
- limited accessible data,
- computational infeasibility.

SHMF treats "practical falsifiability" as a constraint separate from logical falsifiability.

