# Mathematical Model

This section formalizes nested simulations using two coupled processes:

1) **Resource attenuation** across depth, and  
2) **Branching** (the number of child simulations spawned per simulation).

We emphasize that different choices yield qualitatively different conclusions.

## Definitions

- Depth \(d \in \{0,1,2,\dots\}\), where \(d=0\) is base reality.
- Compute budget at depth \(d\): \(C_d\).
- Attenuation/efficiency factor: \(r \in (0,1]\).
- Branching factor at depth \(d\): \(b_d\) (constant or random).
- Number of simulations at depth \(d\): \(N_d\).

## Model M1: Deterministic branching and deterministic attenuation

Assume constant branching \(b_d = b\) and attenuation \(C_d = C_0 r^d\).

Then:
\[
N_d = b^d
\]

If "observer mass" per simulation scales with compute budget, a simple proxy is:
\[
W_d \propto N_d \, C_d = b^d \, C_0 r^d = C_0 (br)^d
\]

### Regimes

- **Subcritical** \(br < 1\): deeper levels contribute less total observer weight.
- **Critical** \(br = 1\): contributions constant per depth; sums are depth-sensitive.
- **Supercritical** \(br > 1\): deeper levels dominate (if unbounded), requiring additional constraints.

## Model M2: Budgeted branching (compute-constrained)

In realistic systems, branching cannot be independent of budget.

Let the cost per child simulation be \(k > 0\) in compute units. Then a budget constraint is:

\[
b_d \le \left\lfloor \frac{C_d}{k} \right\rfloor
\]

This directly prevents runaway branching when \(C_d\) attenuates rapidly.

## Model M3: Stochastic branching with priors

Let \(b_d \sim \mathcal{D}_b(\theta)\), with prior \(p(\theta)\). A Bayesian treatment yields distributions
over \(N_d\) and associated weight \(W_d\). This is the correct approach when there is genuine uncertainty about
simulation generativity.

## Probability of being at a given depth

A common estimator is:

\[
P(d) = \frac{W_d}{\sum_{i \ge 0} W_i}
\]

The key point is that \(P(d)\) is not a single number without specifying:
- \(W_d\) (what "counts" as an observer),
- truncation depth (finite vs infinite),
- priors over \(b_d\), \(r\), and resource costs.

## Connection to current SHMF code

The module `shmf.mathematical_foundations.nested` currently provides primitives to:
- compute depth-dependent budgets under \(C_d = C_0 r^d\),
- aggregate totals,
- sanity-check special cases (e.g., \(r=1\)).

Planned extension (paper-grade): implement M2 and M3 as first-class models with:
- explicit parameter objects,
- optional stochastic simulation,
- closed-form regimes where available,
- deterministic reproducibility hooks (fixed RNG seeds).

## Implemented in SHMF

- **M2 (compute-constrained)**: `shmf.mathematical_foundations.BudgetedBranchingModel`
  - enforces: \( b_d = \lfloor C_d / k \rfloor \)
  - provides: `simulations_per_depth`, `observer_weight_per_depth`, `probability_depth_distribution`

- **M3 (stochastic + Monte Carlo, reproducible)**: `shmf.mathematical_foundations.StochasticBranchingModel`
  - priors: `fixed_rate`, `budget_scaled`
  - provides: `simulate_once`, `monte_carlo`, `probability_depth_distribution_mc`

