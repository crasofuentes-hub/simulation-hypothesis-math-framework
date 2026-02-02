# Assumptions

This repository treats the Simulation Hypothesis as a **model class** rather than a metaphysical assertion.
Any quantitative conclusion depends on explicit assumptions. SHMF separates:

- **Structural assumptions** (how simulations nest, branch, and allocate resources),
- **Physical assumptions** (what bounds constrain information processing),
- **Observational assumptions** (what data streams and tests are admissible).

## A1. Reference class and counting measure

We require a reference class of "observers" (or observer-moments). A typical statistical argument counts
observers in base reality and in simulated realities.

Assumption A1: There exists a well-defined counting measure over observer-moments that does not
change under reparameterization of time resolution or coarse-graining.

If A1 fails, probabilities such as "being in a simulation" can become ill-defined.

## A2. Nested simulation generativity

Let depth \(d = 0\) denote base reality; \(d \ge 1\) denote simulated levels.

Assumption A2: A simulation at depth \(d\) may generate \(b\) child simulations at depth \(d+1\), where
\(b\) may be constant or drawn from a distribution.

We explicitly model both:
- **Deterministic branching**: \(b\) fixed,
- **Stochastic branching**: \(b \sim \mathcal{D}_b\).

## A3. Compute budget and efficiency

Assumption A3: A simulator has an effective compute budget \(C_0\) (normalized units). The budget available
to sustain simulation depth \(d\) is:

\[
C_d = C_0 \, r^d
\]

where \(0 < r \le 1\) is an efficiency/attenuation factor capturing overhead, fidelity, and thermodynamic cost.

Interpretation:
- \(r < 1\): deeper levels have less available compute per simulated "unit time".
- \(r = 1\): no attenuation (a limiting idealization; generally unphysical).

## A4. Fidelity and coarse-graining

Assumption A4: A simulator may trade fidelity for resource savings via coarse-graining. In that case, a simulated
world may be dynamically consistent at macroscopic scales while being incomplete or discretized at microscopic scales.

We treat this as a hypothesis about **observable artifacts**, not as a claim.

## A5. Physical bounds apply to simulators

Assumption A5: The simulator (or the substrate implementing the simulation) obeys physical bounds on computation,
e.g., Landauer dissipation, Margolus–Levitin speed limit, and finite information density constraints.

If A5 fails (e.g., unknown physics), then SHMF results become conditional.

## A6. Observational admissibility and falsifiability

Assumption A6: A claim is considered scientific only if it implies at least one **operational test** that could
update credences (Bayesian or otherwise) given real data and a null model.

SHMF prioritizes test designs where:
- the null model is explicit,
- false positive rates are controlled,
- power is estimated for realistic sample sizes.

