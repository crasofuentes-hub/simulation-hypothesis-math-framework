# Physical Bounds

This section summarizes canonical constraints on computation that are relevant to any simulator implemented
in known physics.

## Landauer bound (minimum dissipation per bit erase)

At temperature \(T\), erasing one bit requires at least:

\[
E_{\min} = k_B T \ln 2
\]

This is a lower bound on energy dissipation for logically irreversible operations.

## Margolus–Levitin bound (max logical operations per energy)

Given available energy \(E\), the maximum rate of distinct state transitions is bounded by:

\[
f_{\max} \le \frac{2E}{h}
\]

where \(h\) is Planck's constant.

Interpretation: computation has a speed limit per unit energy.

## Finite information density / holographic-type constraints

Multiple results suggest upper limits on the amount of information that can be stored in a finite region,
scaling with area rather than volume in certain regimes. The exact applicability depends on the physical
theory assumed (quantum gravity context).

SHMF treats this as a conditional bound: if information density is finite, then an arbitrarily detailed
simulation of a large universe becomes substrate-limited.

## Canonical constants and reproducibility

Paper-grade computations must:
- cite the source of constants (CODATA),
- specify units unambiguously,
- avoid mixing conventions (e.g., \(h\) vs \(\hbar\)).

Planned extension (paper-grade):
- add `docs/constants.md` with a traceable constants table and provenance,
- optionally add unit-aware calculations (e.g., via Pint) behind an optional dependency.

## Connection to current SHMF code

`shmf.physics_constraints.limits` currently includes:
- Landauer energy per bit,
- Margolus–Levitin operations per second per Joule.

Planned extension (paper-grade):
- add careful parameter validation and documented conventions,
- add citations and constant provenance,
- add derived examples with conservative uncertainty bounds.

---
## References
See ../REFERENCES.md and eferences.bib (BibTeX).
