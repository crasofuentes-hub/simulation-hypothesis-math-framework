# Falsifiability & Observables

A simulation claim becomes scientific only when paired with a **testable prediction** and an explicit null model.
This section describes falsifiable directions and how SHMF intends to structure them.

## Observable class O1: Discretization-like artifacts

If a simulator uses coarse-graining or a lattice-like substrate, one might search for:
- anisotropies in high-energy particle distributions,
- quantization/discretization signatures in precision datasets,
- spectral peaks inconsistent with known noise models.

However, "finding patterns" is trivial unless:
- the null model is explicit,
- multiple-hypothesis testing is controlled,
- the analysis has adequate statistical power.

## Null models

A null model must specify:
- data generation or resampling process,
- noise distribution and stationarity assumptions,
- preprocessing pipeline (and its effect on spectra).

Examples:
- Gaussian white noise baseline (weak, often unrealistic),
- colored noise baseline (e.g., 1/f),
- bootstrapped surrogates preserving autocorrelation.

## Multiple comparisons

Artifact detection often involves scanning many frequencies/features. Paper-grade work must control:
- family-wise error rate (FWER), or
- false discovery rate (FDR).

SHMF plan:
- implement correction utilities (Bonferroni, Benjamini–Hochberg),
- produce reproducible reports with declared alpha levels.

## Power analysis

A meaningful "no artifact detected" statement requires power analysis:
- what effect size is detectable at given sample size?
- what is the false negative rate?

SHMF plan:
- implement simple power calculators for spectral peak detection under null distributions,
- provide guidance for minimum sample sizes.

## Connection to current SHMF code

`shmf.simulation_detection.artifacts` currently provides a lightweight spectral peak z-score utility.
Paper-grade extension requires:
- explicit null model generation,
- peak-picking definitions,
- multiple testing correction,
- power analysis scaffolding and example notebooks.

## Implemented in SHMF (statistical scaffolding)

- Null models: `shmf.simulation_detection.NullModel` (`white`, `phase_randomized`)
- Empirical p-values: `shmf.simulation_detection.empirical_p_value`
- FDR control (Benjamini–Hochberg): `shmf.simulation_detection.benjamini_hochberg`
- End-to-end helper: `shmf.simulation_detection.detect_spectral_artifacts`
- Planning scaffold: `shmf.simulation_detection.power_two_sample_normal`

