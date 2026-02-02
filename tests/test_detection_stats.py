import numpy as np

from shmf.simulation_detection.stats import (
    NullModel,
    benjamini_hochberg,
    detect_spectral_artifacts,
    empirical_p_value,
    power_two_sample_normal,
    spectral_peak_zscore,
    zscore_to_p_two_sided,
)


def test_bh_fdr_basic():
    p = np.array([0.001, 0.01, 0.2, 0.9])
    rej = benjamini_hochberg(p, alpha=0.05)
    assert rej.dtype == bool
    assert rej.shape == p.shape
    assert rej[0] is True


def test_empirical_p_value_bounds():
    null = np.linspace(-1, 1, 101)
    p = empirical_p_value(0.0, null, alternative="two-sided")
    assert 0.0 < p <= 1.0


def test_z_to_p_monotone():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    p = zscore_to_p_two_sided(z)
    assert p[0] >= p[1] >= p[2] >= p[3]


def test_detect_spectral_artifacts_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=256)
    nm = NullModel(kind="phase_randomized")
    idxs = [2, 3, 5, 7, 11]
    z, p, rej = detect_spectral_artifacts(x, idxs, null_model=nm, n_null=50, alpha=0.1, seed=1)
    assert z.shape == (len(idxs),)
    assert p.shape == (len(idxs),)
    assert rej.shape == (len(idxs),)
    assert np.all((p > 0.0) & (p <= 1.0))


def test_power_increases_with_effect():
    p1 = power_two_sample_normal(effect_size=0.0, alpha=0.05, n=200, two_sided=True)
    p2 = power_two_sample_normal(effect_size=0.5, alpha=0.05, n=200, two_sided=True)
    assert p2 >= p1


def test_spectral_peak_zscore_defined():
    x = np.sin(np.linspace(0, 20 * np.pi, 256))
    z = spectral_peak_zscore(x)
    assert np.isfinite(z)
