import numpy as np

from shmf.mathematical_foundations.stochastic import StochasticBranchingModel


def test_stochastic_reproducible_mc():
    m = StochasticBranchingModel(
        base_compute_budget=1.0,
        efficiency=0.8,
        prior_kind="fixed_rate",
        lambda0=2.0,
    )
    a = m.monte_carlo(max_depth=6, n_trials=20, seed=123)
    b = m.monte_carlo(max_depth=6, n_trials=20, seed=123)
    assert np.allclose(a["N_mean"], b["N_mean"])
    assert np.allclose(a["W_mean"], b["W_mean"])


def test_stochastic_prob_dist_mc():
    m = StochasticBranchingModel(
        base_compute_budget=1.0,
        efficiency=0.75,
        prior_kind="budget_scaled",
        alpha=1.0,
        cost_scale=0.1,
    )
    p_mean, p_std = m.probability_depth_distribution_mc(max_depth=6, n_trials=30, seed=7)
    assert np.isclose(float(p_mean.sum()), 1.0, atol=1e-6)
    assert np.all(p_mean >= 0.0)
    assert np.all(p_std >= 0.0)
