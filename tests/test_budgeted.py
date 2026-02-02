import numpy as np

from shmf.mathematical_foundations.budgeted import BudgetedBranchingModel


def test_budgeted_branching_monotone_budget():
    m = BudgetedBranchingModel(base_compute_budget=1.0, efficiency=0.5, child_cost=0.1)
    c0 = m.compute_budget_at_depth(0)
    c1 = m.compute_budget_at_depth(1)
    c2 = m.compute_budget_at_depth(2)
    assert c0 > c1 > c2


def test_budgeted_branching_constraints():
    m = BudgetedBranchingModel(base_compute_budget=1.0, efficiency=0.5, child_cost=0.2)
    b0 = m.branching_at_depth(0)
    b5 = m.branching_at_depth(5)
    assert b0 >= b5 >= 0


def test_budgeted_distribution_sums_to_one():
    m = BudgetedBranchingModel(base_compute_budget=1.0, efficiency=0.7, child_cost=0.05)
    p = m.probability_depth_distribution(max_depth=8)
    assert np.isclose(float(p.sum()), 1.0)
    assert np.all(p >= 0.0)
