from shmf.mathematical_foundations.nested import NestedSimulationModel


def test_total_budget_special_case_r_equals_one():
    m = NestedSimulationModel(base_compute_budget=2.0, efficiency=0.5, branching_factor=2)
    assert m.total_budget_up_to_depth(3) == 8.0
