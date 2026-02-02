from shmf.physics_constraints.limits import PhysicalLimits


def test_limits_positive():
    assert PhysicalLimits.landauer_j_per_bit(300.0) > 0.0
    assert PhysicalLimits.margolus_levitin_ops_per_second(1.0) > 0.0
