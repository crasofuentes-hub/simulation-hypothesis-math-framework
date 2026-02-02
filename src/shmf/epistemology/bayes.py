from __future__ import annotations


def bayes_update(prior: float, p_e_given_h: float, p_e_given_not_h: float) -> float:
    """Binary Bayes update with explicit likelihood terms."""
    for name, x in [
        ("prior", prior),
        ("p_e_given_h", p_e_given_h),
        ("p_e_given_not_h", p_e_given_not_h),
    ]:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"{name} must be in [0, 1].")

    denom = p_e_given_h * prior + p_e_given_not_h * (1.0 - prior)
    if denom == 0.0:
        raise ZeroDivisionError("Degenerate evidence model: denominator is zero.")
    return (p_e_given_h * prior) / denom