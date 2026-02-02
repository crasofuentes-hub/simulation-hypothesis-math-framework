from __future__ import annotations

import numpy as np


class PhysicalLimits:
    """Computational physics bounds with explicit SI units."""

    h = 6.62607015e-34  # Planck constant [J*s]
    kB = 1.380649e-23   # Boltzmann constant [J/K]
    ln2 = float(np.log(2.0))

    @staticmethod
    def margolus_levitin_ops_per_second(energy_joules: float) -> float:
        """Upper bound on orthogonal state transitions per second: ~ 2E / h."""
        if energy_joules <= 0:
            raise ValueError("energy_joules must be > 0.")
        return (2.0 * energy_joules) / PhysicalLimits.h

    @staticmethod
    def landauer_j_per_bit(T_kelvin: float) -> float:
        """Minimum energy per bit erase: k_B T ln(2)."""
        if T_kelvin <= 0:
            raise ValueError("T_kelvin must be > 0.")
        return PhysicalLimits.kB * T_kelvin * PhysicalLimits.ln2