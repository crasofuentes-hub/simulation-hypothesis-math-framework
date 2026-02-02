from __future__ import annotations

import numpy as np


def spectral_peak_zscore(series: np.ndarray) -> float:
    """Maximum z-score of the magnitude spectrum (screening statistic)."""
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("series must be 1D with at least 8 samples.")

    spectrum = np.abs(np.fft.rfft(x - x.mean()))
    mu = spectrum.mean()
    sigma = spectrum.std(ddof=1)
    if sigma == 0.0:
        return 0.0
    return float((spectrum.max() - mu) / sigma)