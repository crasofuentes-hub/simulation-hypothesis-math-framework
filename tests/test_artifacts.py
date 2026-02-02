import numpy as np

from shmf.simulation_detection.artifacts import spectral_peak_zscore


def test_zscore_runs():
    x = np.sin(np.linspace(0, 10, 256))
    z = spectral_peak_zscore(x)
    assert isinstance(z, float)
