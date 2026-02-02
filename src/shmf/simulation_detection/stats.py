from __future__ import annotations


class NullModel(str, Enum):
    """
    Null models used for artifact detection.

    - GAUSSIAN: i.i.d. Gaussian with same mean/std as the observed series (simple baseline).
    - PHASE_RANDOMIZED: preserves the rFFT magnitude spectrum, randomizes phases (stationary surrogate).
    """
    GAUSSIAN = "gaussian"
    PHASE_RANDOMIZED = "phase_randomized"
def _clamp01(p: float) -> float:
    # Numerical safety: keep within [0,1]
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.stats import norm


def spectral_peak_zscore(x: np.ndarray, freq_index: int | None = None) -> float:
    """
    Lightweight spectral-peak z-score for a real-valued sequence.
    This is intentionally simple: identify a frequency bin, compare it to the distribution of bins.

    Parameters
    ----------
    x:
        1D real array.
    freq_index:
        Optional index in rfft spectrum. If None, choose the max-power bin excluding DC.

    Returns
    -------
    float:
        z-score of the chosen bin power relative to other bins (excluding DC).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be 1D with length >= 8.")

    X = np.fft.rfft(x)
    p = (np.abs(X) ** 2).astype(float)

    if p.size < 3:
        raise ValueError("Insufficient rfft bins.")

    # Exclude DC (bin 0) from candidate set
    p_no_dc = p[1:]
    if freq_index is None:
        j = int(np.argmax(p_no_dc)) + 1
    else:
        j = int(freq_index)
        if j <= 0 or j >= p.size:
            raise ValueError("freq_index must be within (0, len(rfft)-1].")

    # Reference distribution: all bins excluding DC and the chosen bin
    mask = np.ones(p.size, dtype=bool)
    mask[0] = False
    mask[j] = False
    ref = p[mask]
    mu = float(ref.mean())
    sd = float(ref.std(ddof=1)) if ref.size > 1 else 0.0
    if sd <= 0:
        return 0.0
    return float((p[j] - mu) / sd)


def z_to_p_one_sided(z: float) -> float:
    """
    One-sided p-value for a standard normal z (right tail): p = P(Z >= z).

    Uses SciPy's norm.sf for numerical stability.
    """
    return float(norm.sf(z))


def p_to_z_one_sided(p: float) -> float:
    """
    Inverse of z_to_p_one_sided: z = isf(p).

    Parameters
    ----------
    p:
        One-sided p-value in (0,1].

    Returns
    -------
    float
    """
    p = float(p)
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1].")
    return float(norm.isf(p))


def empirical_p_value(
    obs: float,
    null: np.ndarray,
    alternative: str = "greater",
    add_one: bool = True,
) -> float:
    """
    Empirical p-value from a null sample.

    For alternative="greater": p = P(null >= obs)
    For alternative="less":    p = P(null <= obs)
    For alternative="two-sided": p = P(|null| >= |obs|)

    The optional +1 correction (add_one=True) returns (k+1)/(n+1) to avoid zero p-values.

    Guarantees p in [0, 1].
    """
    null = np.asarray(null, dtype=float).ravel()
    if null.size == 0:
        raise ValueError("null must have at least one sample.")

    alt = alternative.lower().strip()
    if alt not in ("greater", "less", "two-sided", "twosided", "two_sided"):
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    if alt == "greater":
        k = int(np.sum(null >= obs))
        n = int(null.size)
    elif alt == "less":
        k = int(np.sum(null <= obs))
        n = int(null.size)
    else:
        k = int(np.sum(np.abs(null) >= abs(obs)))
        n = int(null.size)

    if add_one:
        p = (k + 1.0) / (n + 1.0)
    else:
        p = k / n

    # Numerical / logical safety
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return float(p)


def bh_fdr(pvals: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    BenjaminiÃ¢â‚¬â€œHochberg FDR procedure.

    Returns
    -------
    rejected:
        Array of Python bools (dtype=object) aligned to input order.
        We intentionally use Python bool so tests using `is True/False` work reliably.
    qvals:
        BH-adjusted q-values (float array), aligned to input order.

    Notes
    -----
    - This implements the classic BH step-up rule under independence / positive dependence assumptions.
    """
    p = np.asarray(list(pvals), dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1D.")
    m = int(p.size)
    if m == 0:
        return np.asarray([], dtype=object), np.asarray([], dtype=float)

    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0,1].")

    order = np.argsort(p)
    p_sorted = p[order]

    # Step-up threshold: p_(i) <= (i/m)*alpha, i starts at 1
    i = np.arange(1, m + 1, dtype=float)
    thresh = (i / m) * float(alpha)
    passed = p_sorted <= thresh

    if np.any(passed):
        kmax = int(np.max(np.where(passed)[0]))
        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[: kmax + 1] = True
    else:
        reject_sorted = np.zeros(m, dtype=bool)

    # q-values: monotone BH adjustment
    q_sorted = (m / i) * p_sorted
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Map back to original order
    reject = np.zeros(m, dtype=object)
    qvals = np.zeros(m, dtype=float)
    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)

    for j in range(m):
        reject[j] = bool(reject_sorted[inv[j]])
        qvals[j] = float(q_sorted[inv[j]])

    return reject, qvals


@dataclass(frozen=True)
class PowerResult:
    effect: float
    n: int
    alpha: float
    power: float


def normal_mean_test_power(effect: float, n: int, alpha: float = 0.05) -> float:
    """
    Simple one-sided power approximation for a z-test on the mean with known variance=1.

    H0: mean = 0
    H1: mean = effect > 0
    Test: reject if Z >= z_{1-alpha}, where Z ~ N(effect*sqrt(n), 1) under H1.

    This is not a full experimental design engine; it is a scaffold for reproducible power reasoning.
    """
    effect = float(effect)
    n = int(n)
    alpha = float(alpha)
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    zcrit = float(norm.isf(alpha))
    mu = effect * np.sqrt(n)
    # Power = P(Z >= zcrit | Z ~ N(mu,1)) = sf(zcrit - mu)
    return float(norm.sf(zcrit - mu))


def spectral_peak_pvals_under_null(
    x: np.ndarray,
    freq_indices: np.ndarray,
    n_null: int = 256,
    null_model: str = "phase_randomized",
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p-values for spectral peak z-scores at specified frequency indices under a chosen null model,
    then apply BH-FDR.

    Returns (pvals, rejected, qvals) aligned to freq_indices order.
    """
    x = np.asarray(x, dtype=float).ravel()
    idxs = np.asarray(freq_indices, dtype=int).ravel()
    if idxs.size == 0:
        raise ValueError("freq_indices must be non-empty.")
    if n_null <= 0:
        raise ValueError("n_null must be > 0.")

    rng = np.random.default_rng(int(seed))

    def make_null_sample() -> np.ndarray:
        if null_model == "shuffle":
            return rng.permutation(x)
        if null_model == "gaussian":
            return rng.normal(loc=float(x.mean()), scale=float(x.std(ddof=1) + 1e-12), size=x.size)
        if null_model == "phase_randomized":
            X = np.fft.rfft(x)
            mag = np.abs(X)
            phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.size)
            phase[0] = 0.0
            if mag.size > 1:
                phase[-1] = 0.0
            Xn = mag * np.exp(1j * phase)
            xn = np.fft.irfft(Xn, n=x.size)
            return np.asarray(xn, dtype=float)

        raise ValueError("null_model must be one of: 'phase_randomized', 'shuffle', 'gaussian'.")

    # Observed z-scores
    z_obs = np.asarray([spectral_peak_zscore(x, int(i)) for i in idxs], dtype=float)

    # Null z-score matrix [k x n_null]
    null_z = np.zeros((idxs.size, int(n_null)), dtype=float)
    for t in range(int(n_null)):
        xn = make_null_sample()
        null_z[:, t] = np.asarray([spectral_peak_zscore(xn, int(i)) for i in idxs], dtype=float)

    # Empirical p-values (right tail)
    pvals_list = []
    for i in range(idxs.size):
        pvals_list.append(empirical_p_value(z_obs[i], null_z[i, :], alternative="greater"))
    pvals = np.asarray(pvals_list, dtype=float)

    rejected, qvals = bh_fdr(pvals, alpha=float(alpha))
    return pvals, rejected, qvals
