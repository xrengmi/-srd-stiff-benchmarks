"""İstatistiksel testler — Wilcoxon + Benjamini-Hochberg FDR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """Eşleştirilmiş Wilcoxon test sonucu."""

    solver_a: str
    solver_b: str
    median_diff_log2: float
    statistic: float
    pvalue: float
    n_pairs: int
    effect_size: float  # Z/√N


def paired_wilcoxon(
    df: pd.DataFrame,
    solver_a: str,
    solver_b: str,
    metric: str = "nhev",
    log_transform: bool = True,
) -> WilcoxonResult:
    """
    İki solver arasında eşleştirilmiş Wilcoxon signed-rank testi.
    Sadece HER İKİSİNİN de yakınsadığı problemler karşılaştırılır.
    """
    wide = df.pivot_table(
        index="problem", columns="solver", values=metric, aggfunc="median",
    )
    wide = wide.dropna(subset=[solver_a, solver_b])

    a = wide[solver_a].values
    b = wide[solver_b].values

    if log_transform:
        valid = (a > 0) & (b > 0)
        a, b = a[valid], b[valid]
        diffs = np.log2(a) - np.log2(b)
    else:
        diffs = a - b

    n_pairs = len(diffs)
    if n_pairs < 6:
        return WilcoxonResult(
            solver_a, solver_b, float(np.median(diffs)),
            0.0, 1.0, n_pairs, 0.0,
        )

    stat, pval = stats.wilcoxon(diffs, zero_method="pratt", alternative="two-sided")
    # Rank-biserial correlation olarak etki boyutu
    z = stats.norm.isf(pval / 2) * np.sign(np.median(diffs))
    effect = float(z / np.sqrt(n_pairs))

    return WilcoxonResult(
        solver_a=solver_a, solver_b=solver_b,
        median_diff_log2=float(np.median(diffs)),
        statistic=float(stat), pvalue=float(pval),
        n_pairs=n_pairs, effect_size=effect,
    )


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR düzeltmesi.

    Returns
    -------
    Düzeltilmiş q-değerleri (orijinal sıralama).
    """
    pvals = np.asarray(pvalues)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.argsort(order) + 1

    sorted_pvals = pvals[order]
    adjusted_sorted = sorted_pvals * n / np.arange(1, n + 1)
    # Monotonizasyon
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.minimum(adjusted_sorted, 1.0)

    # Orijinal sıralamaya geri döndür
    result = np.empty(n)
    for i, idx in enumerate(order):
        result[idx] = adjusted[i]
    return result


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: callable = np.median,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Bootstrap güven aralığı.

    Returns
    -------
    (point_estimate, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        estimates[i] = stat_fn(sample)

    alpha = (1 - confidence) / 2
    point = float(stat_fn(data))
    lo = float(np.percentile(estimates, 100 * alpha))
    hi = float(np.percentile(estimates, 100 * (1 - alpha)))
    return point, lo, hi


def spearman_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank korelasyonu + p-değeri."""
    rho, pval = stats.spearmanr(x, y)
    return float(rho), float(pval)
