"""Performans metrikleri — Dolan-Moré, Moré-Wild, agregat."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_performance_ratio(
    df: pd.DataFrame,
    metric: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Dolan-Moré performans oranı: r_{p,s} = t_{p,s} / min_s t_{p,s}.

    Parameters
    ----------
    df : DataFrame
        Sütunlar: ['problem', 'solver', metric].
    metric : str
        Karşılaştırılacak metrik (örn. 'nhev').
    """
    if group_cols is None:
        group_cols = ["problem"]

    df = df.copy()
    df["_min_per_problem"] = df.groupby(group_cols)[metric].transform("min")
    df["perf_ratio"] = df[metric] / df["_min_per_problem"].replace(0, np.nan)
    df["perf_ratio"] = df["perf_ratio"].fillna(np.inf)
    return df.drop(columns=["_min_per_problem"])


def dolan_more_profile(
    df: pd.DataFrame,
    tau_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Dolan-Moré performans profili.

    ρ_s(τ) = (1/n_P) · |{p : r_{p,s} ≤ τ}|
    """
    if tau_grid is None:
        tau_grid = np.logspace(0, 5, 100, base=2.0)

    solvers = df["solver"].unique()
    n_problems = df["problem"].nunique()

    records = []
    for solver in solvers:
        sub = df[df["solver"] == solver]
        for tau in tau_grid:
            rho = (sub["perf_ratio"] <= tau).sum() / n_problems
            records.append({"solver": solver, "tau": tau, "rho": rho})

    return pd.DataFrame(records)


def more_wild_data_profile(
    df: pd.DataFrame,
    alpha_grid: np.ndarray | None = None,
    cost_col: str = "nhev",
    size_col: str = "n",
) -> pd.DataFrame:
    """Moré-Wild veri profili: büyük ölçekli problemler için uygun."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0, 100, 100)

    solvers = df["solver"].unique()
    n_problems = df["problem"].nunique()

    df = df.copy()
    df["normalized_cost"] = df[cost_col] / (df[size_col] + 1)

    records = []
    for solver in solvers:
        sub = df[df["solver"] == solver]
        for alpha in alpha_grid:
            d = (sub["normalized_cost"] <= alpha).sum() / n_problems
            records.append({"solver": solver, "alpha": alpha, "d": d})

    return pd.DataFrame(records)


def geometric_mean(x: np.ndarray) -> float:
    """Sıfır-güvenli geometrik ortalama."""
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(x))))


def success_rate(df: pd.DataFrame, success_col: str = "converged") -> pd.Series:
    """Solver başına başarı oranı."""
    return df.groupby("solver")[success_col].mean()


def budget_restricted_performance(
    df: pd.DataFrame,
    budgets: list[int],
    cost_col: str = "nhev",
    quality_col: str = "grad_norm_final",
) -> pd.DataFrame:
    """
    Hakem Metrik R4: sabit bütçe ile en iyi ‖∇f‖ elde edilen.

    Returns
    -------
    DataFrame columns: [solver, budget, median_quality, 95ci_lo, 95ci_hi]
    """
    records = []
    for budget in budgets:
        for solver in df["solver"].unique():
            sub = df[(df["solver"] == solver) & (df[cost_col] <= budget)]
            if len(sub) == 0:
                continue
            q = sub[quality_col].values
            med = float(np.median(q))
            lo = float(np.percentile(q, 2.5))
            hi = float(np.percentile(q, 97.5))
            records.append({
                "solver": solver, "budget": budget,
                "median_quality": med, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(records)
