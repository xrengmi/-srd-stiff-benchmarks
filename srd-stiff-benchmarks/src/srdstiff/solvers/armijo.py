"""Armijo geri-izleme ile adım boyutu seçimi."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from srdstiff.core.types import Vector


@dataclass(frozen=True, slots=True)
class ArmijoResult:
    """Armijo geri-izleme sonucu."""

    h: float
    f_new: float
    nfev_used: int
    converged: bool


def armijo_backtrack(
    f_fn: Callable[[Vector], float],
    x: Vector,
    p: Vector,
    g: Vector,
    f_curr: float,
    h0: float = 1.0,
    c1: float = 1e-4,
    tau: float = 0.5,
    h_min: float = 1e-10,
    max_iter: int = 60,
) -> ArmijoResult:
    """
    Armijo yeterli azalma koşulunu sağlayan en büyük adım boyutunu bul.

    Koşul: f(x + h·p) ≤ f(x) + c₁·h·⟨g, p⟩

    Parameters
    ----------
    f_fn : callable
        Amaç fonksiyonu değerlendirici.
    x : Vector
        Mevcut nokta.
    p : Vector
        İniş yönü (⟨g,p⟩ < 0 varsayılır).
    g : Vector
        Mevcut gradyan ∇f(x).
    f_curr : float
        f(x) değeri.
    h0 : float
        Başlangıç adım boyutu.
    c1 : float
        Armijo sabiti (tipik 10⁻⁴).
    tau : float
        Azaltma faktörü (tipik 0.5).
    h_min : float
        Minimum izin verilen adım.
    max_iter : int
        Maksimum geri-izleme adımı.

    Returns
    -------
    ArmijoResult

    Notes
    -----
    Lean teoremi: `armijo_satisfiable` — ⟨g,p⟩ < 0 iken yeterince
    küçük h > 0 koşulu sağlar.
    """
    assert tau > 0 and tau < 1, f"τ ∈ (0,1) gerekli, verilen: {tau}"
    assert c1 > 0 and c1 < 1, f"c₁ ∈ (0,1) gerekli, verilen: {c1}"

    gp = float(np.dot(g, p))
    if gp >= 0:
        # İniş yönü değil!
        return ArmijoResult(h=0.0, f_new=f_curr, nfev_used=0, converged=False)

    h = h0
    nfev = 0
    for _ in range(max_iter):
        x_trial = x + h * p
        f_trial = f_fn(x_trial)
        nfev += 1

        if not np.isfinite(f_trial):
            # NaN veya Inf: adımı küçült
            h *= tau
            if h < h_min:
                return ArmijoResult(h=h_min, f_new=f_curr, nfev_used=nfev, converged=False)
            continue

        # Armijo koşulu
        if f_trial <= f_curr + c1 * h * gp:
            return ArmijoResult(h=h, f_new=f_trial, nfev_used=nfev, converged=True)

        h *= tau
        if h < h_min:
            return ArmijoResult(h=h_min, f_new=f_trial, nfev_used=nfev, converged=False)

    return ArmijoResult(h=h, f_new=f_curr, nfev_used=nfev, converged=False)
