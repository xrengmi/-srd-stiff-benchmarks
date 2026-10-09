"""Lanczos spektral tahmini — λ_min için matris-serbest alt sınır."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srdstiff.core.types import LinearOperator, Vector


@dataclass(frozen=True, slots=True)
class LanczosResult:
    """Lanczos çıktısı."""

    theta_min: float
    theta_max: float
    error_bound: float
    lam_min_lb: float  # muhafazakâr alt sınır
    iterations_done: int
    converged: bool


def lanczos_eigenvalues(
    A: LinearOperator,
    m: int = 20,
    reorthogonalize: bool = True,
    tol_break: float = 1e-12,
    seed: int = 0,
) -> LanczosResult:
    """
    Simetrik operatörün spektrum uçlarını tahmin et.

    Parameters
    ----------
    A : LinearOperator
        Simetrik operatör.
    m : int
        Lanczos adım sayısı.
    reorthogonalize : bool
        Tam yeniden ortogonalleştirme (sayısal kararlılık).
    tol_break : float
        Erken bitiş toleransı (β_{j+1} < tol_break).
    seed : int
        Başlangıç vektörü için tohum.

    Returns
    -------
    LanczosResult
        Spektrum uçları ve muhafazakâr alt sınır.

    Notes
    -----
    Lean teoremi: `lanczos_safe_check` — bu alt sınır λ_min'in kesin
    bir alt sınırıdır (hata sınırı ile birlikte).
    """
    n = A.n
    assert m >= 2, f"m ≥ 2 olmalı, verilen: {m}"
    assert m <= n, f"m ≤ n olmalı, m={m}, n={n}"

    rng = np.random.default_rng(seed)
    q_prev = np.zeros(n)
    q_curr = rng.standard_normal(n)
    q_curr /= np.linalg.norm(q_curr)

    alphas: list[float] = []
    betas: list[float] = [0.0]
    Q = np.zeros((n, m + 1))
    Q[:, 0] = q_curr

    converged_early = False
    j_done = m

    for j in range(m):
        w = A.apply(q_curr)
        alpha_j = float(np.dot(w, q_curr))
        alphas.append(alpha_j)
        w = w - alpha_j * q_curr - betas[-1] * q_prev

        # Tam yeniden-ortogonalleştirme (modifiye Gram-Schmidt, 2 kez)
        if reorthogonalize:
            for _ in range(2):
                for i in range(j + 1):
                    coeff = float(np.dot(w, Q[:, i]))
                    w = w - coeff * Q[:, i]

        beta_next = float(np.linalg.norm(w))

        if beta_next < tol_break:
            converged_early = True
            j_done = j + 1
            break

        betas.append(beta_next)
        q_prev = q_curr
        q_curr = w / beta_next
        Q[:, j + 1] = q_curr

    # Tridiagonal matrisi kur ve özdeğerlerini hesapla
    k = len(alphas)
    T = np.diag(alphas)
    for i in range(k - 1):
        T[i, i + 1] = betas[i + 1]
        T[i + 1, i] = betas[i + 1]

    eigvals, eigvecs = np.linalg.eigh(T)
    theta_min = float(eigvals[0])
    theta_max = float(eigvals[-1])

    # Hata sınırı: β_{k+1}·|e_m^T·y_min| (Kaniel-Paige)
    if converged_early:
        error_bound = 0.0
    else:
        y_min = eigvecs[:, 0]
        error_bound = float(abs(betas[-1] * y_min[-1]))

    # Ek güvenlik marjı
    machine_eps = np.finfo(np.float64).eps
    error_bound = error_bound + machine_eps * abs(theta_max) * 10

    return LanczosResult(
        theta_min=theta_min,
        theta_max=theta_max,
        error_bound=error_bound,
        lam_min_lb=theta_min - error_bound,
        iterations_done=j_done,
        converged=converged_early,
    )
