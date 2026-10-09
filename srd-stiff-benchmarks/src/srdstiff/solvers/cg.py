"""Önkoşullanmış Eşlenik Gradyan (CG) çözücü — matris-serbest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srdstiff.core.types import LinearOperator, Vector


@dataclass(frozen=True, slots=True)
class CGResult:
    """CG çözüm sonucu."""

    p: Vector
    residual_norm: float
    relative_residual: float
    iterations: int
    converged: bool
    curvature_min: float


def conjugate_gradient(
    M: LinearOperator,
    b: Vector,
    tol: float = 1e-1,
    max_iter: int = 1000,
    x0: Vector | None = None,
) -> CGResult:
    """
    M·p = b sistemini CG ile çöz (M simetrik pozitif tanımlı).

    Parameters
    ----------
    M : LinearOperator
        SPD matris operatörü.
    b : Vector
        Sağ taraf.
    tol : float
        Göreli artık toleransı: ‖r_k‖ / ‖b‖ ≤ tol.
    max_iter : int
        Maksimum iterasyon sayısı.
    x0 : Vector | None
        Başlangıç tahmini (varsayılan: 0).

    Returns
    -------
    CGResult
        Çözüm, artık ve meta-veriler.

    Notes
    -----
    Lean teoremleri:
      - cg_iteration_bound: j iterasyonunda yakınsama garantisi
      - cg_rate_lt_one: κ > 1 iken daralma oranı < 1
    Formel ispat: `conditionNumber` fonksiyonu, κ'yı tanımlar.
    """
    n = b.shape[0]
    assert M.n == n, f"Operator boyutu {M.n} ≠ b boyutu {n}"

    p = np.zeros(n) if x0 is None else x0.copy()
    r = b - M.apply(p) if x0 is not None else b.copy()
    d = r.copy()
    rho = float(np.dot(r, r))
    b_norm = float(np.linalg.norm(b))

    if b_norm < 1e-300:
        return CGResult(p=p, residual_norm=0.0, relative_residual=0.0,
                        iterations=0, converged=True, curvature_min=np.inf)

    tolerance = tol * b_norm
    curvature_min = np.inf

    for j in range(max_iter):
        q = M.apply(d)
        curvature = float(np.dot(d, q))
        curvature_min = min(curvature_min, curvature)

        if curvature <= 1e-14:
            # Negative or zero curvature: SPD assumption violated
            return CGResult(
                p=p, residual_norm=float(np.linalg.norm(r)),
                relative_residual=float(np.linalg.norm(r) / b_norm),
                iterations=j, converged=False, curvature_min=curvature_min,
            )

        omega = rho / curvature
        p = p + omega * d
        r = r - omega * q
        r_norm = float(np.linalg.norm(r))

        if r_norm <= tolerance:
            return CGResult(
                p=p, residual_norm=r_norm,
                relative_residual=r_norm / b_norm,
                iterations=j + 1, converged=True,
                curvature_min=curvature_min,
            )

        rho_new = float(np.dot(r, r))
        beta = rho_new / rho
        d = r + beta * d
        rho = rho_new

    r_norm = float(np.linalg.norm(r))
    return CGResult(
        p=p, residual_norm=r_norm, relative_residual=r_norm / b_norm,
        iterations=max_iter, converged=False, curvature_min=curvature_min,
    )
