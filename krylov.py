"""Matrix-free Krylov building blocks used by SS-INK and by the baselines.

* ``cg``               conjugate gradients to a relative residual with an iteration cap
                       (the solve for p in Algorithm 2; the inner solve of RegN);
* ``power_norm``       ||A||_2 by eight power iterations from a seeded start vector
                       (the scale rule eps = 1e-2 ||H(u_0)||_2 of Section 7.1 and the
                       constant L of RegN and PGD);
* ``smallest_eigenpair`` the minimal eigenpair by implicitly restarted Lanczos (ARPACK,
                       ``scipy.sparse.linalg.eigsh``) from a seeded start vector, warm-started
                       from the previous eigenvector when one is supplied (Section 7.1);
* ``lanczos_basis``    a short Lanczos basis with full reorthogonalization (the subspace of
                       dimension at most 12 in which ARC minimizes its model, Table 1, and the
                       Lanczos vectors of Nash's truncated Newton method);
* ``gmres_solve``      restarted GMRES (``scipy.sparse.linalg.gmres``) run for as many restart
                       cycles as needed to reach a forcing term, with a cap on the Krylov
                       iterations, returning whether the forcing term was attained (the linear
                       solver of SS-INK, Section 4.3, and of the INB baseline);
* ``modified_ldl_tridiagonal``  the Gill--Murray modified Cholesky factorization of a symmetric
                       tridiagonal matrix (Nash 1984, Section 2.3--2.4).

All start vectors come from ``numpy.random.default_rng(seed)`` with a fixed seed, so every
run is deterministic (Section 7.1, "Reproducibility").
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

__all__ = ["cg", "power_norm", "smallest_eigenpair", "lanczos_basis", "seeded_vector",
           "gmres_solve", "modified_ldl_tridiagonal"]

SEED = 0


def seeded_vector(n: int, seed: int = SEED) -> np.ndarray:
    """Deterministic unit start vector."""
    v = np.random.default_rng(seed).standard_normal(n)
    return v / np.linalg.norm(v)


def cg(matvec: Callable[[np.ndarray], np.ndarray], b: np.ndarray, rtol: float, maxiter: int,
       x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Conjugate gradients for A x = b, A symmetric.

    Stops when ||r|| <= rtol ||b|| or after ``maxiter`` iterations.  If a direction of
    non-positive curvature is met (A not positive definite), the iteration stops and returns
    the current iterate; the caller decides what to do with it.  Returns (x, iterations).
    """
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    r = b - matvec(x) if x0 is not None else b.copy()
    p = r.copy()
    rr = float(r @ r)
    bnorm = float(np.linalg.norm(b))
    it = 0
    while it < maxiter and np.sqrt(rr) > rtol * bnorm:
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            break
        a = rr / pAp
        x = x + a * p
        r = r - a * Ap
        rr_new = float(r @ r)
        p = r + (rr_new / rr) * p
        rr = rr_new
        it += 1
    return x, it


def power_norm(matvec: Callable[[np.ndarray], np.ndarray], n: int, iters: int = 8,
               seed: int = SEED) -> float:
    """Estimate ||A||_2 of a symmetric operator by ``iters`` power iterations."""
    v = seeded_vector(n, seed)
    est = 0.0
    for _ in range(iters):
        w = matvec(v)
        est = float(np.linalg.norm(w))
        if est == 0.0:
            break
        v = w / est
    return est


def smallest_eigenpair(matvec: Callable[[np.ndarray], np.ndarray], n: int,
                       v0: Optional[np.ndarray] = None, tol: float = 1e-8,
                       seed: int = SEED) -> Tuple[float, np.ndarray]:
    """Smallest eigenpair (lambda_min, v_min) of a symmetric operator by ARPACK Lanczos.

    ``v0`` is the start vector (the previous eigenvector when warm-starting); when ``None``
    the seeded vector is used.  ``tol`` is ARPACK's relative accuracy for the eigenvalue.
    For n <= 3 ARPACK cannot be used and the dense eigenproblem is solved instead.
    """
    if n <= 3:
        A = np.column_stack([matvec(np.eye(n)[:, j]) for j in range(n)])
        w, V = np.linalg.eigh(0.5 * (A + A.T))
        return float(w[0]), V[:, 0]
    if v0 is None:
        v0 = seeded_vector(n, seed)
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    try:
        w, V = spla.eigsh(op, k=1, which="SA", v0=v0, tol=tol, maxiter=50 * n, ncv=min(n, 20))
    except spla.ArpackNoConvergence as exc:    # use what ARPACK has
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            raise
        w, V = exc.eigenvalues, exc.eigenvectors
    v = V[:, 0]
    return float(w[0]), v / np.linalg.norm(v)


def lanczos_basis(matvec: Callable[[np.ndarray], np.ndarray], b: np.ndarray, m: int,
                  breakdown: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Lanczos process on a symmetric operator from ``b``, at most ``m`` vectors.

    Returns an orthonormal basis Q (n x k, k <= m) with Q[:, 0] = b/||b|| and the tridiagonal
    T = Q^T A Q (k x k), computed with full reorthogonalization.  Stops early on breakdown.
    Costs k Hessian-vector products.
    """
    n = b.size
    m = max(1, min(m, n))
    Q = np.zeros((n, m))
    alpha = np.zeros(m)
    beta = np.zeros(m)
    q = b / np.linalg.norm(b)
    k = 0
    for j in range(m):
        Q[:, j] = q
        w = matvec(q)
        alpha[j] = float(q @ w)
        w = w - alpha[j] * q - (beta[j - 1] * Q[:, j - 1] if j > 0 else 0.0)
        w = w - Q[:, : j + 1] @ (Q[:, : j + 1].T @ w)      # full reorthogonalization
        k = j + 1
        bj = float(np.linalg.norm(w))
        if j == m - 1 or bj <= breakdown * max(1.0, abs(alpha[j])):
            break
        beta[j] = bj
        q = w / bj
    T = np.diag(alpha[:k]) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
    return Q[:, :k], T


def gmres_solve(matvec: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray, rtol: float,
                restart: int, max_krylov: int) -> Tuple[np.ndarray, float, bool]:
    """Restarted GMRES for ``A x = rhs`` with ``A`` given by ``matvec``.

    GMRES(``restart``) is run for as many restart cycles as needed to bring the relative
    residual below ``rtol``, up to ``max_krylov`` Krylov iterations in total (Section 4.3).
    ``scipy.sparse.linalg.gmres`` forms the true residual at the start of every restart cycle,
    which guards against the drift of the recursively updated residual documented by
    Eisenstat and Walker (1996, Section 3.4) for finite-difference operators.

    Returns ``(x, achieved, attained)``: the iterate, the relative residual estimate at exit
    (Arnoldi estimate of the last cycle, relative to ``||rhs||``) and whether the forcing term
    was attained, i.e. SciPy reported convergence (true residual <= rtol ||rhs|| within the cap)
    and the estimate is <= rtol.  A zero right-hand side returns the zero vector.
    """
    n = rhs.size
    if not np.any(rhs):
        return np.zeros(n), 0.0, True
    restart = max(1, min(restart, n))
    cycles = max(1, int(math.ceil(max_krylov / restart)))
    op = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    hist = []
    x, info = spla.gmres(op, rhs, rtol=rtol, atol=0.0, restart=restart, maxiter=cycles,
                         callback=hist.append, callback_type="pr_norm")
    achieved = float(hist[-1]) if hist else 1.0
    return x, achieved, bool(info == 0 and achieved <= rtol)


def modified_ldl_tridiagonal(alpha: np.ndarray, beta: np.ndarray, delta: float
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gill--Murray modified Cholesky factorization ``T + E = L D L^T`` of a symmetric
    tridiagonal ``T`` with diagonal ``alpha`` (length k) and subdiagonal ``beta`` (length k-1).

    This is the factorization Nash applies to the Lanczos tridiagonal matrix (1984, Sections
    2.3--2.4): the diagonal pivots are forced to be at least ``delta`` and the growth of the
    off-diagonal elements of ``L`` is bounded, so that the diagonal modification ``E`` obeys the
    bound of his Theorem 2.  Concretely, with ``theta_j = alpha_j - l_{j-1}^2 d_{j-1}`` the pivot is
    ``d_j = max(delta, |theta_j|, beta_j^2 / beta_bar^2)`` where ``beta_bar^2`` is the
    Gill--Murray element bound ``max(gamma, xi / sqrt(k^2 - 1), eps)`` built from the largest
    diagonal ``gamma`` and off-diagonal ``xi`` moduli of ``T`` (Gill, Murray and Wright 1981,
    Section 4.4.2.2).  Returns ``(d, l, e)``: the pivots, the subdiagonal of ``L`` and the diagonal
    modification ``E``.
    """
    k = alpha.size
    d = np.zeros(k)
    l = np.zeros(max(k - 1, 0))
    e = np.zeros(k)
    gamma = float(np.max(np.abs(alpha))) if k else 0.0
    xi = float(np.max(np.abs(beta))) if k > 1 else 0.0
    beta_bar2 = max(gamma, xi / math.sqrt(max(k * k - 1, 1)), np.finfo(float).eps)
    for j in range(k):
        theta = alpha[j] - (l[j - 1] ** 2 * d[j - 1] if j > 0 else 0.0)
        off = beta[j] if j < k - 1 else 0.0
        d[j] = max(delta, abs(theta), off * off / beta_bar2)
        e[j] = d[j] - theta
        if j < k - 1:
            l[j] = off / d[j]
    return d, l, e
