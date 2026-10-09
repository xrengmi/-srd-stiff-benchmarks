"""Residual and generalized-Jacobian action of the implicit Euler step of SRD.

These two functions carry no package dependencies so that the consistency of the
operator handed to GMRES with the residual it is meant to linearize can be tested in
isolation (``tests/test_ssink_jacobian.py``).  They implement, respectively, the
left-preconditioned residual

    R_u(u, alpha) = (H(u) + alpha I)(u - u_k) + dt g(u),
    R_a(u, alpha) = (alpha - alpha_k) - dt ( K max(0, eps - lambda_min(H(u))) - gamma alpha ),

and the action of the matrix  diag(H(u) + alpha I, 1) V  on a direction (d_u, d_a), where
V is an element of the Bouligand subdifferential of

    G(u, alpha) = (u - u_k + dt (H(u) + alpha I)^{-1} g(u),
                   alpha - alpha_k - dt ( K max(0, eps - lambda_min(H(u))) - gamma alpha )).

Since R = diag(H + alpha I, 1) G, the linear system  [diag(H+alpha I,1) V] d = -R  has the same
solution as the Newton system V d = -G; the left factor only changes the norm in which GMRES
measures its residual.  The u-row of the operator is therefore NOT the Jacobian of R_u: it
differs from it by the derivative of the preconditioner acting on G_u, a term that vanishes
at the solution.  The alpha-row is the exact Jacobian row of R_a whenever the eigenpair
(lam, vec) passed in is the eigenpair of H(u) at the *current* iterate u.

Both third-derivative actions are finite differences of two Hessian-vector products.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

HVP = Callable[[np.ndarray, np.ndarray], np.ndarray]

__all__ = ["residual", "jacobian_action"]


def residual(
    u: np.ndarray,
    alpha: float,
    *,
    u_k: np.ndarray,
    alpha_k: float,
    dt: float,
    grad_u: np.ndarray,
    hvp: HVP,
    lam: float,
    K: float,
    gamma: float,
    eps: float,
) -> np.ndarray:
    """Left-preconditioned residual R(u, alpha) of the implicit Euler step.

    ``grad_u`` is the gradient at ``u`` and ``lam`` the minimum Hessian eigenvalue at ``u``
    (both evaluated at the point where the residual is requested, which is either the current
    inner iterate or a trial point of the damping test; the same function serves both, so the
    damping test compares one consistent residual at the two states).  Costs one
    Hessian-vector product, except at u = u_k (the first inner iterate), where
    H(u)(u - u_k) = 0 is known without a product.
    """
    d = u - u_k
    r_u = (hvp(u, d) if d.any() else np.zeros_like(d)) + alpha * d + dt * grad_u
    drive = K * max(0.0, eps - lam)
    r_a = (alpha - alpha_k) - dt * (drive - gamma * alpha)
    return np.concatenate([r_u, [r_a]])


def jacobian_action(
    w: np.ndarray,
    *,
    u: np.ndarray,
    alpha: float,
    dt: float,
    p: np.ndarray,
    vec: np.ndarray,
    active: bool,
    K: float,
    gamma: float,
    hvp: HVP,
    h_p: Optional[np.ndarray],
    h_v: Optional[np.ndarray],
    tau: float,
    third_u: bool = True,
    third_alpha: bool = True,
    site_hook: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Action of diag(H(u)+alpha I, 1) V on w = (d_u, d_a), V in the B-subdifferential of G.

    Parameters
    ----------
    p
        Solution of (H(u) + alpha I) p = g(u) at the current iterate.
    vec
        Unit eigenvector of H(u) for its minimum eigenvalue at the current iterate.
    active
        Whether eps - lambda_min(H(u)) > 0 at the current iterate (active branch).
    h_p, h_v
        Cached products H(u) p and H(u) vec (``None`` when the corresponding
        third-derivative action is omitted, as in SS-INK-2).
    tau
        Finite-difference length scale; the actual step is tau / ||d_u||.
    site_hook
        Optional callback receiving the label of the call site ("third" or "Hd") before each
        Hessian-vector product, for cost attribution.

    Cost: one Hessian-vector product for H d_u, plus one for each third-derivative action
    that is switched on (two on the active branch of SS-INK, none for SS-INK-2).
    """
    n = u.size
    d_u, d_a = w[:n], w[n]
    nd = float(np.linalg.norm(d_u))
    t3_u = np.zeros(n)
    t3_a = 0.0
    if nd > 0.0:
        t = tau / nd
        if site_hook is not None:
            site_hook("third")
        if third_u and h_p is not None:
            t3_u = (hvp(u + t * d_u, p) - h_p) / t          # ~ grad^3 f(u)[d_u, p, .]
        if third_alpha and active and h_v is not None:
            t3_a = float(vec @ ((hvp(u + t * d_u, vec) - h_v) / t))  # ~ grad^3 f(u)[d_u, v, v]
    if site_hook is not None:
        site_hook("Hd")
    h_d = hvp(u, d_u)
    top = h_d + alpha * d_u + dt * (h_d - t3_u - d_a * p)
    bot = (1.0 + dt * gamma) * d_a + (dt * K * t3_a if active else 0.0)
    return np.concatenate([top, [bot]])
