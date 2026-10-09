"""Semismooth inexact Newton-Krylov solver for Spectral Regularization Dynamics (SS-INK).

The flow is

    u' = -(H(u) + alpha I)^{-1} grad f(u),
    alpha' = K max(0, eps - lambda_min(H(u))) - gamma alpha,

integrated by implicit Euler with pseudo-time step ``dt``.  Each step requires the solution
of the nonsmooth system

    G(y) = y - y_k - dt F(y) = 0,   y = (u, alpha),

by a semismooth Newton iteration whose linear subproblems are solved matrix-free by GMRES.
This module realizes Algorithms 1-3 of the paper (Sections 3-4) as stated:

* Fully implicit alpha-row.  The activation term K max(0, eps - lambda_min(H(u))) is
  evaluated at the *current inner iterate* u^(s), and again at every trial point of the
  damping test; the eigenpair is obtained by warm-started Lanczos (``smallest_eigenpair``)
  at each of these points and the eigenpair of an accepted trial point is kept.
* Left-preconditioned u-row (Section 4.2).  The residual formed is R = diag(H + alpha I, 1) G
  and the operator handed to GMRES is diag(H + alpha I, 1) V, V in the B-subdifferential of G
  (``ssink_core``); its alpha-row carries the third-derivative action dt K chi grad^3 f[d_u,v,v]
  and its u-row the action dt grad^3 f[d_u, p, .], both finite differences of two
  Hessian-vector products with the unperturbed product cached once per Newton iteration.
* Forcing terms (Section 4.3).  Eisenstat-Walker Choice 2 with the lower safeguard and the
  cap eta_max, eqs. (ew_raw)-(ew_upper); eta_0 = eta_max; eta_{s-1} is the forcing term
  actually used at iteration s-1, after any retry.  GMRES(restart) is run for as many
  restart cycles as needed to reach eta_s, up to ``krylov_max`` Krylov iterations in total;
  whether eta_s was reached and the achieved relative residual are recorded.
* Damping (Section 4.3).  Residual-decrease test ||R(y_t)|| <= (1 - zeta/2) ||R(y^(s))|| with
  the trial state y_t = (u + zeta d_u, max{0, alpha + zeta d_a}) and ONE consistent residual
  R of eq. (preconditioned_residual) evaluated at both states, i.e. with H, the gradient and
  lambda_min all taken at the trial point (eq. (trial_residual)); zeta is halved down to
  zeta_min.  The residual of the accepted trial point is the residual of the next inner
  iterate and is not recomputed.  If the test fails for every admissible zeta the forcing
  term is divided by four and the linear system re-solved, up to ``retry_max`` solves; then
  the step is rejected.  The forcing term entering the next Eisenstat-Walker safeguard is
  the one after that tightening and before damping.
* Inner stopping rule (inner_stop).  ||R(y^(s))|| <= inner_tol max{1, dt ||grad f(u^(s))||}.
  If it is not met within ``s_max`` inner iterations the step is rejected (dt is quartered),
  exactly as a failed damping test is.
* Outer loop (Algorithm 3).  Acceptance by decrease of f with roundoff slack
  sigma_f = 1e2 eps_mach max{1, |f_prev|}; pseudo-transient growth of dt by
  min{dt_growth_cap, max{1, g_prev/g_new}}; alpha projected onto the safe set
  D = {lambda_min(H(u)) + alpha >= eps - delta} at the start of every outer step.
* Initial regularization.  alpha_0 = (K/gamma) max(0, eps - lambda_min(H(u_0))) with the
  eigenpair from a seeded Lanczos start; eps = eps_factor ||H(u_0)||_2 by eight power
  iterations; delta = delta_factor eps.

Everything is deterministic given the seeded start vectors of ``srdssink.krylov``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from .krylov import cg, gmres_solve, power_norm, smallest_eigenpair
from .oracle import BudgetExceeded, Oracle, Problem
from .result import Recorder, Result
from .ssink_core import jacobian_action, residual

__all__ = ["ssink"]

_EPS_MACH = float(np.finfo(float).eps)


def _spectral_scale(oracle: Oracle, x: np.ndarray) -> float:
    """||H(x)||_2 by eight power iterations (Section 7.1); floored to avoid eps = 0."""
    return max(1e-12, power_norm(lambda v: oracle.hvp(x, v), x.size, iters=8))


def _forcing(res_norm: float, res_prev: Optional[float], eta_prev: float, eta_max: float,
             ew_gamma: float, ew_alpha: float) -> float:
    """Two-sided Eisenstat-Walker rule, eqs. (ew_raw)-(ew_upper) of the paper.

    Choice 2 is eta_s^raw = ew_gamma (||R_s|| / ||R_{s-1}||)^ew_alpha (Eisenstat and Walker,
    SIAM J. Sci. Comput. 17(1):16-32, 1996, eq. (2.6)).  Their lower safeguard (Section 2.1)
    replaces eta_s by max{eta_s, ew_gamma eta_{s-1}^ew_alpha} whenever
    ew_gamma eta_{s-1}^ew_alpha > 0.1; the cap eta_s <= eta_max is their Section 3.1
    requirement.  The pair (1/2, 2) is the paper's choice (Section 4.3).
    """
    if res_prev is None or res_prev <= 0.0:
        return eta_max
    raw = ew_gamma * (res_norm / res_prev) ** ew_alpha
    lower = ew_gamma * eta_prev ** ew_alpha
    if lower > 0.1:
        raw = max(raw, lower)
    return float(min(eta_max, raw))


def ssink(
    oracle: Oracle,
    problem: Problem,
    x0: np.ndarray,
    tol_rel: float = 1e-6,
    *,
    K: float = 2.0,
    gamma: float = 1.0,
    eps: Optional[float] = None,
    eps_factor: float = 1e-2,
    delta_factor: float = 0.5,
    dt0: float = 1.0,
    dt_min: float = 1e-12,
    dt_growth_cap: float = 5.0,
    eta_max: float = 0.9,
    ew_gamma: float = 0.5,
    ew_alpha: float = 2.0,
    s_max: int = 20,
    inner_tol: float = 1e-10,
    gmres_restart: int = 20,
    krylov_max: int = 200,
    p_rtol: float = 1e-2,
    p_maxiter: int = 50,
    zeta_min: float = 1e-3,
    retry_max: int = 4,
    max_outer: int = 10**6,
    lanczos_tol: float = 1e-8,
    third_u: bool = True,
    third_alpha: bool = True,
    inner_rtol: Optional[float] = None,
    gradient_test: bool = True,
    on_step: Optional[Callable[[Dict], None]] = None,
) -> Result:
    """Run SS-INK (``third_u=third_alpha=True``) or SS-INK-2 (both ``False``).

    Parameters (defaults are the values of Section 7.1 of the paper)
    ----------
    eps
        Activation threshold.  If ``None`` it is set to ``eps_factor * ||H(x0)||_2`` (eight
        power iterations).  The buffer is ``delta = delta_factor * eps``.
    eta_max, ew_gamma, ew_alpha
        Forcing-term cap and Eisenstat-Walker Choice 2 parameters (see ``_forcing``).
    gmres_restart, krylov_max
        GMRES restart length m and the cap m_max on Krylov iterations per linear solve.
    p_rtol, p_maxiter
        Relative residual tau_p and iteration cap of the conjugate-gradient solve for p.
    inner_tol
        epsilon_tol of the inner stopping rule (inner_stop).
    zeta_min, retry_max, s_max
        Damping floor, retry cap r_max and inner iteration cap of Algorithm 2.
    lanczos_tol
        Relative tolerance of the ARPACK eigensolver.
    inner_rtol
        If given, the inner iteration also stops once ||R_s|| <= inner_rtol ||R_0|| (the
        truncated inner solve of Section 7.3).
    gradient_test
        If ``False`` the stopping rule is disabled (saddle-escape experiments, Section 7.6).
    on_step
        Called after every accepted outer step with a dictionary of the controller state.
    """
    rec = Recorder(oracle, problem, "SS-INK" if (third_u and third_alpha) else "SS-INK-2",
                   tol_rel, gradient_test=gradient_test)
    u = np.array(x0, dtype=float)
    n = u.size
    forcing_missed = 0
    linear_solves = 0
    lanczos_calls = 0
    p_zero = 0
    eta_achieved_all = []

    def eigpair(x: np.ndarray, v0):
        nonlocal lanczos_calls
        oracle.site = "lanczos"
        lanczos_calls += 1
        return smallest_eigenpair(lambda w: oracle.hvp(x, w), n, v0=v0, tol=lanczos_tol)

    def stats(status):
        return rec.finish(status, forcing_missed=forcing_missed, linear_solves=linear_solves,
                          lanczos_calls=lanczos_calls, eta_achieved=eta_achieved_all,
                          p_zero=p_zero, dt=dt, alpha=alpha, eps=eps)

    dt, alpha = dt0, 0.0
    try:
        f_prev = oracle.f(u)                                   # Algorithm 3, line 2
        g = rec.start(u, f_prev)
        if eps is None:
            oracle.site = "scale"
            eps = eps_factor * _spectral_scale(oracle, u)
        delta = delta_factor * eps

        lam, vec = eigpair(u, None)                            # Algorithm 3, line 1 (seeded start)
        alpha = (K / gamma) * max(0.0, eps - lam)              # quasi-equilibrium of the alpha-dynamics
        g_prev = float(np.linalg.norm(g))
        eta_last = eta_max
        eta_achieved = 1.0

        for _ in range(max_outer):
            if rec.converged(g):
                return stats("converged")

            # (lam, vec) is the Lanczos eigenpair at the current u: computed at the trial
            # point when the previous step was accepted, or restored with u_k after a
            # rejection (Algorithm 3, line 4).  Projection onto D (Algorithm 3, line 5).
            alpha = max(alpha, eps - delta - lam)
            u_k, alpha_k, lam_k, vec_k = u.copy(), alpha, lam, vec
            grad = g                                           # gradient at u^(0) = u_k

            ok = True
            inner_converged = False
            eta_prev, res_prev = eta_max, None
            res_first = None
            s = 0
            # residual at y^(0) = y_k: H(u_k)(u_k - u_k) = 0 is known without a product
            oracle.site = "residual"
            res = residual(u, alpha, u_k=u_k, alpha_k=alpha_k, dt=dt, grad_u=grad,
                           hvp=oracle.hvp, lam=lam, K=K, gamma=gamma, eps=eps)
            while True:
                res_norm = float(np.linalg.norm(res))
                if res_first is None:
                    res_first = res_norm
                if res_norm <= inner_tol * max(1.0, dt * float(np.linalg.norm(grad))):
                    inner_converged = True                     # eq. (inner_stop)
                    break
                if inner_rtol is not None and s > 0 and res_norm <= inner_rtol * res_first:
                    inner_converged = True                     # truncated inner solve, Section 7.3
                    break
                if s >= s_max:                                 # Algorithm 2: test at y^(s_max) too
                    break

                active = (eps - lam) > 0.0
                oracle.site = "p-solve"
                p, p_iters = cg(lambda w: oracle.hvp(u, w) + alpha * w, grad, p_rtol, min(p_maxiter, n))
                if p_iters == 0 and float(np.linalg.norm(grad)) > 0.0:
                    p_zero += 1        # non-positive curvature on the first CG direction: p = 0
                tau = math.sqrt(_EPS_MACH) * (1.0 + float(np.linalg.norm(u)))
                oracle.site = "third"
                h_p = oracle.hvp(u, p) if third_u else None                     # c_p
                h_v = oracle.hvp(u, vec) if (third_alpha and active) else None  # c_v

                def matvec(w: np.ndarray, _u=u, _alpha=alpha, _p=p, _vec=vec, _active=active,
                           _h_p=h_p, _h_v=h_v, _tau=tau) -> np.ndarray:
                    return jacobian_action(w, u=_u, alpha=_alpha, dt=dt, p=_p, vec=_vec,
                                           active=_active, K=K, gamma=gamma, hvp=oracle.hvp,
                                           h_p=_h_p, h_v=_h_v, tau=_tau,
                                           third_u=third_u, third_alpha=third_alpha,
                                           site_hook=lambda label: setattr(oracle, "site", label))

                eta = _forcing(res_norm, res_prev, eta_prev, eta_max, ew_gamma, ew_alpha)
                res_prev = res_norm

                accepted, zeta, d_u, d_a = False, 1.0, None, 0.0
                lam_try, vec_try, grad_try, res_try = lam, vec, grad, res
                for _retry in range(retry_max):
                    step, eta_achieved, attained = gmres_solve(matvec, -res, eta, gmres_restart, krylov_max)
                    linear_solves += 1
                    eta_achieved_all.append((eta, eta_achieved, int(not attained)))
                    if not attained:
                        forcing_missed += 1
                    eta_last = eta
                    d_u, d_a = step[:n], step[n]
                    zeta = 1.0
                    while zeta >= zeta_min:
                        u_try = u + zeta * d_u
                        a_try = max(0.0, alpha + zeta * d_a)          # alpha clipped at 0
                        lam_try, vec_try = eigpair(u_try, vec)      # fully implicit alpha-row
                        oracle.site = "damping"
                        grad_try = oracle.g(u_try)
                        res_try = residual(u_try, a_try, u_k=u_k, alpha_k=alpha_k, dt=dt,
                                           grad_u=grad_try, hvp=oracle.hvp, lam=lam_try,
                                           K=K, gamma=gamma, eps=eps)   # eq. (trial_residual)
                        if float(np.linalg.norm(res_try)) <= (1.0 - 0.5 * zeta) * res_norm:
                            accepted = True
                            break
                        zeta *= 0.5
                    if accepted:
                        break
                    eta *= 0.25                                    # retry with a tighter forcing term
                eta_prev = eta          # the forcing term actually used enters the next safeguard
                if not accepted:
                    ok = False
                    break
                u, alpha = u_try, a_try
                lam, vec, grad, res = lam_try, vec_try, grad_try, res_try   # reused: eigenpair, gradient, residual
                s += 1

            if ok and not inner_converged:
                ok = False      # unconverged inner solve: rejected like a damping failure

            f_new = oracle.f(u) if ok else f_prev
            # the flow decreases f, not ||grad f||; the slack is needed because near a
            # saddle df/dt falls below machine precision and a strict test rejects every step
            slack = 1e2 * _EPS_MACH * max(1.0, abs(f_prev))
            if not ok or f_new > f_prev + slack:
                u, alpha, lam, vec = u_k, alpha_k, lam_k, vec_k   # g is still the gradient at u_k
                dt *= 0.25
                if dt < dt_min:
                    return stats("stalled")
                continue

            g = grad                                           # gradient at the accepted state
            g_norm = float(np.linalg.norm(g))
            rec.log(u, g, f_new)
            # pseudo-transient continuation; never shrink on an accepted step, since
            # ||grad f|| grows while a saddle is being escaped
            dt *= float(np.clip(g_prev / max(g_norm, 1e-300), 1.0, dt_growth_cap))
            f_prev, g_prev = f_new, g_norm
            if on_step is not None:
                on_step({"outer": rec.outer, "f": f_new, "grad_norm": g_norm,
                         "lambda_min": lam, "alpha": alpha, "dt": dt,
                         "inner": s, "cost": oracle.cost, "eta": eta_last,
                         "eta_achieved": eta_achieved, "forcing_missed": forcing_missed,
                         "lanczos_calls": lanczos_calls})

        return stats("iteration cap")
    except BudgetExceeded:
        return stats(oracle.exhausted or "budget")
