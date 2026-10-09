"""The nine matrix-free baselines of Table 1 of the paper (Section 7.1).

All share the oracle, the cost model, the budget and the stopping rule of SS-INK.  The
parameters fixed by Table 1 are the defaults below; the implementation details the table
does not fix are stated in the paragraph after Table 1 and are, for reference:

* Armijo backtracking (NCG, RegN, L-BFGS): c = 1e-4, the trial step is halved until the
  Armijo condition holds, at most 40 halvings; the initial trial step is 1 for RegN and
  L-BFGS (Newton-type directions), except on the first L-BFGS iteration where it is
  1/||grad f||, and for NCG it is the rule of Nocedal and Wright, p. 59 (the display
  preceding their eq. (3.60)):
  alpha_0 = alpha_{k-1} g_{k-1}^T d_{k-1} / (g_k^T d_k), with 1/||grad f(x_0)|| at k = 0.
  A direction that is not a descent direction is replaced by the negative gradient
  (L-BFGS also clears its memory).
* TR-NCG: the Steihaug-Toint iteration stops at the trust-region boundary, at negative
  curvature, after 50 iterations, or when the residual falls below
  min(1/2, sqrt(||g||) ) ||g|| (Nocedal and Wright, Algorithm 7.1, used by Algorithm 7.2).
* RegN: the conjugate-gradient solve for (H + lambda_k I) d = -g is capped at 200 iterations
  and stops at negative curvature with the current iterate.
* ARC: the cubic model is minimized exactly on the Lanczos subspace (Cartis, Gould and Toint,
  Theorem 3.1: (T + lambda I) z = -||g|| e_1, lambda = sigma ||z||, T + lambda I positive
  semidefinite), the scalar secular equation being solved by bisection to relative
  accuracy 1e-10; the subspace is built by the Lanczos process from the gradient with
  full reorthogonalization, stopping early only on breakdown; sigma is floored at 1e-16.
* Adam: epsilon = 1e-8 in the denominator, bias-corrected moments.
* PGD: r = 1e-2, g_thres = 1e-2 max{1, ||grad f(x_0)||}, t_thres = 10; the perturbation is
  drawn uniformly from the ball of radius r with a seeded generator; L is the eight-step
  power-iteration estimate of ||grad^2 f(x_0)||_2 (the same as the scale rule of SS-INK).
* TN (Nash, SIAM J. Numer. Anal. 21 (1984) 770-788): truncated Newton via the modified
  Lanczos method.  The search direction minimizes the quadratic model over the Lanczos
  subspace started from -g with the tridiagonal matrix T_q replaced by T_q + E_q, E_q the
  Gill-Murray modified Cholesky modification (Sections 2.3-2.4 there; ``modified_ldl_tridiagonal``),
  so it is a descent direction for every q (his Theorem 3); the Lanczos iteration is truncated
  by the Dembo-Steihaug rule ||r_q|| <= min{1/k, ||g||} ||g|| (his Section 4) and capped at n/2
  iterations, the residual norm being the Lanczos estimate (||E_q y||^2 + (beta_{q+1} y_q)^2)^{1/2};
  at a stationary point (gradient within the protocol's tolerance) the direction of negative
  curvature of his Section 2.5, p = V_q L_q^{-T} e_q, is used when the modified factorization
  detected indefiniteness.  Nash preconditions the Lanczos process by a two-step limited-memory
  quasi-Newton update with a diagonal scaling (his Section 3.2, Nash 1982) and uses the
  Gill-Murray line search with accuracies 0.25, 0.1 and 0.001; here the process is
  unpreconditioned and the line search is the shared Armijo backtracking from the step
  min{1, Delta/||p||} with his bound Delta = 10 on the change of x per iteration, so that the
  direction and the step limit are Nash's and the acceptance rule is the protocol's.
* INB (Eisenstat and Walker, SIAM J. Optim. 4 (1994) 393-422, Algorithm INB; Pawlowski,
  Shadid, Simonis and Walker, SIAM Rev. 48 (2006) 700-721, Algorithm INB-Q): inexact Newton
  backtracking applied to the gradient system F(x) = grad f(x) = 0, with the Newton equation
  H s = -F solved by GMRES(20) (at most 200 Krylov iterations, as for SS-INK) to the forcing
  term eta_k of Choice 1 in the form (2.2) of Eisenstat and Walker (1996) with its safeguard
  eta_k <- max{eta_k, eta_{k-1}^{(1+sqrt 5)/2}} whenever eta_{k-1}^{(1+sqrt 5)/2} > 0.1,
  eta_0 = 0.01 and eta_max = 0.9 (Pawlowski et al., Section 2.1); acceptance test
  ||F(x + s)|| <= [1 - t (1 - eta)] ||F(x)|| with t = 1e-4; step-length reduction factor
  theta in [1/10, 1/2] minimizing the quadratic p(theta) with p(0) = ||F||^2/2,
  p'(0) = F^T H s and p(1) = ||F(x+s)||^2/2 (INB-Q), with eta <- 1 - theta (1 - eta), at most
  ten reductions per step (Eisenstat and Walker 1996, Section 3.1).  The linear residual
  F + H s of the accepted step, needed by Choice 1, is formed with one Hessian-vector product.
  INB converges to zeros of grad f without regard to the curvature there; it is the
  Newton-Krylov method whose attraction to saddles motivates the regularization (Section 1).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .krylov import cg, gmres_solve, lanczos_basis, modified_ldl_tridiagonal, power_norm
from .oracle import BudgetExceeded, Oracle, Problem
from .result import Recorder, Result

__all__ = ["arc", "trncg", "ncg", "regn", "lbfgs", "adam", "pgd", "tn", "inb", "BASELINES"]

ARMIJO_C = 1e-4
ARMIJO_MAX_HALVINGS = 40


def _finish(rec: Recorder, oracle: Oracle, status: str, **extra) -> Result:
    return rec.finish(status, **extra)


def _armijo(oracle: Oracle, x: np.ndarray, f: float, g: np.ndarray, d: np.ndarray,
            alpha0: float):
    """Backtracking line search; returns (alpha, f_new) or (None, None) if it fails."""
    gd = float(g @ d)
    alpha = alpha0
    for _ in range(ARMIJO_MAX_HALVINGS + 1):
        f_new = oracle.f(x + alpha * d)
        if f_new <= f + ARMIJO_C * alpha * gd:
            return alpha, f_new
        alpha *= 0.5
    return None, None


# ------------------------------------------------------------------------------- ARC ------


def _cubic_subspace(T: np.ndarray, gnorm: float, sigma: float):
    """Global minimizer of gnorm e1^T z + z^T T z / 2 + sigma ||z||^3 / 3 (Cartis et al., Thm 3.1).

    With T = U diag(lam) U^T and gh = U^T (gnorm e_1), the minimizer is z(lmb) = -(T + lmb I)^{-1}
    gnorm e_1 with lmb = sigma ||z(lmb)|| and T + lmb I positive semidefinite.  Writing
    phi(lmb) = ||z(lmb)|| - lmb / sigma, phi is decreasing on (max{0, -lam_1}, inf).  Easy case:
    phi(max{0, -lam_1}^+) > 0 and the root is found by bisection to relative accuracy 1e-10.
    Hard case (lam_1 < 0, gh has no component in the eigenspace of lam_1 and the pseudo-inverse
    solution zp at lmb = -lam_1 satisfies ||zp|| < -lam_1 / sigma): z = zp + t u_1 with t >= 0
    chosen so that ||z|| = -lam_1 / sigma.
    """
    lam_T, U = np.linalg.eigh(T)
    gh = U[0, :] * gnorm                       # U^T (gnorm e_1)
    lo = max(0.0, -lam_T[0])
    tiny = 1e-14 * max(1.0, abs(lam_T[0]))

    def z_of(lmb):
        return -gh / (lam_T + lmb)

    def phi(lmb):                              # ||z(lmb)|| - lmb/sigma, decreasing in lmb
        return float(np.linalg.norm(z_of(lmb))) - lmb / sigma

    if lam_T[0] < 0.0:
        lmb0 = -lam_T[0]
        null = np.abs(lam_T + lmb0) <= tiny    # eigenspace of lam_1 (index 0, possibly more)
        if np.all(np.abs(gh[null]) <= 1e-14 * max(1.0, gnorm)):
            zp = np.zeros_like(gh)
            zp[~null] = -gh[~null] / (lam_T[~null] + lmb0)
            target = lmb0 / sigma
            zpn = float(np.linalg.norm(zp))
            if zpn < target:                   # hard case
                t = np.sqrt(target * target - zpn * zpn)
                e1 = np.zeros_like(gh)
                e1[0] = 1.0
                return U @ (zp + t * e1), lmb0
            # otherwise phi(lmb0) >= 0 with gh orthogonal to the null space: easy case below,
            # evaluated with the null components removed
            gh = np.where(null, 0.0, gh)
    a = lo + tiny
    if phi(a) <= 0.0:                          # only possible when lo = 0 and g = 0
        return U @ z_of(a), a
    b = max(2.0 * a, 1.0)
    while phi(b) > 0.0:
        b *= 2.0
    for _ in range(200):
        mid = 0.5 * (a + b)
        if phi(mid) > 0.0:
            a = mid
        else:
            b = mid
        if b - a <= 1e-10 * max(1.0, b):
            break
    lmb = 0.5 * (a + b)
    return U @ z_of(lmb), lmb


def arc(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
        sigma0: float = 1.0, subspace: int = 12, accept: float = 0.1, sigma_min: float = 1e-16,
        gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Adaptive cubic regularization on a Lanczos subspace (Table 1)."""
    rec = Recorder(oracle, problem, "ARC", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    sigma = sigma0
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        basis = None
        for _ in range(max_iter):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            gnorm = float(np.linalg.norm(g))
            if basis is None:                  # the subspace depends on (x, g) only: reused
                oracle.site = "arc-lanczos"    # after a rejection, as in Cartis et al.
                basis = lanczos_basis(lambda v: oracle.hvp(x, v), g, subspace)
            Q, T = basis
            z, _ = _cubic_subspace(T, gnorm, sigma)
            s = Q @ z
            znorm = float(np.linalg.norm(z))
            model = gnorm * z[0] + 0.5 * float(z @ (T @ z)) + sigma / 3.0 * znorm ** 3
            f_trial = oracle.f(x + s)
            rho = (f - f_trial) / max(-model, 1e-300)
            if rho >= accept:
                x = x + s
                f = f_trial
                g = oracle.g(x)
                sigma = max(0.5 * sigma, sigma_min)
                basis = None
                rec.log(x, g, f)
            else:
                sigma *= 2.0
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ---------------------------------------------------------------------------- TR-NCG ------


def _steihaug(oracle: Oracle, x: np.ndarray, g: np.ndarray, delta: float, max_cg: int):
    """Steihaug-Toint CG for min g^T s + s^T H s / 2 subject to ||s|| <= delta.

    Returns (s, H s); the product H s is accumulated from the CG products, so the model
    value costs no additional Hessian-vector product.
    """
    n = g.size
    s = np.zeros(n)
    Hs = np.zeros(n)
    r = g.copy()
    d = -r
    gnorm = float(np.linalg.norm(g))
    tol = min(0.5, np.sqrt(gnorm)) * gnorm
    rr = float(r @ r)
    for _ in range(max_cg):
        if np.sqrt(rr) <= tol:
            break
        Hd = oracle.hvp(x, d)
        dHd = float(d @ Hd)
        if dHd <= 0.0:                                          # negative curvature: to the boundary
            tau = _boundary_step(s, d, delta)
            return s + tau * d, Hs + tau * Hd
        a = rr / dHd
        s_new = s + a * d
        if float(np.linalg.norm(s_new)) >= delta:
            tau = _boundary_step(s, d, delta)
            return s + tau * d, Hs + tau * Hd
        s = s_new
        Hs = Hs + a * Hd
        r = r + a * Hd
        rr_new = float(r @ r)
        d = -r + (rr_new / rr) * d
        rr = rr_new
    return s, Hs


def _boundary_step(s: np.ndarray, d: np.ndarray, delta: float) -> float:
    """tau >= 0 with ||s + tau d|| = delta."""
    a = float(d @ d)
    b = 2.0 * float(s @ d)
    c = float(s @ s) - delta * delta
    return (-b + np.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)


def trncg(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
          delta0: float = 1.0, max_cg: int = 50, accept: float = 0.1, expand: float = 0.75,
          gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Trust region with Steihaug-Toint conjugate gradients (Table 1)."""
    rec = Recorder(oracle, problem, "TR-NCG", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    delta = delta0
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        for _ in range(max_iter):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            oracle.site = "tr-cg"
            s, Hs = _steihaug(oracle, x, g, delta, max_cg)
            model = float(g @ s) + 0.5 * float(s @ Hs)
            f_trial = oracle.f(x + s)
            rho = (f - f_trial) / max(-model, 1e-300)
            if rho > accept:
                x = x + s
                f = f_trial
                g = oracle.g(x)
                rec.log(x, g, f)
                if rho > expand:
                    delta *= 2.0
            else:
                delta *= 0.25
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ------------------------------------------------------------------------------- NCG ------


def ncg(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
        gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Polak-Ribiere+ nonlinear conjugate gradients with Armijo backtracking (Table 1)."""
    rec = Recorder(oracle, problem, "NCG", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        d = -g
        alpha_prev, gd_prev = None, None
        for _ in range(max_iter):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            gd = float(g @ d)
            if gd >= 0.0:                                       # descent lost: restart
                d = -g
                gd = float(g @ d)
            alpha0 = 1.0 / max(float(np.linalg.norm(g)), 1e-300) if alpha_prev is None \
                else alpha_prev * gd_prev / gd
            alpha, f_new = _armijo(oracle, x, f, g, d, alpha0)
            if alpha is None:
                return _finish(rec, oracle, "stalled")
            x_new = x + alpha * d
            g_new = oracle.g(x_new)
            beta = max(0.0, float(g_new @ (g_new - g)) / max(float(g @ g), 1e-300))
            d_new = -g_new + beta * d
            alpha_prev, gd_prev = alpha, gd
            x, f, g, d = x_new, f_new, g_new, d_new
            rec.log(x, g, f)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ------------------------------------------------------------------------------ RegN ------


def regn(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
         cg_rtol: float = 1e-2, cg_maxiter: int = 200, gradient_test: bool = True,
         max_iter: int = 10**6) -> Result:
    """Regularized Newton with lambda_k = sqrt(L ||grad f||) (Mishchenko, Table 1)."""
    rec = Recorder(oracle, problem, "RegN", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        oracle.site = "scale"
        L = max(1e-12, power_norm(lambda v: oracle.hvp(x, v), x.size, iters=8))
        for _ in range(max_iter):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            lam = np.sqrt(L * float(np.linalg.norm(g)))
            oracle.site = "regn-cg"
            d, _ = cg(lambda v: oracle.hvp(x, v) + lam * v, -g, cg_rtol, min(cg_maxiter, x.size))
            if float(g @ d) >= 0.0:
                d = -g
            alpha, f_new = _armijo(oracle, x, f, g, d, 1.0)
            if alpha is None:
                return _finish(rec, oracle, "stalled")
            x = x + alpha * d
            f = f_new
            g = oracle.g(x)
            rec.log(x, g, f)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ---------------------------------------------------------------------------- L-BFGS ------


def lbfgs(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
          memory: int = 10, skip_tol: float = 1e-12, gradient_test: bool = True,
          max_iter: int = 10**6) -> Result:
    """Limited-memory BFGS, ten pairs, Armijo backtracking (Table 1)."""
    rec = Recorder(oracle, problem, "L-BFGS", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    S, Y, RHO = deque(), deque(), deque()
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        first = True
        for _ in range(max_iter):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            # two-loop recursion
            q = g.copy()
            alphas = []
            for s, y, rho in zip(reversed(S), reversed(Y), reversed(RHO)):
                a = rho * float(s @ q)
                alphas.append(a)
                q -= a * y
            if S:
                gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
                q *= gamma
            for (s, y, rho), a in zip(zip(S, Y, RHO), reversed(alphas)):
                b = rho * float(y @ q)
                q += (a - b) * s
            d = -q
            if float(g @ d) >= 0.0:                             # not a descent direction
                S.clear(); Y.clear(); RHO.clear()
                d = -g
                first = True
            alpha0 = 1.0 / max(float(np.linalg.norm(g)), 1e-300) if first else 1.0
            alpha, f_new = _armijo(oracle, x, f, g, d, alpha0)
            if alpha is None:
                return _finish(rec, oracle, "stalled")
            x_new = x + alpha * d
            g_new = oracle.g(x_new)
            s, y = x_new - x, g_new - g
            sy = float(s @ y)
            if sy > skip_tol:
                S.append(s); Y.append(y); RHO.append(1.0 / sy)
                if len(S) > memory:
                    S.popleft(); Y.popleft(); RHO.popleft()
            first = False
            x, f, g = x_new, f_new, g_new
            rec.log(x, g, f)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ------------------------------------------------------------------------------ Adam ------


def adam(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
         step: float = 1e-2, beta1: float = 0.9, beta2: float = 0.999, eps_adam: float = 1e-8,
         gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Full-batch Adam (Table 1)."""
    rec = Recorder(oracle, problem, "Adam", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    try:
        g = rec.start(x)
        for t in range(1, max_iter + 1):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g * g
            mhat = m / (1.0 - beta1 ** t)
            vhat = v / (1.0 - beta2 ** t)
            x = x - step * mhat / (np.sqrt(vhat) + eps_adam)
            g = oracle.g(x)
            rec.log(x, g)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ------------------------------------------------------------------------------- PGD ------


def pgd(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
        r: float = 1e-2, g_thres_factor: float = 1e-2, t_thres: int = 10, seed: int = 0,
        gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Perturbed gradient descent (Jin et al.), parameters of Table 1."""
    rec = Recorder(oracle, problem, "PGD", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        g_thres = g_thres_factor * max(1.0, float(np.linalg.norm(g)))
        oracle.site = "scale"
        L = max(1e-12, power_norm(lambda v: oracle.hvp(x, v), n, iters=8))
        eta = 1.0 / L
        t_noise = -t_thres - 1
        perturbed = False
        for t in range(max_iter):
            if not perturbed and rec.converged(g):
                return _finish(rec, oracle, "converged")
            if (not perturbed and float(np.linalg.norm(g)) <= g_thres
                    and t - t_noise > t_thres):
                xi = rng.standard_normal(n)
                xi *= r * rng.uniform() ** (1.0 / n) / np.linalg.norm(xi)   # uniform in the ball
                x = x + xi
                t_noise = t
                f = oracle.f(x)
                g = oracle.g(x)
                perturbed = True
                rec.log(x, g, f)
                continue
            while True:                                          # step halved when f increases
                x_new = x - eta * g
                f_new = oracle.f(x_new)
                if f_new <= f:
                    break
                eta *= 0.5
                if eta < 1e-300:
                    return _finish(rec, oracle, "stalled")
            x, f = x_new, f_new
            g = oracle.g(x)
            perturbed = False
            rec.log(x, g, f)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# -------------------------------------------------------------------------------- TN ------


def _modified_lanczos_direction(oracle: Oracle, x: np.ndarray, g: np.ndarray, rtol: float,
                                max_iter: int, delta: float):
    """Nash's modified Lanczos direction (1984, Sections 2.1-2.5, Theorem 3).

    Runs the Lanczos process from v_1 = -g/||g|| with full reorthogonalization; after each
    step factors the tridiagonal T_q as T_q + E_q = L D L^T (Gill-Murray), solves
    (T_q + E_q) y = ||g|| e_1 and forms p_q = V_q y, the minimizer of the modified quadratic
    model over the Krylov subspace.  The residual of the linear system G p + g = 0 is
    r_q = -V_q E_q y + beta_{q+1} y_q v_{q+1}, whose norm is computed from the Lanczos
    quantities without a further product.  Stops when ||r_q|| <= rtol ||g||, at breakdown or
    after ``max_iter`` steps.  Returns (p, p_nc, modified): the direction, the negative-curvature
    candidate V_q L^{-T} e_q of Section 2.5 (or None) and whether E_q != 0.
    """
    n = g.size
    gnorm = float(np.linalg.norm(g))
    m = max(1, min(max_iter, n))
    V = np.zeros((n, m + 1))
    alpha = np.zeros(m)
    beta = np.zeros(m)
    V[:, 0] = -g / gnorm
    y = None
    q = 0
    for j in range(m):
        w = oracle.hvp(x, V[:, j])
        alpha[j] = float(V[:, j] @ w)
        w = w - alpha[j] * V[:, j] - (beta[j - 1] * V[:, j - 1] if j > 0 else 0.0)
        w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)               # full reorthogonalization
        bj = float(np.linalg.norm(w))
        q = j + 1
        d, l, e = modified_ldl_tridiagonal(alpha[:q], beta[: q - 1], delta)
        # solve L D L^T y = ||g|| e_1 by forward and backward substitution on the bidiagonal L
        z = np.zeros(q)
        z[0] = gnorm
        for i in range(1, q):
            z[i] = -l[i - 1] * z[i - 1]
        y = z / d
        for i in range(q - 2, -1, -1):
            y[i] -= l[i] * y[i + 1]
        res_norm = math.sqrt(float(np.sum((e * y) ** 2)) + (bj * y[-1]) ** 2)
        if bj <= 1e-12 * max(1.0, abs(alpha[j])) or res_norm <= rtol * gnorm:
            break
        beta[j] = bj
        V[:, j + 1] = w / bj
    p = V[:, :q] @ y
    modified = bool(np.any(e > 0.0))
    p_nc = None
    if modified:                                        # Section 2.5: V_q L^{-T} e_q
        t = np.zeros(q)
        t[-1] = 1.0
        for i in range(q - 2, -1, -1):
            t[i] = -l[i] * t[i + 1]
        p_nc = V[:, :q] @ t
    return p, p_nc, modified


def tn(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
       delta_factor: float = 1e2, step_limit: float = 10.0, gradient_test: bool = True,
       max_iter: int = 10**6) -> Result:
    """Nash's truncated Newton method via the modified Lanczos method (Table 1).

    ``delta_factor * eps_mach`` is the pivot tolerance of the modified factorization (Nash:
    "usually a multiple of the relative machine precision"); ``step_limit`` is his bound
    Delta = 10 on ||x_{k+1} - x_k||_2, applied here by starting the backtracking at
    min{1, Delta/||d||}.
    """
    rec = Recorder(oracle, problem, "TN", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    n = x.size
    delta = delta_factor * float(np.finfo(float).eps)
    try:
        f = oracle.f(x)
        g = rec.start(x, f)
        g0 = float(np.linalg.norm(g))
        for k in range(1, max_iter + 1):
            if rec.converged(g):
                return _finish(rec, oracle, "converged")
            gnorm = float(np.linalg.norm(g))
            oracle.site = "tn-lanczos"
            rtol = min(1.0 / k, gnorm)                          # Dembo-Steihaug truncation
            d, d_nc, modified = _modified_lanczos_direction(oracle, x, g, rtol, max(1, n // 2), delta)
            stationary = gnorm <= tol_rel * max(1.0, g0)
            if stationary and modified and d_nc is not None:
                d = -d_nc if float(g @ d_nc) > 0.0 else d_nc     # negative curvature, Section 2.5
            if float(g @ d) > 0.0:
                d = -d
            alpha, f_new = _armijo(oracle, x, f, g, d, min(1.0, step_limit / max(float(np.linalg.norm(d)), 1e-300)))
            if alpha is None:
                return _finish(rec, oracle, "stalled")
            x = x + alpha * d
            f = f_new
            g = oracle.g(x)
            rec.log(x, g, f)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


# ------------------------------------------------------------------------------- INB ------

_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0


def inb(oracle: Oracle, problem: Problem, x0: np.ndarray, tol_rel: float = 1e-6, *,
        eta0: float = 1e-2, eta_max: float = 0.9, t: float = 1e-4, theta_min: float = 0.1,
        theta_max: float = 0.5, max_backtracks: int = 10, gmres_restart: int = 20,
        krylov_max: int = 200, gradient_test: bool = True, max_iter: int = 10**6) -> Result:
    """Inexact Newton backtracking (Eisenstat-Walker INB, Pawlowski et al. INB-Q) on grad f = 0."""
    rec = Recorder(oracle, problem, "INB", tol_rel, gradient_test)
    x = np.array(x0, dtype=float)
    eta = eta0
    try:
        F = rec.start(x)
        for _ in range(max_iter):
            if rec.converged(F):
                return _finish(rec, oracle, "converged")
            Fnorm = float(np.linalg.norm(F))
            oracle.site = "inb-gmres"
            s, _, _ = gmres_solve(lambda v: oracle.hvp(x, v), -F, eta, gmres_restart, krylov_max)
            F_new = oracle.g(x + s)
            Hs = None
            backtracks = 0
            while float(np.linalg.norm(F_new)) > (1.0 - t * (1.0 - eta)) * Fnorm:
                if backtracks >= max_backtracks:
                    return _finish(rec, oracle, "stalled")
                if Hs is None:
                    oracle.site = "inb-linesearch"
                    Hs = oracle.hvp(x, s)                        # p'(0) = F^T F'(x) s
                p0, dp0, p1 = 0.5 * Fnorm ** 2, float(F @ Hs), 0.5 * float(F_new @ F_new)
                curv = p1 - p0 - dp0
                theta = -dp0 / (2.0 * curv) if curv > 0.0 else theta_max
                theta = min(theta_max, max(theta_min, theta))
                s = theta * s
                Hs = theta * Hs
                eta = 1.0 - theta * (1.0 - eta)
                F_new = oracle.g(x + s)
                backtracks += 1
            if Hs is None:
                oracle.site = "inb-residual"
                Hs = oracle.hvp(x, s)
            lin_res = float(np.linalg.norm(F + Hs))              # ||F_k + F'_k s_k||
            F_new_norm = float(np.linalg.norm(F_new))
            eta_new = abs(F_new_norm - lin_res) / max(Fnorm, 1e-300)   # Choice 1, form (2.2)
            safeguard = eta ** _GOLDEN
            if safeguard > 0.1:
                eta_new = max(eta_new, safeguard)
            eta = min(eta_max, eta_new)
            x = x + s
            F = F_new
            rec.log(x, F)
        return _finish(rec, oracle, "iteration cap")
    except BudgetExceeded:
        return _finish(rec, oracle, oracle.exhausted or "budget")


BASELINES = {"ARC": arc, "TR-NCG": trncg, "NCG": ncg, "RegN": regn, "L-BFGS": lbfgs,
             "Adam": adam, "PGD": pgd, "TN": tn, "INB": inb}
