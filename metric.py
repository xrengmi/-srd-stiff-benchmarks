"""Experiment behind Appendix B (inner convergence in the discrete H^1 metric).

For the five-point Allen-Cahn energy of Section 7.2 at its starting point u0:

* the matrix M_h = h^2 (I+L) is identified with the H^1 Gram matrix of the piecewise linear (P1)
  basis on the uniform triangulation obtained by halving each square cell along one diagonal,
  with the L^2 part lumped: the P1 stiffness matrix is assembled element by element and compared
  with h^2 L, and the integrals of the interior hat functions (the vertex-quadrature lumped mass)
  are compared with h^2 (N = 8, 16, 32);
* the discrete splitting

      M_h^{-1} H = eps_AC I + (I+L)^{-1} diag(3 u0^2 - 1 - eps_AC)

  is checked to rounding error on a random vector, and the number of singular values of the
  second term above 1e-2 is counted (dense, N = 8, 16, 32, 64);
* the regularized Newton system at u0 is solved by conjugate gradients to a relative Euclidean
  residual of 1e-10 for N = 8, 16, 32, 64, 128, with the regularization placed in the metric in
  which the iteration works:
    (a) Euclidean metric: (H + alpha_E I) x = -grad f(u0), alpha_E = 1e-3 ||H||_2, plain CG;
    (b) M_h metric: (H + alpha_M M_h) x = -grad f(u0), i.e. (M_h^{-1} H + alpha_M I) x = -M_h^{-1} grad f(u0),
        alpha_M = 1e-3 max |mu| over the generalized eigenvalues H w = mu M_h w, solved by CG
        preconditioned by M_h (CG in the M_h inner product).
  Eigenvalues are computed densely for N <= 64 and by ARPACK for N = 128.  The right-hand side
  is the Newton right-hand side at u0; the manuscript does not fix it, and the choice is stated
  in the text.

Run as ``python -m srdssink.metric`` from the package root; writes results/metric.json.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from .problems import allen_cahn

__all__ = ["run", "pcg"]

SV_SIZES = (8, 16, 32, 64)
CG_SIZES = (8, 16, 32, 64, 128)
P1_SIZES = (8, 16, 32)
SV_THRESHOLD = 1e-2
ALPHA_FACTOR = 1e-3
CG_RTOL = 1e-10
CG_MAXITER = 5000


def laplacian_matrix(N: int) -> sp.csr_matrix:
    """Sparse five-point Laplacian times (N+1)^2 with zero Dirichlet values (same as problems.laplacian_5pt)."""
    e = np.ones(N)
    T = sp.diags([-e[1:], 2.0 * e, -e[1:]], [-1, 0, 1], format="csr")
    I = sp.identity(N, format="csr")
    return ((sp.kron(I, T) + sp.kron(T, I)) * (N + 1) ** 2).tocsr()


def pcg(matvec, b, rtol, maxiter, prec=None):
    """Conjugate gradients on a symmetric positive definite operator; stops when the Euclidean
    residual satisfies ||r|| <= rtol ||b||.  ``prec`` applies the inverse preconditioner (M^{-1}).
    Returns (x, iterations, final relative residual, curvature flag)."""
    x = np.zeros_like(b)
    r = b.copy()
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return x, 0, 0.0, True
    z = prec(r) if prec is not None else r
    p = z.copy()
    rz = float(r @ z)
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            return x, k - 1, float(np.linalg.norm(r)) / bnorm, False
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        if float(np.linalg.norm(r)) <= rtol * bnorm:
            return x, k, float(np.linalg.norm(r)) / bnorm, True
        z = prec(r) if prec is not None else r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter, float(np.linalg.norm(r)) / bnorm, True


def p1_matrices(N: int):
    """P1 stiffness and consistent mass matrices on the unit square, uniform grid with N interior
    points per direction (h = 1/(N+1)), each square cell halved along the diagonal from its
    lower-left to its upper-right corner; rows and columns of the (N+2)^2 grid restricted to the
    N^2 interior nodes (zero Dirichlet values).  Returns (K, M, w): the interior blocks of the
    stiffness and consistent mass matrices as dense arrays, and the integrals w_i of the interior
    hat functions (the weights of the vertex quadrature rule, i.e. the lumped mass).  The interior
    nodes are ordered with the x index outermost; the five-point matrix of laplacian_matrix is
    invariant under exchanging the two grid indices, so the comparison with it is order-free."""
    m = N + 2
    h = 1.0 / (N + 1)
    idx = lambda i, j: i * m + j                      # i = x index, j = y index
    K = np.zeros((m * m, m * m))
    M = np.zeros((m * m, m * m))
    Mloc = (h * h / 2.0) / 12.0 * (np.ones((3, 3)) + np.eye(3))   # consistent mass of a triangle
    for i in range(m - 1):
        for j in range(m - 1):
            p00, p10, p01, p11 = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            for tri in ((p00, p10, p11), (p00, p11, p01)):
                X = np.array([[a * h, b * h] for a, b in tri])
                B = np.array([X[1] - X[0], X[2] - X[0]]).T
                area = 0.5 * abs(np.linalg.det(B))
                G = np.linalg.solve(B.T, np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]))
                Kloc = area * (G.T @ G)
                ids = [idx(a, b) for a, b in tri]
                for r in range(3):
                    for c in range(3):
                        K[ids[r], ids[c]] += Kloc[r, c]
                        M[ids[r], ids[c]] += Mloc[r, c]
    interior = [idx(i, j) for i in range(1, m - 1) for j in range(1, m - 1)]
    # integral of each interior hat function = row sum of the consistent mass matrix over all nodes
    hat_integrals = M[interior, :].sum(axis=1)
    return K[np.ix_(interior, interior)], M[np.ix_(interior, interior)], hat_integrals


def _hessian_matrix(N: int, u0: np.ndarray, eps_ac: float, h2: float) -> sp.csr_matrix:
    L = laplacian_matrix(N)
    return (h2 * (eps_ac * L + sp.diags(3.0 * u0 * u0 - 1.0))).tocsr()


def run(out_path: str = os.path.join("results", "metric.json"), verbose: bool = True) -> dict:
    out = {"sv_threshold": SV_THRESHOLD, "alpha_factor": ALPHA_FACTOR, "cg_rtol": CG_RTOL,
           "splitting_error": {}, "sv_count": {}, "cg": {}, "p1": {}}
    for N in P1_SIZES:
        K1, M1, w1 = p1_matrices(N)
        h = 1.0 / (N + 1)
        hL = (h * h) * laplacian_matrix(N).toarray()
        out["p1"][str(N)] = {
            "stiffness_vs_h2L_maxabs": float(np.max(np.abs(K1 - hL))),
            "stiffness_maxabs": float(np.max(np.abs(K1))),
            "lumped_mass_vs_h2_maxrel": float(np.max(np.abs(w1 - h * h)) / (h * h)),
            "consistent_mass_offdiag_nonzero": bool(np.max(np.abs(M1 - np.diag(np.diag(M1)))) > 0.0),
        }
        if verbose:
            q = out["p1"][str(N)]
            print(f"P1 N={N}: |K - h^2 L|_max = {q['stiffness_vs_h2L_maxabs']:.1e} (|K|_max {q['stiffness_maxabs']:.1f}),"
                  f" max |w_i/h^2 - 1| = {q['lumped_mass_vs_h2_maxrel']:.1e}", flush=True)
    for N in sorted(set(SV_SIZES) | set(CG_SIZES)):
        t0 = time.time()
        p = allen_cahn(N)
        n, h, eps_ac = N * N, p.extra["h"], p.extra["eps_ac"]
        h2 = h * h
        u0 = p.x0
        L = laplacian_matrix(N)
        H = _hessian_matrix(N, u0, eps_ac, h2)
        # consistency with the matrix-free oracle
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n)
        oracle_err = float(np.linalg.norm(H @ v - p.hvp(u0, v)) / np.linalg.norm(H @ v))
        IL = (sp.identity(n, format="csc") + L).tocsc()
        lu = spla.splu(IL)
        D = 3.0 * u0 * u0 - 1.0 - eps_ac
        # splitting check: M_h^{-1} H v  vs  eps v + (I+L)^{-1} (D v)
        lhs = lu.solve(H @ v) / h2
        rhs = eps_ac * v + lu.solve(D * v)
        split_err = float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs))
        out["splitting_error"][str(N)] = {"splitting": split_err, "oracle": oracle_err}
        # singular values of (I+L)^{-1} diag(D)
        if N in SV_SIZES:
            IL_dense = IL.toarray()
            second = np.linalg.solve(IL_dense, np.diag(D))
            s = np.linalg.svd(second, compute_uv=False)
            out["sv_count"][str(N)] = {"n": n, "count_above": int(np.sum(s > SV_THRESHOLD)),
                                       "largest": float(s[0]), "smallest": float(s[-1])}
        # Euclidean spectrum of H and generalized spectrum of (H, M_h)
        Mh = (h2 * IL).tocsr()
        if N <= 64:
            Hd = H.toarray()
            w = np.linalg.eigvalsh(Hd)
            hnorm, lam_min = float(np.max(np.abs(w))), float(w[0])
            mu = sla.eigh(Hd, Mh.toarray(), eigvals_only=True)
            munorm, mu_min = float(np.max(np.abs(mu))), float(mu[0])
        else:
            hnorm = float(abs(spla.eigsh(H, k=1, which="LM", return_eigenvectors=False)[0]))
            lam_min = float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
            # generalized eigenvalues H w = mu M_h w (mu are the eigenvalues of M_h^{-1} H);
            # the smallest one by shift-invert about -1, below the whole spectrum
            munorm = float(abs(spla.eigsh(H, k=1, M=Mh.tocsc(), which="LM", return_eigenvectors=False)[0]))
            mu_min = float(spla.eigsh(H, k=1, M=Mh.tocsc(), sigma=-1.0, which="LM", return_eigenvectors=False)[0])
        if N in CG_SIZES:
            b = -p.grad(u0)
            alpha_e = ALPHA_FACTOR * hnorm
            A_e = (H + alpha_e * sp.identity(n, format="csr")).tocsr()
            _, it_e, res_e, ok_e = pcg(lambda x: A_e @ x, b, CG_RTOL, CG_MAXITER)
            alpha_m = ALPHA_FACTOR * munorm
            A_m = (H + alpha_m * Mh).tocsr()
            _, it_m, res_m, ok_m = pcg(lambda x: A_m @ x, b, CG_RTOL, CG_MAXITER,
                                       prec=lambda r: lu.solve(r) / h2)
            out["cg"][str(N)] = {"n": n,
                                 "alpha": alpha_e, "hnorm": hnorm, "lambda_min": lam_min,
                                 "euclid_pd_spectral": bool(lam_min + alpha_e > 0.0),
                                 "euclid_iters": it_e, "euclid_resid": res_e, "euclid_pd": ok_e,
                                 "alpha_mh": alpha_m, "munorm": munorm, "mu_min": mu_min,
                                 "mh_pd_spectral": bool(mu_min + alpha_m > 0.0),
                                 "mh_iters": it_m, "mh_resid": res_m, "mh_pd": ok_m}
        if verbose:
            msg = f"N={N:4d} n={n:6d} split {split_err:.1e} oracle {oracle_err:.1e} |H|={hnorm:.3e} lmin={lam_min:+.3e}"
            if str(N) in out["sv_count"]:
                msg += f"  sv>{SV_THRESHOLD:g}: {out['sv_count'][str(N)]['count_above']}"
            if str(N) in out["cg"]:
                c = out["cg"][str(N)]
                msg += (f"  CG euclid {c['euclid_iters']} ({c['euclid_resid']:.1e}, pd {c['euclid_pd_spectral']})"
                        f" M_h {c['mh_iters']} ({c['mh_resid']:.1e}, mu_min {c['mu_min']:+.2e}, alpha_M {c['alpha_mh']:.2e}, pd {c['mh_pd_spectral']})")
            print(msg + f"  [{time.time()-t0:.1f}s]", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
