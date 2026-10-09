"""Test problems of Section 7.2 of the paper, with analytic gradients and Hessian-vector products.

Groups (budgets of Section 7.1 in work units):
  classical (3e5): extended Rosenbrock (n = 100, 1000), indefinite quartic (n = 100, 1000),
                   discrete Allen-Cahn (n = 256, 1024);
  cuter     (3e5): ARWHEAD, ENGVAL1, LIARWHD, NONDIA, FLETCHCR, EDENSCH, GENHUMPS, COSINE,
                   CURLY10, DQDRTIC at n = 1000 and 10000 (CUTEr definitions and starting
                   points, except GENHUMPS whose CUTEr point is scaled by 1/100, Section 7.1);
  andrei   (1.5e5): extended Himmelblau, EG2, generalized tridiagonal 1, diagonal 3,
                   extended Beale, Raydan 1, TRIDIA at n = 3000 (Andrei's collection).

Definitions the manuscript leaves to the authors, and the choices made here (each is stated
in the manuscript with the run):
  * indefinite quartic:  f(u) = sum_i (u_i^4/4 - u_i^2), the separable quartic consistent with
    every statement of the manuscript about it (f(0) = 0, lambda_min(H(0)) = -2 with
    multiplicity n, minimizers at u_i = +-sqrt(2) with f = -n, no critical value below -n);
  * Allen-Cahn: f(u) = h^2 [ eps_AC/2 u^T L u + sum_i (u_i^2 - 1)^2/4 ] with the five-point
    Laplacian L = h^{-2} * stencil(4, -1, -1, -1, -1) and homogeneous Dirichlet boundary
    values on the unit square, N x N interior points, h = 1/(N+1), eps_AC = 0.01,
    u_0 ~ U(-1, 1)^n with numpy seed 2024;
  * the low-rank factorization instance is not defined in the manuscript and is therefore
    not part of this campaign;
  * the machine-learning instance (Section 7.9): x_i ~ N(0, I_d), teacher parameters ~ N(0, 1)
    (seed 0), student starting point ~ N(0, 1) (seed 1), N = 500, lambda = 1e-3.

Every problem is checked by ``srdssink.verify`` before a campaign (Section 7.2).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .oracle import Problem

__all__ = ["classical_suite", "cuter_suite", "andrei_suite", "all_instances", "mlp_problem",
           "quartic", "allen_cahn", "rosenbrock", "by_name"]

BUDGET_CLASSICAL = 3.0e5
BUDGET_CUTER = 3.0e5
BUDGET_ANDREI = 1.5e5
BUDGET_MLP = 3.0e5

# --------------------------------------------------------------------------- classical ----


def rosenbrock(n: int) -> Problem:
    """Extended Rosenbrock (More, Garbow, Hillstrom): sum_i 100 (x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2."""
    assert n % 2 == 0

    def f(x):
        a, b = x[0::2], x[1::2]
        return float(np.sum(100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2))

    def g(x):
        a, b = x[0::2], x[1::2]
        out = np.empty_like(x)
        out[0::2] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a)
        out[1::2] = 200.0 * (b - a * a)
        return out

    def hvp(x, v):
        a, b = x[0::2], x[1::2]
        va, vb = v[0::2], v[1::2]
        haa = 1200.0 * a * a - 400.0 * b + 2.0
        hab = -400.0 * a
        out = np.empty_like(x)
        out[0::2] = haa * va + hab * vb
        out[1::2] = hab * va + 200.0 * vb
        return out

    x0 = np.empty(n)
    x0[0::2], x0[1::2] = -1.2, 1.0
    return Problem(f"ROSENBROCK-{n}", f, g, hvp, x0, "classical", BUDGET_CLASSICAL)


def quartic(n: int) -> Problem:
    """Indefinite quartic f(u) = sum_i (u_i^4/4 - u_i^2); H(0) = -2 I; start 0.1 * ones."""

    def f(x):
        return float(np.sum(0.25 * x ** 4 - x ** 2))

    def g(x):
        return x ** 3 - 2.0 * x

    def hvp(x, v):
        return (3.0 * x * x - 2.0) * v

    return Problem(f"QUARTIC-{n}", f, g, hvp, 0.1 * np.ones(n), "classical", BUDGET_CLASSICAL,
                   extra={"saddle": np.zeros(n), "escape_direction": np.ones(n) / np.sqrt(n),
                          "lambda_min_saddle": -2.0, "multiplicity_saddle": n})


def laplacian_5pt(N: int) -> Callable[[np.ndarray], np.ndarray]:
    """Action of the five-point Laplacian (times h^{-2}) with zero Dirichlet boundary values."""
    h2 = (N + 1) ** 2

    def apply(u):
        U = u.reshape(N, N)
        out = 4.0 * U
        out[1:, :] -= U[:-1, :]
        out[:-1, :] -= U[1:, :]
        out[:, 1:] -= U[:, :-1]
        out[:, :-1] -= U[:, 1:]
        return (out * h2).ravel()

    return apply


def allen_cahn(N: int, eps_ac: float = 0.01, seed: int = 2024) -> Problem:
    """Discrete Allen-Cahn energy on the unit square, n = N^2 interior points."""
    n = N * N
    h = 1.0 / (N + 1)
    h2 = h * h
    L = laplacian_5pt(N)

    def f(u):
        return float(h2 * (0.5 * eps_ac * (u @ L(u)) + np.sum(0.25 * (u * u - 1.0) ** 2)))

    def g(u):
        return h2 * (eps_ac * L(u) + (u * u - 1.0) * u)

    def hvp(u, v):
        return h2 * (eps_ac * L(v) + (3.0 * u * u - 1.0) * v)

    u0 = np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    return Problem(f"ALLENCAHN-{n}", f, g, hvp, u0, "classical", BUDGET_CLASSICAL,
                   extra={"N": N, "h": h, "eps_ac": eps_ac, "seed": seed})


def classical_suite(sizes=(100, 1000), ac_sizes=(16, 32)) -> List[Problem]:
    out = [rosenbrock(n) for n in sizes] + [quartic(n) for n in sizes]
    out += [allen_cahn(N) for N in ac_sizes]
    return out


# ------------------------------------------------------------------------------- CUTEr ----


def arwhead(n: int) -> Problem:
    """sum_{i<n} [(x_i^2 + x_n^2)^2 - 4 x_i + 3], x0 = 1."""

    def f(x):
        s = x[:-1] ** 2 + x[-1] ** 2
        return float(np.sum(s * s - 4.0 * x[:-1] + 3.0))

    def g(x):
        s = x[:-1] ** 2 + x[-1] ** 2
        out = np.empty_like(x)
        out[:-1] = 4.0 * s * x[:-1] - 4.0
        out[-1] = 4.0 * x[-1] * np.sum(s)
        return out

    def hvp(x, v):
        xn, vn = x[-1], v[-1]
        s = x[:-1] ** 2 + xn * xn
        out = np.empty_like(x)
        out[:-1] = (4.0 * s + 8.0 * x[:-1] ** 2) * v[:-1] + 8.0 * x[:-1] * xn * vn
        out[-1] = 8.0 * xn * np.sum(x[:-1] * v[:-1]) + np.sum(4.0 * s + 8.0 * xn * xn) * vn
        return out

    return Problem(f"ARWHEAD-{n}", f, g, hvp, np.ones(n), "cuter", BUDGET_CUTER)


def engval1(n: int) -> Problem:
    """sum_{i<n} [(x_i^2 + x_{i+1}^2)^2 - 4 x_i + 3], x0 = 2."""

    def f(x):
        s = x[:-1] ** 2 + x[1:] ** 2
        return float(np.sum(s * s - 4.0 * x[:-1] + 3.0))

    def g(x):
        s = x[:-1] ** 2 + x[1:] ** 2
        out = np.zeros_like(x)
        out[:-1] += 4.0 * s * x[:-1] - 4.0
        out[1:] += 4.0 * s * x[1:]
        return out

    def hvp(x, v):
        a, b, va, vb = x[:-1], x[1:], v[:-1], v[1:]
        s = a * a + b * b
        out = np.zeros_like(x)
        out[:-1] += (4.0 * s + 8.0 * a * a) * va + 8.0 * a * b * vb
        out[1:] += 8.0 * a * b * va + (4.0 * s + 8.0 * b * b) * vb
        return out

    return Problem(f"ENGVAL1-{n}", f, g, hvp, 2.0 * np.ones(n), "cuter", BUDGET_CUTER)


def liarwhd(n: int) -> Problem:
    """sum_i [4 (x_i^2 - x_1)^2 + (x_i - 1)^2], x0 = 4."""

    def f(x):
        r = x * x - x[0]
        return float(np.sum(4.0 * r * r + (x - 1.0) ** 2))

    def g(x):
        r = x * x - x[0]
        out = 16.0 * r * x + 2.0 * (x - 1.0)
        out[0] -= 8.0 * np.sum(r)
        return out

    def hvp(x, v):
        r = x * x - x[0]
        c = 2.0 * x * v - v[0]                 # a_j^T v with a_j = 2 x_j e_j - e_1
        out = 8.0 * (2.0 * x * c) + (16.0 * r + 2.0) * v
        out[0] -= 8.0 * np.sum(c)
        return out

    return Problem(f"LIARWHD-{n}", f, g, hvp, 4.0 * np.ones(n), "cuter", BUDGET_CUTER)


def nondia(n: int) -> Problem:
    """(x_1 - 1)^2 + 100 sum_{i=2}^n (x_1 - x_{i-1}^2)^2, x0 = -1."""

    def f(x):
        r = x[0] - x[:-1] ** 2
        return float((x[0] - 1.0) ** 2 + 100.0 * np.sum(r * r))

    def g(x):
        r = x[0] - x[:-1] ** 2                   # r_i for i = 2..n indexed by i-1 = 0..n-2
        out = np.zeros_like(x)
        out[:-1] += -400.0 * r * x[:-1]
        out[0] += 2.0 * (x[0] - 1.0) + 200.0 * np.sum(r)
        return out

    def hvp(x, v):
        xm, vm = x[:-1], v[:-1]
        r = x[0] - xm * xm
        c = v[0] - 2.0 * xm * vm                 # b_i^T v with b_i = e_1 - 2 x_{i-1} e_{i-1}
        out = np.zeros_like(x)
        out[:-1] += 200.0 * (-2.0 * xm * c) + 200.0 * r * (-2.0) * vm
        out[0] += 2.0 * v[0] + 200.0 * np.sum(c)
        return out

    return Problem(f"NONDIA-{n}", f, g, hvp, -np.ones(n), "cuter", BUDGET_CUTER)


def fletchcr(n: int) -> Problem:
    """100 sum_{i<n} (x_{i+1} - x_i + 1 - x_i^2)^2, x0 = 0."""

    def f(x):
        r = x[1:] - x[:-1] + 1.0 - x[:-1] ** 2
        return float(100.0 * np.sum(r * r))

    def g(x):
        a = x[:-1]
        r = x[1:] - a + 1.0 - a * a
        out = np.zeros_like(x)
        out[:-1] += 200.0 * r * (-1.0 - 2.0 * a)
        out[1:] += 200.0 * r
        return out

    def hvp(x, v):
        a, va, vb = x[:-1], v[:-1], v[1:]
        r = x[1:] - a + 1.0 - a * a
        da = -1.0 - 2.0 * a
        c = da * va + vb                          # grad r_i . v
        out = np.zeros_like(x)
        out[:-1] += 200.0 * (da * c + r * (-2.0) * va)
        out[1:] += 200.0 * c
        return out

    return Problem(f"FLETCHCR-{n}", f, g, hvp, np.zeros(n), "cuter", BUDGET_CUTER)


def edensch(n: int) -> Problem:
    """16 + sum_{i<n} [(x_i - 2)^4 + (x_i x_{i+1} - 2 x_{i+1})^2 + (x_{i+1} + 1)^2], x0 = 0."""

    def f(x):
        a, b = x[:-1], x[1:]
        return float(16.0 + np.sum((a - 2.0) ** 4 + (b * (a - 2.0)) ** 2 + (b + 1.0) ** 2))

    def g(x):
        a, b = x[:-1], x[1:]
        q = b * (a - 2.0)
        out = np.zeros_like(x)
        out[:-1] += 4.0 * (a - 2.0) ** 3 + 2.0 * q * b
        out[1:] += 2.0 * q * (a - 2.0) + 2.0 * (b + 1.0)
        return out

    def hvp(x, v):
        a, b, va, vb = x[:-1], x[1:], v[:-1], v[1:]
        q = b * (a - 2.0)
        dq = b * va + (a - 2.0) * vb              # grad q . v
        out = np.zeros_like(x)
        out[:-1] += 12.0 * (a - 2.0) ** 2 * va + 2.0 * b * dq + 2.0 * q * vb
        out[1:] += 2.0 * (a - 2.0) * dq + 2.0 * q * va + 2.0 * vb
        return out

    return Problem(f"EDENSCH-{n}", f, g, hvp, np.zeros(n), "cuter", BUDGET_CUTER)


def genhumps(n: int, zeta: float = 20.0, scale: float = 1.0 / 100.0) -> Problem:
    """sum_{i<n} [sin^2(zeta x_i) sin^2(zeta x_{i+1}) + 0.05 (x_i^2 + x_{i+1}^2)]; CUTEr point scaled."""

    def f(x):
        s = np.sin(zeta * x) ** 2
        return float(np.sum(s[:-1] * s[1:] + 0.05 * (x[:-1] ** 2 + x[1:] ** 2)))

    def g(x):
        s = np.sin(zeta * x) ** 2
        ds = zeta * np.sin(2.0 * zeta * x)
        out = np.zeros_like(x)
        out[:-1] += ds[:-1] * s[1:] + 0.1 * x[:-1]
        out[1:] += s[:-1] * ds[1:] + 0.1 * x[1:]
        return out

    def hvp(x, v):
        s = np.sin(zeta * x) ** 2
        ds = zeta * np.sin(2.0 * zeta * x)
        dds = 2.0 * zeta * zeta * np.cos(2.0 * zeta * x)
        va, vb = v[:-1], v[1:]
        out = np.zeros_like(x)
        out[:-1] += (dds[:-1] * s[1:] + 0.1) * va + ds[:-1] * ds[1:] * vb
        out[1:] += ds[:-1] * ds[1:] * va + (s[:-1] * dds[1:] + 0.1) * vb
        return out

    x0 = -506.2 * np.ones(n)
    x0[0] = -506.0
    return Problem(f"GENHUMPS-{n}", f, g, hvp, scale * x0, "cuter", BUDGET_CUTER,
                   extra={"zeta": zeta, "start_scale": scale})


def cosine(n: int) -> Problem:
    """sum_{i<n} cos(x_i^2 - x_{i+1}/2), x0 = 1."""

    def f(x):
        return float(np.sum(np.cos(x[:-1] ** 2 - 0.5 * x[1:])))

    def g(x):
        th = x[:-1] ** 2 - 0.5 * x[1:]
        s = np.sin(th)
        out = np.zeros_like(x)
        out[:-1] += -s * 2.0 * x[:-1]
        out[1:] += -s * (-0.5)
        return out

    def hvp(x, v):
        a, va, vb = x[:-1], v[:-1], v[1:]
        th = a * a - 0.5 * x[1:]
        s, c = np.sin(th), np.cos(th)
        dth = 2.0 * a * va - 0.5 * vb
        out = np.zeros_like(x)
        out[:-1] += -c * dth * 2.0 * a - s * 2.0 * va
        out[1:] += -c * dth * (-0.5)
        return out

    return Problem(f"COSINE-{n}", f, g, hvp, np.ones(n), "cuter", BUDGET_CUTER)


def curly10(n: int, band: int = 10) -> Problem:
    """sum_i phi(q_i), q_i = sum_{j=i}^{min(i+10,n)} x_j, phi(q) = q^4 - 20 q^2 - 0.1 q; x0_i = 1e-4 i/(n+1)."""
    idx_hi = np.minimum(np.arange(n) + band, n - 1)       # 0-based upper index of each window

    def S(x):                                             # q = S x
        C = np.concatenate([[0.0], np.cumsum(x)])
        return C[idx_hi + 1] - C[np.arange(n)]

    def ST(y):                                            # S^T y
        C = np.concatenate([[0.0], np.cumsum(y)])
        j = np.arange(n)
        lo = np.maximum(j - band, 0)
        return C[j + 1] - C[lo]

    def f(x):
        q = S(x)
        return float(np.sum(q ** 4 - 20.0 * q * q - 0.1 * q))

    def g(x):
        q = S(x)
        return ST(4.0 * q ** 3 - 40.0 * q - 0.1)

    def hvp(x, v):
        q = S(x)
        return ST((12.0 * q * q - 40.0) * S(v))

    x0 = 1e-4 * np.arange(1, n + 1) / (n + 1)
    return Problem(f"CURLY10-{n}", f, g, hvp, x0, "cuter", BUDGET_CUTER)


def dqdrtic(n: int) -> Problem:
    """sum_{i<=n-2} [x_i^2 + 100 x_{i+1}^2 + 100 x_{i+2}^2], x0 = 3."""
    c = np.zeros(n)
    c[: n - 2] += 1.0
    c[1: n - 1] += 100.0
    c[2:] += 100.0

    def f(x):
        return float(np.sum(c * x * x))

    def g(x):
        return 2.0 * c * x

    def hvp(x, v):
        return 2.0 * c * v

    return Problem(f"DQDRTIC-{n}", f, g, hvp, 3.0 * np.ones(n), "cuter", BUDGET_CUTER)


def tridia(n: int, group: str = "andrei", budget: float = BUDGET_ANDREI) -> Problem:
    """(x_1 - 1)^2 + sum_{i=2}^n i (2 x_i - x_{i-1})^2, x0 = 1."""
    w = np.arange(2, n + 1, dtype=float)

    def f(x):
        r = 2.0 * x[1:] - x[:-1]
        return float((x[0] - 1.0) ** 2 + np.sum(w * r * r))

    def g(x):
        r = 2.0 * x[1:] - x[:-1]
        out = np.zeros_like(x)
        out[0] += 2.0 * (x[0] - 1.0)
        out[1:] += 4.0 * w * r
        out[:-1] += -2.0 * w * r
        return out

    def hvp(x, v):
        dr = 2.0 * v[1:] - v[:-1]
        out = np.zeros_like(x)
        out[0] += 2.0 * v[0]
        out[1:] += 4.0 * w * dr
        out[:-1] += -2.0 * w * dr
        return out

    return Problem(f"TRIDIA-{n}", f, g, hvp, np.ones(n), group, budget)


def cuter_suite(sizes=(1000, 10000)) -> List[Problem]:
    out = []
    for n in sizes:
        out += [arwhead(n), engval1(n), liarwhd(n), nondia(n), fletchcr(n), edensch(n),
                genhumps(n), cosine(n), curly10(n), dqdrtic(n)]
    return out


# ------------------------------------------------------------------------------ Andrei ----


def ext_himmelblau(n: int) -> Problem:
    """sum_i (x_{2i-1}^2 + x_{2i} - 11)^2 + (x_{2i-1} + x_{2i}^2 - 7)^2, x0 = 1."""
    assert n % 2 == 0

    def f(x):
        a, b = x[0::2], x[1::2]
        return float(np.sum((a * a + b - 11.0) ** 2 + (a + b * b - 7.0) ** 2))

    def g(x):
        a, b = x[0::2], x[1::2]
        r1, r2 = a * a + b - 11.0, a + b * b - 7.0
        out = np.empty_like(x)
        out[0::2] = 4.0 * a * r1 + 2.0 * r2
        out[1::2] = 2.0 * r1 + 4.0 * b * r2
        return out

    def hvp(x, v):
        a, b, va, vb = x[0::2], x[1::2], v[0::2], v[1::2]
        haa = 12.0 * a * a + 4.0 * b - 42.0
        hbb = 12.0 * b * b + 4.0 * a - 26.0
        hab = 4.0 * (a + b)
        out = np.empty_like(x)
        out[0::2] = haa * va + hab * vb
        out[1::2] = hab * va + hbb * vb
        return out

    return Problem(f"EXTHIMMELBLAU-{n}", f, g, hvp, np.ones(n), "andrei", BUDGET_ANDREI)


def eg2(n: int) -> Problem:
    """sum_{i<n} sin(x_1 + x_i^2 - 1) + sin(x_n^2)/2, x0 = 1."""

    def f(x):
        return float(np.sum(np.sin(x[0] + x[:-1] ** 2 - 1.0)) + 0.5 * np.sin(x[-1] ** 2))

    def g(x):
        th = x[0] + x[:-1] ** 2 - 1.0
        c = np.cos(th)
        out = np.zeros_like(x)
        out[:-1] += c * 2.0 * x[:-1]
        out[0] += np.sum(c)
        out[-1] += x[-1] * np.cos(x[-1] ** 2)
        return out

    def hvp(x, v):
        a, va = x[:-1], v[:-1]
        th = x[0] + a * a - 1.0
        s, c = np.sin(th), np.cos(th)
        dth = v[0] + 2.0 * a * va               # grad theta_i . v
        out = np.zeros_like(x)
        out[:-1] += -s * dth * 2.0 * a + c * 2.0 * va
        out[0] += np.sum(-s * dth)
        xn, vn = x[-1], v[-1]
        out[-1] += (np.cos(xn * xn) - 2.0 * xn * xn * np.sin(xn * xn)) * vn
        return out

    return Problem(f"EG2-{n}", f, g, hvp, np.ones(n), "andrei", BUDGET_ANDREI)


def gen_tridiag1(n: int) -> Problem:
    """sum_{i<n} (x_i + x_{i+1} - 3)^2 + (x_i - x_{i+1} + 1)^4, x0 = 2."""

    def f(x):
        p, q = x[:-1] + x[1:] - 3.0, x[:-1] - x[1:] + 1.0
        return float(np.sum(p * p + q ** 4))

    def g(x):
        p, q = x[:-1] + x[1:] - 3.0, x[:-1] - x[1:] + 1.0
        out = np.zeros_like(x)
        out[:-1] += 2.0 * p + 4.0 * q ** 3
        out[1:] += 2.0 * p - 4.0 * q ** 3
        return out

    def hvp(x, v):
        q = x[:-1] - x[1:] + 1.0
        va, vb = v[:-1], v[1:]
        dp, dq = va + vb, va - vb
        out = np.zeros_like(x)
        out[:-1] += 2.0 * dp + 12.0 * q * q * dq
        out[1:] += 2.0 * dp - 12.0 * q * q * dq
        return out

    return Problem(f"GENTRIDIAG1-{n}", f, g, hvp, 2.0 * np.ones(n), "andrei", BUDGET_ANDREI)


def diagonal3(n: int) -> Problem:
    """sum_i (exp(x_i) - i sin(x_i)), x0 = 1."""
    i = np.arange(1, n + 1, dtype=float)

    def f(x):
        return float(np.sum(np.exp(x) - i * np.sin(x)))

    def g(x):
        return np.exp(x) - i * np.cos(x)

    def hvp(x, v):
        return (np.exp(x) + i * np.sin(x)) * v

    return Problem(f"DIAGONAL3-{n}", f, g, hvp, np.ones(n), "andrei", BUDGET_ANDREI)


def ext_beale(n: int) -> Problem:
    """Extended Beale, pairs (a, b): sum of [1.5 - a(1-b)]^2 + [2.25 - a(1-b^2)]^2 + [2.625 - a(1-b^3)]^2."""
    assert n % 2 == 0

    def parts(a, b):
        r1, r2, r3 = 1.5 - a * (1.0 - b), 2.25 - a * (1.0 - b * b), 2.625 - a * (1.0 - b ** 3)
        return r1, r2, r3

    def f(x):
        a, b = x[0::2], x[1::2]
        r1, r2, r3 = parts(a, b)
        return float(np.sum(r1 * r1 + r2 * r2 + r3 * r3))

    def g(x):
        a, b = x[0::2], x[1::2]
        r1, r2, r3 = parts(a, b)
        out = np.empty_like(x)
        out[0::2] = 2.0 * (r1 * (-(1.0 - b)) + r2 * (-(1.0 - b * b)) + r3 * (-(1.0 - b ** 3)))
        out[1::2] = 2.0 * (r1 * a + r2 * 2.0 * a * b + r3 * 3.0 * a * b * b)
        return out

    def hvp(x, v):
        a, b, va, vb = x[0::2], x[1::2], v[0::2], v[1::2]
        r1, r2, r3 = parts(a, b)
        # gradients of the residuals
        g1a, g1b = -(1.0 - b), a
        g2a, g2b = -(1.0 - b * b), 2.0 * a * b
        g3a, g3b = -(1.0 - b ** 3), 3.0 * a * b * b
        d1, d2, d3 = g1a * va + g1b * vb, g2a * va + g2b * vb, g3a * va + g3b * vb
        # Hessians of the residuals: r1_ab = 1; r2_ab = 2b, r2_bb = 2a; r3_ab = 3b^2, r3_bb = 6ab
        out = np.empty_like(x)
        out[0::2] = 2.0 * (g1a * d1 + g2a * d2 + g3a * d3
                           + r1 * vb + r2 * (2.0 * b * vb) + r3 * (3.0 * b * b * vb))
        out[1::2] = 2.0 * (g1b * d1 + g2b * d2 + g3b * d3
                           + r1 * va + r2 * (2.0 * b * va + 2.0 * a * vb)
                           + r3 * (3.0 * b * b * va + 6.0 * a * b * vb))
        return out

    x0 = np.empty(n)
    x0[0::2], x0[1::2] = 1.0, 0.8
    return Problem(f"EXTBEALE-{n}", f, g, hvp, x0, "andrei", BUDGET_ANDREI)


def raydan1(n: int) -> Problem:
    """sum_i (i/10)(exp(x_i) - x_i), x0 = 1."""
    w = np.arange(1, n + 1, dtype=float) / 10.0

    def f(x):
        return float(np.sum(w * (np.exp(x) - x)))

    def g(x):
        return w * (np.exp(x) - 1.0)

    def hvp(x, v):
        return w * np.exp(x) * v

    return Problem(f"RAYDAN1-{n}", f, g, hvp, np.ones(n), "andrei", BUDGET_ANDREI)


def andrei_suite(n: int = 3000) -> List[Problem]:
    return [ext_himmelblau(n), eg2(n), gen_tridiag1(n), diagonal3(n), ext_beale(n), raydan1(n),
            tridia(n)]


def all_instances() -> List[Problem]:
    """The campaign of Section 7 (the undefined low-rank instance excluded)."""
    return classical_suite() + cuter_suite() + andrei_suite()


def by_name(name: str) -> Problem:
    for p in all_instances():
        if p.name == name:
            return p
    raise KeyError(name)


# ------------------------------------------------------------------- machine learning ----


def mlp_problem(d: int, h: int, N: int = 500, lam: float = 1e-3, seed_data: int = 0,
                seed_start: int = 1) -> Problem:
    """Full-batch teacher-student regression with a one-hidden-layer tanh network, eq. (mlp).

    Parameters theta = (W in R^{h x d}, b in R^h, v in R^h, c), n = h d + 2 h + 1.
    f = (1/2N) sum_i (v^T tanh(W x_i + b) + c - y_i)^2 + (lam/2) ||theta||^2.
    """
    rng = np.random.default_rng(seed_data)
    X = rng.standard_normal((N, d))
    Wt, bt, vt, ct = rng.standard_normal((h, d)), rng.standard_normal(h), rng.standard_normal(h), rng.standard_normal()
    y = np.tanh(X @ Wt.T + bt) @ vt + ct
    n = h * d + 2 * h + 1

    def unpack(th):
        W = th[: h * d].reshape(h, d)
        b = th[h * d: h * d + h]
        v = th[h * d + h: h * d + 2 * h]
        c = th[-1]
        return W, b, v, c

    def f(th):
        W, b, v, c = unpack(th)
        A = np.tanh(X @ W.T + b)
        e = A @ v + c - y
        return float(0.5 * np.mean(e * e) + 0.5 * lam * (th @ th))

    def g(th):
        W, b, v, c = unpack(th)
        A = np.tanh(X @ W.T + b)
        e = A @ v + c - y
        S = 1.0 - A * A
        Wg = (S * (e[:, None] * v[None, :])).T @ X / N       # h x d
        bg = (S * e[:, None] * v[None, :]).sum(0) / N
        vg = (A * e[:, None]).sum(0) / N
        cg = np.mean(e)
        return np.concatenate([Wg.ravel(), bg, vg, [cg]]) + lam * th

    def hvp(th, dth):
        W, b, v, c = unpack(th)
        dW, db, dv, dc = unpack(dth)
        Z = X @ W.T + b
        A = np.tanh(Z)
        S = 1.0 - A * A
        e = A @ v + c - y
        dZ = X @ dW.T + db
        dA = S * dZ
        de = A @ dv + dA @ v + dc
        dS = -2.0 * A * dA
        Wv = e[:, None] * v[None, :]                          # N x h
        dWv = de[:, None] * v[None, :] + e[:, None] * dv[None, :]
        M = S * Wv                                            # N x h, = e_i w_i
        dM = dS * Wv + S * dWv
        Wh = dM.T @ X / N
        bh = dM.sum(0) / N
        vh = (dA * e[:, None] + A * de[:, None]).sum(0) / N
        ch = np.mean(de)
        return np.concatenate([Wh.ravel(), bh, vh, [ch]]) + lam * dth

    th0 = np.random.default_rng(seed_start).standard_normal(n)
    ybar = float(np.mean(y))
    saddle = np.zeros(n)
    saddle[-1] = ybar / (1.0 + lam)
    return Problem(f"MLP-{n}", f, g, hvp, th0, "mlp", BUDGET_MLP,
                   extra={"d": d, "h": h, "N": N, "lam": lam, "saddle": saddle,
                          "seed_data": seed_data, "seed_start": seed_start})
