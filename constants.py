"""Constants quoted in the gain-trade-off remark (Section 2.4) and the third-derivative-omission
remark (Section 5.3) of the manuscript, for extended Rosenbrock at n = 100.

At the starting point u0 = (-1.2, 1, -1.2, 1, ...) the script evaluates

* ||grad f(u0)||_2;
* the activation threshold eps = eps_factor ||H(u0)||_2 exactly as SS-INK computes it (eight
  seeded power iterations, srdssink.ssink._spectral_scale), and checks it against the value
  stored with the SS-INK run in results/main.json;
* the norm ||nabla^3 f(u0)||_2 of the third-derivative trilinear form.  The objective is a sum of
  two-variable terms 100 (b - a^2)^2 + (1 - a)^2, whose only non-zero third derivatives are
  f_aaa = 2400 a and f_aab = -400, so the form is block diagonal over the pairs (a, b).  For a
  symmetric trilinear form on a Euclidean space the operator norm equals max_{||v||=1} |T(v,v,v)|
  (Banach's theorem on symmetric multilinear forms), and for a block-diagonal cubic form this
  maximum is attained inside a single block, since sum_i ||v_i||^3 |c_i| <= max_i |c_i| for
  sum_i ||v_i||^2 = 1.  The two-dimensional maximum is computed on a grid of 2e6 angles and
  refined by golden-section search; the value is cross-checked against the upper bound
  sup_{||v||=||w||=1} ||T[v, w, .]||_2 evaluated on the same grid (the two must agree);
* the lower bound gamma + 4 ||nabla^3 f(u0)||_2 ||grad f(u0)||_2 / eps^2 on the gain threshold
  gamma + L_H G_max / (delta (eps - delta)) of eq. (gain_dominance): L_H >= ||nabla^3 f(u0)||_2 and
  G_max >= ||grad f(u0)||_2 because u0 is in Omega_0, and delta (eps - delta) <= eps^2 / 4 with
  equality at delta = eps / 2, the buffer SS-INK uses (delta_factor = 0.5).

Run as ``python -m srdssink.constants`` from the package root; writes results/constants.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .oracle import Oracle
from .problems import rosenbrock
from .ssink import _spectral_scale

__all__ = ["run"]

N_ANGLES = 2_000_000
EPS_FACTOR = 1e-2          # default of srdssink.ssink.ssink
GAMMA = 1.0                # default of srdssink.ssink.ssink
DELTA_FACTOR = 0.5         # default of srdssink.ssink.ssink


def _cubic(a: float, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """T(v, v, v) for one pair with v = (c, s): f_aaa c^3 + 3 f_aab c^2 s."""
    return 2400.0 * a * c ** 3 + 3.0 * (-400.0) * c * c * s


def _pair_norm(a: float) -> dict:
    th = np.linspace(0.0, 2.0 * np.pi, N_ANGLES, endpoint=False)
    vals = np.abs(_cubic(a, np.cos(th), np.sin(th)))
    k = int(np.argmax(vals))
    # golden-section refinement on the bracket around the grid maximum
    lo, hi = th[k] - 2 * np.pi / N_ANGLES, th[k] + 2 * np.pi / N_ANGLES
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    F = lambda t: abs(float(_cubic(a, np.cos(t), np.sin(t))))
    for _ in range(200):
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if F(m1) > F(m2):
            hi = m2
        else:
            lo = m1
    sym_max = F(0.5 * (lo + hi))
    # upper bound: sup over unit v of the spectral norm of the 2x2 matrix T[v, ., .]
    c, s = np.cos(th[::20]), np.sin(th[::20])
    taaa, taab = 2400.0 * a, -400.0
    m11, m12, m22 = taaa * c + taab * s, taab * c, np.zeros_like(c)
    tr, det = m11 + m22, m11 * m22 - m12 * m12
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    op_max = float(np.max(np.maximum(np.abs(tr / 2.0 + disc), np.abs(tr / 2.0 - disc))))
    return {"sym_max": sym_max, "matrix_slice_max": op_max, "theta": 0.5 * (lo + hi)}


def run(out_path: str = os.path.join("results", "constants.json"), verbose: bool = True) -> dict:
    p = rosenbrock(100)
    u0 = p.x0.copy()
    grad_norm = float(np.linalg.norm(p.grad(u0)))
    oracle = Oracle(p)
    eps = EPS_FACTOR * _spectral_scale(oracle, u0)
    eps_stored = None
    main_path = os.path.join("results", "main.json")
    if os.path.exists(main_path):
        with open(main_path) as fh:
            for r in json.load(fh):
                if r.get("problem") == "ROSENBROCK-100" and r.get("method") == "SS-INK":
                    eps_stored = float(r["eps"])
                    break
    pair = _pair_norm(float(u0[0]))
    third = pair["sym_max"]
    threshold = GAMMA + 4.0 * third * grad_norm / eps ** 2
    delta = DELTA_FACTOR * eps
    threshold_delta = GAMMA + third * grad_norm / (delta * (eps - delta))
    out = {"problem": p.name, "grad_norm_u0": grad_norm, "eps": eps, "eps_stored_main": eps_stored,
           "third_norm_u0": third, "third_matrix_slice_bound": pair["matrix_slice_max"],
           "gamma": GAMMA, "delta_factor": DELTA_FACTOR,
           "gain_threshold_lower_bound": threshold, "gain_threshold_at_delta_used": threshold_delta,
           "n_angles": N_ANGLES}
    if verbose:
        print(json.dumps(out, indent=1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
