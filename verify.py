"""Derivative verification of Section 7.2 of the paper.

Each problem is checked in three ways at its starting point, along seeded random directions:
  (a) the gradient against a central difference of the objective,
  (b) the Hessian-vector product against a central difference of the gradient,
  (c) the Hessian-vector product against its adjoint (symmetry),
with tolerances 1e-5, 1e-5 and 1e-9.  The difference step is t max(1, ||x||_inf), t = 1e-5.  Errors are relative: (a) |g.v - (f(x+tv)-f(x-tv))/2t| /
max(1, |g.v|); (b) ||Hv - (g(x+tv)-g(x-tv))/2t|| / max(1, ||Hv||); (c) |w.Hv - v.Hw| /
max(1, |w.Hv|).  The campaign driver refuses to run if any check fails.

Run as ``python -m srdssink.verify`` to print the largest error of each kind.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .oracle import Problem

__all__ = ["check_problem", "verify_all", "TOL"]

TOL = (1e-5, 1e-5, 1e-9)


def check_problem(p: Problem, n_dir: int = 3, seed: int = 0, t: float = 1e-5) -> Tuple[float, float, float]:
    """Largest relative errors (a), (b), (c) over ``n_dir`` seeded directions at the start."""
    rng = np.random.default_rng(seed)
    x = p.x0.copy()
    scale = max(1.0, float(np.max(np.abs(x))))
    ea = eb = ec = 0.0
    g = p.grad(x)
    for _ in range(n_dir):
        v = rng.standard_normal(x.size)
        v /= np.linalg.norm(v)
        w = rng.standard_normal(x.size)
        w /= np.linalg.norm(w)
        h = t * scale
        # (a) gradient
        fd = (p.f(x + h * v) - p.f(x - h * v)) / (2.0 * h)
        gv = float(g @ v)
        ea = max(ea, abs(gv - fd) / max(1.0, abs(gv)))
        # (b) Hessian-vector product
        Hv = p.hvp(x, v)
        fdg = (p.grad(x + h * v) - p.grad(x - h * v)) / (2.0 * h)
        eb = max(eb, float(np.linalg.norm(Hv - fdg)) / max(1.0, float(np.linalg.norm(Hv))))
        # (c) symmetry
        Hw = p.hvp(x, w)
        s1, s2 = float(w @ Hv), float(v @ Hw)
        ec = max(ec, abs(s1 - s2) / max(1.0, abs(s1)))
    return ea, eb, ec


def verify_all(problems: Iterable[Problem], tol=TOL, verbose: bool = True) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    worst = [0.0, 0.0, 0.0]
    failed = []
    for p in problems:
        errs = check_problem(p)
        out[p.name] = errs
        worst = [max(a, b) for a, b in zip(worst, errs)]
        bad = any(e > tl for e, tl in zip(errs, tol))
        if bad:
            failed.append(p.name)
        if verbose:
            print(f"{p.name:22s} grad {errs[0]:9.2e}  hvp {errs[1]:9.2e}  sym {errs[2]:9.2e}"
                  + ("  FAIL" if bad else ""))
    if verbose:
        print(f"largest errors: {worst[0]:.2e} {worst[1]:.2e} {worst[2]:.2e}  (tolerances {tol})")
    if failed:
        raise RuntimeError("derivative check failed: " + ", ".join(failed))
    return out


def save_report(out: Dict[str, Tuple[float, float, float]], path: str) -> None:
    """Write the per-problem errors and the largest error of each kind (results/verify.json)."""
    import json
    import os
    bench = {k: v for k, v in out.items() if not k.startswith("MLP-")}
    mlp = {k: v for k, v in out.items() if k.startswith("MLP-")}
    rec = {"tolerances": list(TOL), "step": 1e-5, "directions": 3, "errors": {k: list(v) for k, v in out.items()},
           "largest_benchmark": [max(v[i] for v in bench.values()) for i in range(3)] if bench else None,
           "largest_mlp": [max(v[i] for v in mlp.values()) for i in range(3)] if mlp else None}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    import os
    from .problems import all_instances, mlp_problem
    out = verify_all(all_instances() + [mlp_problem(10, 20), mlp_problem(20, 50)])
    save_report(out, os.path.join(os.environ.get("SRDSSINK_RESULTS", "results"), "verify.json"))
