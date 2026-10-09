"""Repeatability of the ARPACK eigensolve (Section 7.1, reproducibility paragraph).

The first eigensolve of SS-INK on QUARTIC-100 (seeded start vector, tolerance 1e-8, the call
made in Algorithm 3, line 1) is repeated ``REPEATS`` times in one process with identical input,
and the distinct eigenvalues returned, bit for bit, are recorded together with the number of
Hessian-vector products of each call.

Run as ``python -m srdssink.arpack_repeat`` from the package root; writes
results/arpack_repeat.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .krylov import smallest_eigenpair
from .problems import by_name

__all__ = ["run"]

REPEATS = 20
INSTANCE = "QUARTIC-100"


def run(out_path: str = os.path.join("results", "arpack_repeat.json"), verbose: bool = True) -> dict:
    p = by_name(INSTANCE)
    u0 = p.x0.copy()
    values, counts = [], []
    for _ in range(REPEATS):
        c = [0]

        def mv(w):
            c[0] += 1
            return p.hvp(u0, w)

        lam, _ = smallest_eigenpair(mv, p.n, v0=None, tol=1e-8)
        values.append(float(lam))
        counts.append(int(c[0]))
    distinct = sorted({v.hex() for v in values})
    out = {"instance": INSTANCE, "repeats": REPEATS, "tol": 1e-8,
           "distinct_values_hex": distinct, "distinct_values": [float.fromhex(h) for h in distinct],
           "n_distinct": len(distinct), "hvp_counts": sorted(set(counts)),
           "max_abs_difference": float(max(values) - min(values))}
    if verbose:
        print(json.dumps(out, indent=1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    run()
