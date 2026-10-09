"""Run records shared by SS-INK and the baselines.

A run is *solved* when ||grad f(u_k)||_2 <= tol_rel max{1, ||grad f(u_0)||_2} (Section 7.1,
"Stopping rule and budget"); every other exit is unsolved.  ``history`` lists
(cost, ||grad f||, f) after every accepted iteration and is what the convergence-history
figures plot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .oracle import Oracle, Problem

__all__ = ["Result", "Recorder"]


class Result(dict):
    """A plain dictionary; keys are documented in ``Recorder.finish``."""

    @property
    def solved(self) -> bool:
        return self.get("status") == "converged"


class Recorder:
    def __init__(self, oracle: Oracle, problem: Problem, method: str, tol_rel: float,
                 gradient_test: bool = True):
        self.oracle, self.problem, self.method, self.tol_rel = oracle, problem, method, tol_rel
        self.gradient_test = gradient_test
        self.outer = 0
        self.history = []
        self.g0: Optional[float] = None
        self.f_last: Optional[float] = None
        self.x_last: Optional[np.ndarray] = None

    def start(self, x: np.ndarray, f_value: Optional[float] = None) -> np.ndarray:
        g = self.oracle.g(x)
        self.g0 = float(np.linalg.norm(g))
        self.x_last = np.array(x, dtype=float)
        self.history.append((self.oracle.cost, self.g0, f_value))
        return g

    def converged(self, g: np.ndarray) -> bool:
        if not self.gradient_test:
            return False
        return float(np.linalg.norm(g)) <= self.tol_rel * max(1.0, self.g0)

    def log(self, x: np.ndarray, g: np.ndarray, f_value: Optional[float] = None) -> None:
        self.outer += 1
        self.x_last = np.array(x, dtype=float)
        self.f_last = f_value
        self.history.append((self.oracle.cost, float(np.linalg.norm(g)), f_value))

    def finish(self, status: str, **extra) -> Result:
        """Keys: method, problem, status ('converged' = solved), outer (NI), nfg (NFG),
        n_f, n_g, n_hvp (inner), cost, grad_norm (at exit), f (last accepted value or None),
        history, hvp_by_site, time (s), x (final iterate), plus any ``extra``."""
        r = Result(method=self.method, problem=self.problem.name, status=status,
                   outer=self.outer, nfg=self.oracle.nfg, n_f=self.oracle.n_f,
                   n_g=self.oracle.n_g, n_hvp=self.oracle.n_h, cost=self.oracle.cost,
                   grad_norm=self.history[-1][1], f=self.f_last, history=list(self.history),
                   hvp_by_site=dict(self.oracle.site_hvp), time=self.oracle.elapsed,
                   x=self.x_last)
        r.update(extra)
        return r
