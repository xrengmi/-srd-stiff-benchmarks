"""Counting oracle, cost model and budget of Section 7.1 of the paper.

Cost model (eq. cost_model):  cost = N_f + 4 N_grad + 5 N_hvp  (work units).
The budget is enforced inside the oracle: every evaluation first checks the budget, so a
single Krylov call cannot exceed it (Section 7.1, "Stopping rule and budget").  A non-finite
value raises the same exception; both are recorded as unsolved by the campaign driver.
The wall-clock limit (300 s) and the iteration limit (10^6) are safety limits of the
protocol; the wall-clock limit is enforced here as well.

Hessian-vector products are attributed to the call site named in ``Oracle.site`` (Table 8 of
the paper): "scale", "lanczos", "residual", "p-solve", "third", "Hd", "damping".
Evaluations made with ``Oracle.uncounted()`` (the eigensolves at the starting point and at exit
reported in Tables 4 and 9, "not charged to any budget") are not counted and not attributed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["BudgetExceeded", "Problem", "Oracle", "W_F", "W_G", "W_H"]

W_F, W_G, W_H = 1.0, 4.0, 5.0   # eq. (cost_model): f = 1, gradient = 4, Hessian-vector product = 5


class BudgetExceeded(Exception):
    """Raised by the oracle when the cost budget, the wall-clock limit or finiteness is violated."""


class Problem:
    """A test problem with analytic objective, gradient and Hessian-vector product."""

    def __init__(self, name: str, f: Callable, grad: Callable, hvp: Callable, x0: np.ndarray,
                 group: str = "", budget: float = 3.0e5, extra: Optional[dict] = None):
        self.name = name
        self._f, self._g, self._hvp = f, grad, hvp
        self.x0 = np.asarray(x0, dtype=float)
        self.n = self.x0.size
        self.group = group
        self.budget = float(budget)
        self.extra = dict(extra or {})

    # raw (uncounted) access, used by the derivative checks and by the diagnostic eigensolves
    def f(self, x):
        return float(self._f(np.asarray(x, dtype=float)))

    def grad(self, x):
        return np.asarray(self._g(np.asarray(x, dtype=float)), dtype=float)

    def hvp(self, x, v):
        return np.asarray(self._hvp(np.asarray(x, dtype=float), np.asarray(v, dtype=float)), dtype=float)


class Oracle:
    """Counting oracle with the cost model of eq. (cost_model) and an interruptible budget."""

    def __init__(self, problem: Problem, budget: Optional[float] = None, time_limit: float = 300.0):
        self.problem = problem
        self.budget = float(problem.budget if budget is None else budget)
        self.time_limit = float(time_limit)
        self.n_f = self.n_g = self.n_h = 0
        self.site = "other"
        self.site_hvp: Dict[str, int] = {}
        self.t0 = time.perf_counter()
        self._counting = True
        self.exhausted: Optional[str] = None

    # ------------------------------------------------------------------ bookkeeping ---
    @property
    def cost(self) -> float:
        return W_F * self.n_f + W_G * self.n_g + W_H * self.n_h

    @property
    def nfg(self) -> int:
        """NFG of the paper: objective evaluations plus gradient evaluations."""
        return self.n_f + self.n_g

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @contextmanager
    def uncounted(self):
        """Evaluations inside this block are not charged (diagnostic eigensolves)."""
        saved = self._counting
        self._counting = False
        try:
            yield
        finally:
            self._counting = saved

    def _charge(self, kind: str) -> None:
        if not self._counting:
            return
        weight = {"f": W_F, "g": W_G}.get(kind, W_H)
        if self.cost + weight > self.budget:      # the evaluation would exceed the budget
            self.exhausted = "budget"
            raise BudgetExceeded("budget")
        if self.elapsed > self.time_limit:
            self.exhausted = "time"
            raise BudgetExceeded("time")
        if kind == "f":
            self.n_f += 1
        elif kind == "g":
            self.n_g += 1
        else:
            self.n_h += 1
            self.site_hvp[self.site] = self.site_hvp.get(self.site, 0) + 1

    def _check(self, value):
        if not np.all(np.isfinite(value)):
            self.exhausted = "non-finite"
            raise BudgetExceeded("non-finite")
        return value

    # ------------------------------------------------------------------ evaluations ---
    def f(self, x: np.ndarray) -> float:
        self._charge("f")
        return float(self._check(np.asarray(self.problem._f(x), dtype=float)))

    def g(self, x: np.ndarray) -> np.ndarray:
        self._charge("g")
        return self._check(np.asarray(self.problem._g(x), dtype=float))

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        self._charge("h")
        return self._check(np.asarray(self.problem._hvp(x, v), dtype=float))
