"""Tüm solver'lar için soyut taban sınıf."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from srdstiff.core.types import (
    ConvergenceStatus,
    FailureMode,
    Objective,
    SolverConfig,
    SolverResult,
    Vector,
)


class BaseSolver(ABC):
    """Soyut solver taban sınıfı — tüm karşılaştırmalar için ortak arayüz."""

    name: str = "base"

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self._nfev = 0
        self._ngev = 0
        self._nhev = 0
        self._start_time = 0.0

    @abstractmethod
    def _solve_impl(self, objective: Objective, x0: Vector) -> SolverResult:
        """Alt sınıf tarafından uygulanacak gerçek çözüm."""
        ...

    def solve(self, objective: Objective) -> SolverResult:
        """Standart arayüz."""
        self._reset_counters()
        self._start_time = time.perf_counter()
        try:
            return self._solve_impl(objective, objective.x_init.copy())
        except (OverflowError, FloatingPointError, np.linalg.LinAlgError) as e:
            return self._failure_result(
                objective, FailureMode.NUMERICAL_CRASH, str(e)
            )

    def _reset_counters(self) -> None:
        self._nfev = 0
        self._ngev = 0
        self._nhev = 0

    def _elapsed(self) -> float:
        return time.perf_counter() - self._start_time

    def _check_grad_convergence(self, grad_norm: float, grad_norm_init: float) -> bool:
        """Gradyan yakınsama kontrolü — tüm solver'larda EŞİT."""
        rel_tol = self.config.tol_grad_rel * max(1.0, grad_norm_init)
        abs_tol = self.config.tol_grad_abs
        return grad_norm <= max(rel_tol, abs_tol)

    def _check_budget_exceeded(self) -> ConvergenceStatus | None:
        """Bütçe kontrolleri."""
        if self._elapsed() > self.config.max_time_sec:
            return ConvergenceStatus.MAX_TIME
        if self._nfev > self.config.max_nfev:
            return ConvergenceStatus.MAX_NFEV
        return None

    def _failure_result(
        self, objective: Objective, mode: FailureMode, msg: str = ""
    ) -> SolverResult:
        """Başarısızlık durumunda standart sonuç."""
        return SolverResult(
            x_final=objective.x_init.copy(),
            f_final=float("nan"),
            grad_norm_final=float("nan"),
            status=ConvergenceStatus.NUMERICAL_FAILURE,
            iterations=0,
            nfev=self._nfev,
            ngev=self._ngev,
            nhev=self._nhev,
            time_total=self._elapsed(),
            failure_mode=mode,
            solver_name=self.name,
            problem_name=objective.name,
            seed=self.config.seed,
        )
