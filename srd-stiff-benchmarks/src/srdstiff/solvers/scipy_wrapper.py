"""SciPy solvers wrap'i — benchmark arayüzüne uyum sağlamak için."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize, Bounds

from srdstiff.core.types import (
    ConvergenceStatus,
    FailureMode,
    Objective,
    SolverConfig,
    SolverResult,
    Vector,
)
from srdstiff.solvers.base import BaseSolver


class ScipySolverWrapper(BaseSolver):
    """scipy.optimize.minimize için genel sarmalayıcı."""

    def __init__(self, config: SolverConfig, method: str) -> None:
        super().__init__(config)
        self.method = method
        self.name = method  # "L-BFGS-B" vs "Newton-CG"
        self._cfg = config

    def _solve_impl(self, objective: Objective, x0: Vector) -> SolverResult:
        cfg = self.config
        
        # Sayaçları scipy içinde de takip etmek için wrapper fonksiyonlar
        def obj_f(x):
            if self._check_budget_exceeded():
                raise TimeoutError("Budget exceeded")
            val = objective.f(x)
            self._nfev += 1
            return val

        def obj_g(x):
            if self._check_budget_exceeded():
                raise TimeoutError("Budget exceeded")
            grad = objective.grad(x)
            self._ngev += 1
            return grad

        def obj_h(x):
            if self._check_budget_exceeded():
                raise TimeoutError("Budget exceeded")
            hess = objective.hess(x)
            self._nhev += 1
            return hess
        
        def obj_hessp(x, p):
            if self._check_budget_exceeded():
                raise TimeoutError("Budget exceeded")
            hp = objective.hvp(x, p)
            self._nhev += 1
            return hp

        jac = obj_g
        
        if self.method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-exact"]:
            hess = obj_hessp if self.method != "trust-exact" else obj_h
            options = {"maxiter": cfg.max_iter}
        elif self.method in ["L-BFGS-B", "CG", "BFGS"]:
            hess = None
            options = {"maxiter": cfg.max_iter, "ftol": 1e-12, "gtol": cfg.tol_grad_abs}
        else:
            hess = None
            options = {"maxiter": cfg.max_iter}

        try:
            res = minimize(
                obj_f,
                x0,
                method=self.method,
                jac=jac,
                hessp=hess if hess != obj_h else None,
                hess=hess if hess == obj_h else None,
                options=options,
            )
            
            x_final = res.x
            f_final = res.fun
            g_final = objective.grad(x_final)
            grad_norm = float(np.linalg.norm(g_final))
            
            status = ConvergenceStatus.CONVERGED_GRAD if res.success else ConvergenceStatus.MAX_ITER
            if "Budget exceeded" in str(res.message):
                 status = ConvergenceStatus.MAX_TIME
                 
            return SolverResult(
                x_final=x_final,
                f_final=float(f_final),
                grad_norm_final=float(grad_norm),
                status=status,
                iterations=res.nit if hasattr(res, "nit") else 0,
                nfev=self._nfev,
                ngev=self._ngev,
                nhev=self._nhev,
                time_total=self._elapsed(),
                history=[],
                solver_name=self.name,
                problem_name=objective.name,
                seed=self.config.seed,
            )
        except TimeoutError:
            # Yakalanan timeout hatası
            return self._timeout_result(objective, x0, float("nan"), float("nan"), 0, [], ConvergenceStatus.MAX_TIME, "n/a")
        except Exception as e:
            return self._failure_result(objective, FailureMode.NUMERICAL_CRASH, str(e))
