"""CUTEst problem yükleyici — pycutest arayüzü."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srdstiff.core.types import Objective, Vector


@dataclass(slots=True)
class CUTEstProblem:
    """CUTEst problem sarıcısı — Objective protokolünü uygular."""

    name: str
    _problem: object = None
    n: int = 0
    _x_init: Vector = None

    def __post_init__(self) -> None:
        """Problemi başlat."""
        try:
            import pycutest
        except ImportError as e:
            raise ImportError(
                "pycutest kurulu değil. Docker konteynerini kullanın."
            ) from e

        if not pycutest.problem_properties(self.name)["unconstrained"]:
            raise ValueError(f"{self.name} kısıtsız değil")

        self._problem = pycutest.import_problem(self.name)
        self.n = self._problem.n
        self._x_init = np.asarray(self._problem.x0, dtype=np.float64)

    @property
    def x_init(self) -> Vector:
        return self._x_init.copy()

    def f(self, x: Vector) -> float:
        """f(x) — CUTEst üzerinden."""
        return float(self._problem.obj(x))

    def grad(self, x: Vector) -> Vector:
        """∇f(x) — CUTEst analitik gradyan."""
        _, g = self._problem.obj(x, gradient=True)
        return np.asarray(g, dtype=np.float64)

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        """∇²f(x)·v — CUTEst Hessian-vektör çarpımı."""
        return np.asarray(self._problem.hprod(v, x=x), dtype=np.float64)


def load_cutest_problem(name: str) -> Objective:
    """Problem adıyla yükle."""
    from srdstiff.problems.classic_problems import PROBLEMS
    
    if name in PROBLEMS:
        return PROBLEMS[name]()
        
    return CUTEstProblem(name=name)


def list_available_problems(category: str | None = None) -> list[str]:
    """Mevcut CUTEst problemlerini listele."""
    from srdstiff.problems.classic_problems import PROBLEMS
    
    try:
        import pycutest
        all_problems = pycutest.find_problems(
            constraints="unconstrained",
            regular=True,
        )
        return sorted(list(set(all_problems) | set(PROBLEMS.keys())))
    except ImportError:
        return sorted(list(PROBLEMS.keys()))
