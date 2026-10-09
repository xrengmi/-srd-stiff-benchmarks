"""Matris-serbest doğrusal operatörler."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from srdstiff.core.types import LinearOperator, Objective, Vector


@dataclass(slots=True)
class RegularizedHessian:
    """M = H(x) + α·I operatörü. Hiçbir zaman tam H kurulmaz."""

    objective: Objective
    x: Vector
    alpha: float
    n: int = 0
    _nhv_count: int = 0

    def __post_init__(self) -> None:
        """Boyut tutarlılığı kontrolü."""
        self.n = self.objective.n
        assert self.x.shape == (self.n,), f"x boyutu {self.x.shape} ≠ ({self.n},)"
        assert self.alpha >= 0, f"alpha negatif: {self.alpha}"

    def apply(self, v: Vector) -> Vector:
        """M·v = H·v + α·v. 1 Hessian-vektör çarpımı sayılır."""
        assert v.shape == (self.n,), f"v boyutu {v.shape} ≠ ({self.n},)"
        hv = self.objective.hess_vec(self.x, v)
        self._nhv_count += 1
        return hv + self.alpha * v

    @property
    def nhv_count(self) -> int:
        """Yapılan Hessian-vektör çarpımı sayısı."""
        return self._nhv_count


@dataclass(slots=True)
class FiniteDifferenceHessian:
    """∇²f·v ≈ (∇f(x+ε·v) − ∇f(x−ε·v)) / (2ε) — merkezi fark."""

    objective: Objective
    x: Vector
    eps: float = 1e-7
    n: int = 0
    _ngv_count: int = 0

    def __post_init__(self) -> None:
        self.n = self.objective.n
        assert self.eps > 0

    def apply(self, v: Vector) -> Vector:
        """2 gradyan çağrısı ile Hessian-vektör çarpımı."""
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return np.zeros_like(v)
        h = self.eps * max(1.0, np.linalg.norm(self.x)) / v_norm
        g_plus = self.objective.grad(self.x + h * v)
        g_minus = self.objective.grad(self.x - h * v)
        self._ngv_count += 2
        return (g_plus - g_minus) / (2 * h)
