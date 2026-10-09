"""Tip tanımları — tüm paketin tip güvenliğinin temeli."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Scalar = np.float64


class ConvergenceStatus(str, Enum):
    """Dur(dur)ma nedeni — tüm solver'larda ortak."""

    CONVERGED_GRAD = "converged_grad"
    CONVERGED_FUNC = "converged_func"
    MAX_ITER = "max_iter"
    MAX_TIME = "max_time"
    MAX_NFEV = "max_nfev"
    NUMERICAL_FAILURE = "numerical_failure"
    STEP_TOO_SMALL = "step_too_small"
    DIVERGED = "diverged"
    USER_INTERRUPT = "user_interrupt"


class FailureMode(str, Enum):
    """Hakem Metrik R3: başarısızlık sınıflandırması."""

    SLOW_CONVERGENCE = "slow_convergence"
    STAGNATION = "stagnation"
    DIVERGENCE = "divergence"
    NUMERICAL_CRASH = "numerical_crash"
    TIME_LIMIT = "time_limit"
    MEMORY_LIMIT = "memory_limit"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Solver konfigürasyonu — hepsi için ortak taban."""

    tol_grad_rel: float = 1e-6
    tol_grad_abs: float = 1e-8
    max_iter: int = 10_000
    max_time_sec: float = 3600.0
    max_nfev: int = 50_000
    seed: int = 42

    def __post_init__(self) -> None:
        """Değerleri sağlamalaştır."""
        assert self.tol_grad_rel > 0, "tol_grad_rel pozitif olmalı"
        assert self.tol_grad_abs > 0, "tol_grad_abs pozitif olmalı"
        assert self.max_iter > 0, "max_iter pozitif olmalı"
        assert self.max_time_sec > 0, "max_time_sec pozitif olmalı"


@dataclass(frozen=True, slots=True)
class SRDStiffConfig(SolverConfig):
    """SRD-STIFF algoritmasına özgü parametreler."""

    K: float = 10.0
    gamma: float = 0.5
    epsilon: float = 1e-3
    delta: float = 1e-4
    alpha_init: float = 1.0
    c1_armijo: float = 1e-4
    tau_backtrack: float = 0.5
    h_max: float = 1.0
    h_min: float = 1e-8
    eta_cg: float = 0.1
    max_cg_iter: int = 1000
    m_lanczos: int = 20
    L_H_estimate: float = 1.0

    def __post_init__(self) -> None:
        """Kazanım koşulunu doğrula (Prop 4.3)."""
        SolverConfig.__post_init__(self)
        assert self.K > 0 and self.gamma > 0, "K, γ pozitif olmalı"
        assert 0 < self.delta < self.epsilon, "0 < δ < ε gerekli"
        assert 0 < self.c1_armijo < 1, "c₁ ∈ (0,1) gerekli"
        assert 0 < self.tau_backtrack < 1, "τ ∈ (0,1) gerekli"
        assert 0 < self.eta_cg < 1, "η_CG ∈ (0,1) gerekli"
        assert self.m_lanczos >= 2, "m_Lanczos ≥ 2 gerekli"


@dataclass(slots=True)
class IterationRecord:
    """Tek iterasyonun tam kaydı — makine okunabilir."""

    k: int
    f_val: float
    grad_norm: float
    alpha: float
    lam_min_lb: float
    step_size: float
    cg_iters: int
    nfev: int
    ngev: int
    nhev: int
    time_elapsed: float
    safe_set_ok: bool

    def __post_init__(self) -> None:
        """Sağlık kontrolleri."""
        assert np.isfinite(self.f_val), f"f_val sonsuz veya NaN: {self.f_val}"
        assert self.grad_norm >= 0, f"grad_norm negatif: {self.grad_norm}"
        assert self.alpha >= 0, f"alpha negatif: {self.alpha}"


@dataclass(slots=True)
class SolverResult:
    """Solver çıktısı — tüm meta-veriler dahil."""

    x_final: Vector
    f_final: float
    grad_norm_final: float
    status: ConvergenceStatus
    iterations: int
    nfev: int
    ngev: int
    nhev: int
    time_total: float
    history: list[IterationRecord] = field(default_factory=list)
    failure_mode: FailureMode | None = None
    solver_name: str = ""
    problem_name: str = ""
    seed: int = 0
    config_hash: str = ""

    def is_converged(self) -> bool:
        """Yakınsama kontrolü."""
        return self.status in (
            ConvergenceStatus.CONVERGED_GRAD,
            ConvergenceStatus.CONVERGED_FUNC,
        )

    def to_dict(self) -> dict:
        """JSON-serializable sözlüğe dönüştür."""
        return {
            "x_final": self.x_final.tolist(),
            "f_final": float(self.f_final),
            "grad_norm_final": float(self.grad_norm_final),
            "status": self.status.value,
            "iterations": self.iterations,
            "nfev": self.nfev,
            "ngev": self.ngev,
            "nhev": self.nhev,
            "time_total": self.time_total,
            "history": [
                {
                    "k": r.k,
                    "f_val": float(r.f_val),
                    "grad_norm": float(r.grad_norm),
                    "alpha": float(r.alpha),
                    "lam_min_lb": float(r.lam_min_lb),
                    "step_size": float(r.step_size),
                    "cg_iters": r.cg_iters,
                    "nfev": r.nfev,
                    "ngev": r.ngev,
                    "nhev": r.nhev,
                    "time_elapsed": r.time_elapsed,
                    "safe_set_ok": r.safe_set_ok,
                }
                for r in self.history
            ],
            "failure_mode": self.failure_mode.value if self.failure_mode else None,
            "solver_name": self.solver_name,
            "problem_name": self.problem_name,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }


@runtime_checkable
class LinearOperator(Protocol):
    """Matris-serbest doğrusal operatör arayüzü."""

    n: int

    def apply(self, v: Vector) -> Vector:
        """M·v çarpımını hesapla."""
        ...


@runtime_checkable
class Objective(Protocol):
    """Amaç fonksiyonu oracle arayüzü."""

    n: int
    name: str

    def f(self, x: Vector) -> float:
        """f(x) değerini hesapla."""
        ...

    def grad(self, x: Vector) -> Vector:
        """∇f(x) hesapla."""
        ...

    def hess_vec(self, x: Vector, v: Vector) -> Vector:
        """∇²f(x)·v çarpımını hesapla."""
        ...

    @property
    def x_init(self) -> Vector:
        """Başlangıç noktası."""
        ...
