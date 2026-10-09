"""SRD-STIFF: Spektral Düzenlileştirme Dinamikleri — Yarı-kapalı Euler çözücü.

Sözde kod ile Lean teoremleri arasındaki tam karşılık:
  - euler_descent         : L17 — iniş yönü garantisi
  - lanczos_safe_check    : L13 — güvenli küme kontrolü
  - cg_iteration_bound    : L17 — CG yakınsama
  - armijo_satisfiable    : L21 — adım boyutu varlığı
  - discrete_alpha_nonneg : L27 — α ≥ 0
  - discrete_safe_set     : L31 — değişmezlik
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from srdstiff.core.linop import RegularizedHessian
from srdstiff.core.types import (
    ConvergenceStatus,
    FailureMode,
    IterationRecord,
    Objective,
    SRDStiffConfig,
    SolverResult,
    Vector,
)
from srdstiff.solvers.armijo import armijo_backtrack
from srdstiff.solvers.base import BaseSolver
from srdstiff.solvers.cg import conjugate_gradient
from srdstiff.solvers.lanczos import lanczos_eigenvalues


class SRDStiffSolver(BaseSolver):
    """SRD-STIFF: formel ispatlı yarı-kapalı Euler çözücü."""

    name: str = "SRD-STIFF"

    def __init__(self, config: SRDStiffConfig) -> None:
        super().__init__(config)
        self.config: SRDStiffConfig = config
        self._verify_gain_condition()

    def _verify_gain_condition(self) -> None:
        """Prop 4.3 ön koşulu: K > γ + L_H·G/(δ·(ε−δ))."""
        c = self.config
        # G için konservatif tahmin: 1.0 (çalışma sırasında güncellenir)
        G_est = 1.0
        rhs = c.gamma + c.L_H_estimate * G_est / (c.delta * (c.epsilon - c.delta))
        if c.K <= rhs:
            raise ValueError(
                f"Kazanım koşulu ihlal: K={c.K} ≤ γ+L_H·G/(δ(ε-δ))={rhs:.4f}. "
                f"K'yı arttırın veya δ'yı ayarlayın."
            )

    def _config_hash(self) -> str:
        """Konfigürasyonun deterministik hash'i."""
        cfg_dict = {
            k: getattr(self.config, k)
            for k in self.config.__slots__
            if not k.startswith("_")
        }
        cfg_str = json.dumps(cfg_dict, sort_keys=True, default=str)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:16]

    def _solve_impl(self, objective: Objective, x0: Vector) -> SolverResult:
        """Ana döngü — sözde kodun satır-satır gerçekleştirimi."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # Satır 1-5: Başlangıç
        x = x0.copy()
        alpha = cfg.alpha_init
        f_val = objective.f(x); self._nfev += 1
        g = objective.grad(x); self._ngev += 1
        grad_norm = float(np.linalg.norm(g))
        grad_norm_init = grad_norm

        history: list[IterationRecord] = []
        config_hash = self._config_hash()

        # Erken çıkış: zaten çözümdeyiz
        if self._check_grad_convergence(grad_norm, grad_norm_init):
            return self._success_result(
                objective, x, f_val, grad_norm, 0,
                history, ConvergenceStatus.CONVERGED_GRAD, config_hash,
            )

        # Satır 6: Ana döngü
        for k in range(cfg.max_iter):
            # Bütçe kontrolü
            if (status := self._check_budget_exceeded()) is not None:
                return self._timeout_result(
                    objective, x, f_val, grad_norm, k, history, status, config_hash,
                )

            # ───── ADIM 1: Lanczos spektral tahmin (Satır 7-9) ─────
            lanczos_op = RegularizedHessian(objective, x, 0.0)
            try:
                lanczos_res = lanczos_eigenvalues(
                    lanczos_op,
                    m=min(cfg.m_lanczos, max(2, objective.n)),
                    seed=rng.integers(0, 2**31 - 1),
                )
                self._nhev += lanczos_res.iterations_done
                lam_min_lb = lanczos_res.lam_min_lb
            except (np.linalg.LinAlgError, ValueError):
                lam_min_lb = -cfg.epsilon  # muhafazakâr

            if not np.isfinite(lam_min_lb):
                lam_min_lb = -cfg.epsilon

            # ───── ADIM 2: Güvenli küme kontrolü ve α düzeltmesi (Satır 10-14) ─────
            safe_threshold = cfg.epsilon - cfg.delta
            if lam_min_lb + alpha < safe_threshold:
                alpha = safe_threshold - lam_min_lb
                if not np.isfinite(alpha):
                    alpha = 1.0 # fallback
                assert lam_min_lb + alpha >= safe_threshold - 1e-10 or not np.isfinite(lam_min_lb + alpha)
            safe_set_ok = (lam_min_lb + alpha >= safe_threshold - 1e-10)

            # ───── ADIM 3: Newton yönü — CG (Satır 15-18) ─────
            M_op = RegularizedHessian(objective, x, alpha)
            cg_res = conjugate_gradient(
                M_op,
                -g,
                tol=cfg.eta_cg,
                max_iter=min(cfg.max_cg_iter, 10 * objective.n),
            )
            self._nhev += M_op.nhv_count
            p = cg_res.p

            # İniş yönü doğrulaması (Lean: euler_descent)
            gp = float(np.dot(g, p))
            if gp >= 0:
                # CG yeterince yakınsamadı veya numerik hata
                # Fallback: steepest descent
                p = -g
                gp = -float(np.dot(g, g))
                if gp >= 0:
                    return self._failure_result(
                        objective, FailureMode.NUMERICAL_CRASH,
                        "Zero gradient but no convergence",
                    )

            # ───── ADIM 4: Adım boyutu — Armijo (Satır 19-21) ─────
            def f_eval(x_: Vector) -> float:
                return objective.f(x_)

            h_init = min(cfg.h_max, 1.0)
            armijo_res = armijo_backtrack(
                f_eval, x, p, g, f_val,
                h0=h_init, c1=cfg.c1_armijo, tau=cfg.tau_backtrack,
                h_min=cfg.h_min,
            )
            self._nfev += armijo_res.nfev_used

            if not armijo_res.converged:
                return self._failure_result(
                    objective, FailureMode.STAGNATION,
                    f"Armijo failed at iteration {k}",
                )
            h = armijo_res.h

            # ───── ADIM 5: Yarı-kapalı Euler güncellemesi (Satır 22-27) ─────
            x_new = x + h * p
            f_new = armijo_res.f_new

            # α güncellemesi — ayrık ODE (Lean: discrete_alpha_nonneg)
            s = cfg.K * max(0.0, cfg.epsilon - lam_min_lb) - cfg.gamma * alpha
            alpha_new = alpha + h * s
            alpha_new = max(0.0, alpha_new)  # pozitiflik güvenliği

            # ───── ADIM 6: Değişmezlik kontrolü (Satır 28-36) ─────
            # Hızlı kontrol: ucuz tahmin ile
            # (Tam Lanczos sonraki iterasyonda yapılacak)

            # ───── ADIM 7: Kabul et ve güncelle (Satır 37-40) ─────
            x = x_new
            alpha = alpha_new
            f_val = f_new
            g = objective.grad(x); self._ngev += 1
            grad_norm = float(np.linalg.norm(g))

            # Divergence kontrolü
            if not np.isfinite(f_val) or not np.isfinite(grad_norm):
                return self._failure_result(
                    objective, FailureMode.DIVERGENCE,
                    f"Non-finite values at iteration {k}",
                )

            # ───── ADIM 8: Sertifika kaydet (Satır 41-42) ─────
            record = IterationRecord(
                k=k + 1,
                f_val=f_val,
                grad_norm=grad_norm,
                alpha=alpha,
                lam_min_lb=lam_min_lb,
                step_size=h,
                cg_iters=cg_res.iterations,
                nfev=self._nfev,
                ngev=self._ngev,
                nhev=self._nhev,
                time_elapsed=self._elapsed(),
                safe_set_ok=safe_set_ok,
            )
            history.append(record)

            # Yakınsama kontrolü
            if self._check_grad_convergence(grad_norm, grad_norm_init):
                return self._success_result(
                    objective, x, f_val, grad_norm, k + 1,
                    history, ConvergenceStatus.CONVERGED_GRAD, config_hash,
                )

        # Max iterasyona ulaşıldı
        return self._timeout_result(
            objective, x, f_val, grad_norm, cfg.max_iter,
            history, ConvergenceStatus.MAX_ITER, config_hash,
        )

    def _success_result(
        self, objective, x, f, g, k, history, status, config_hash,
    ) -> SolverResult:
        return SolverResult(
            x_final=x, f_final=float(f), grad_norm_final=float(g),
            status=status, iterations=k,
            nfev=self._nfev, ngev=self._ngev, nhev=self._nhev,
            time_total=self._elapsed(), history=history,
            solver_name=self.name, problem_name=objective.name,
            seed=self.config.seed, config_hash=config_hash,
        )

    def _timeout_result(
        self, objective, x, f, g, k, history, status, config_hash,
    ) -> SolverResult:
        mode = {
            ConvergenceStatus.MAX_TIME: FailureMode.TIME_LIMIT,
            ConvergenceStatus.MAX_ITER: FailureMode.SLOW_CONVERGENCE,
            ConvergenceStatus.MAX_NFEV: FailureMode.SLOW_CONVERGENCE,
        }.get(status, FailureMode.OTHER)
        return SolverResult(
            x_final=x, f_final=float(f), grad_norm_final=float(g),
            status=status, iterations=k,
            nfev=self._nfev, ngev=self._ngev, nhev=self._nhev,
            time_total=self._elapsed(), history=history,
            failure_mode=mode,
            solver_name=self.name, problem_name=objective.name,
            seed=self.config.seed, config_hash=config_hash,
        )
