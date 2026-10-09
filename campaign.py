"""Campaign driver for the experiments of Section 7 of the paper.

Every experiment of the section is a function here; ``python -m srdssink.campaign all``
runs them in order with checkpointing (a run already present in the results file is not
repeated).  Results are JSON files in ``results/``:

  main.json          34-instance campaign (Tables 2-5, 8, Figures 1-8)   [33 instances: the
                     low-rank instance is undefined in the manuscript]
  escape.json        saddle escape on the quartic (Table 6)
  sensitivity.json   eta_max and K/gamma on extended Rosenbrock n = 100 (Table 7)
  cosine_trunc.json  SS-INK-2 with the truncated inner solve on COSINE (Section 7.3)
  mlp.json           machine-learning instance, main runs and variants (Table 9)
  mlp_escape.json    escape from the symmetric saddle of the network (Table 10)
  starts.json        lambda_min(H(u_0)) of every instance, eigensolve to 1e-10, uncounted

The derivative checks of ``srdssink.verify`` are run first; the driver stops if one fails.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from .baselines import BASELINES
from .krylov import smallest_eigenpair
from .oracle import Oracle, Problem
from .problems import all_instances, by_name, mlp_problem, quartic
from .ssink import ssink

__all__ = ["run_one", "run_campaign", "METHODS"]

METHODS = ["SS-INK", "SS-INK-2", "ARC", "TR-NCG", "NCG", "RegN", "L-BFGS", "Adam", "PGD", "TN", "INB"]
RESULTS_DIR = os.environ.get("SRDSSINK_RESULTS", "results")


def _solver(method: str):
    if method == "SS-INK":
        return lambda o, p, x0, **kw: ssink(o, p, x0, **kw)
    if method == "SS-INK-2":
        return lambda o, p, x0, **kw: ssink(o, p, x0, third_u=False, third_alpha=False, **kw)
    fn = BASELINES[method]
    return lambda o, p, x0, **kw: fn(o, p, x0, **kw)


def _eig_uncounted(oracle: Oracle, x: np.ndarray, tol: float = 1e-10):
    """Smallest eigenpair at x to relative tolerance ``tol``, not charged (Sections 7.2 and 7.4)."""
    with oracle.uncounted():
        return smallest_eigenpair(lambda v: oracle.hvp(x, v), x.size, tol=tol)


def _thin(history, max_points: int = 4000):
    """Keep every k-th accepted iterate (first and last always) so that stored histories
    stay bounded; the figures plot cost against ||grad f|| and f and lose nothing visible."""
    m = len(history)
    if m <= max_points:
        return history
    k = int(math.ceil((m - 1) / (max_points - 1)))
    out = history[0:m - 1:k]
    if out[-1] is not history[-1]:
        out.append(history[-1])
    return out


def run_one(problem: Problem, method: str, budget: Optional[float] = None, x0=None,
            gradient_test: bool = True, exit_eig: bool = True, **kw) -> Dict:
    """One run; returns a JSON-serializable record."""
    oracle = Oracle(problem, budget=budget)
    x0 = problem.x0 if x0 is None else np.asarray(x0, dtype=float)
    res = _solver(method)(oracle, problem, x0, gradient_test=gradient_test, **kw)
    rec = {k: v for k, v in res.items() if k not in ("x", "history")}
    rec["history"] = _thin([(float(c), float(g), (None if f is None else float(f)))
                            for c, g, f in res["history"]])
    rec["history_len"] = len(res["history"])
    x = res["x"]
    with oracle.uncounted():
        rec["f_exit"] = oracle.f(x)
        rec["grad_norm_exit"] = float(np.linalg.norm(oracle.g(x)))
    if exit_eig:
        lam, _ = _eig_uncounted(oracle, x)
        rec["lambda_min_exit"] = lam
    if "eta_achieved" in rec:
        # a forcing term counts as attained iff GMRES reported convergence (true residual
        # <= eta ||rhs|| within the cap) and its residual estimate is <= eta; this is the
        # same criterion as the solver's own ``forcing_missed`` counter
        rec["eta_achieved"] = [(float(t), float(a), int(i)) for t, a, i in rec["eta_achieved"]]
        rec["forcing_attained_all"] = rec.get("forcing_missed", 0) == 0
    rec["n"] = int(problem.n)
    rec["group"] = problem.group
    rec["budget"] = float(oracle.budget)
    for k, v in list(rec.items()):
        if isinstance(v, (np.floating, np.integer)):
            rec[k] = v.item()
    return rec


def _job(args):
    name, method, kind, kw = args
    if name.startswith("MLP-"):
        d, h = kw.pop("_dh")
        p = mlp_problem(d, h)
    else:
        p = by_name(name)
    x0 = kw.pop("_x0", None)
    meta = {k[5:]: kw.pop(k) for k in list(kw) if k.startswith("meta_")}
    t = time.perf_counter()
    # the exit eigensolve (to 1e-10, uncounted) is only reported for the machine-learning
    # instance (Section 7.9); on the main campaign it would cost minutes per run on the
    # n = 10000 instances with clustered spectra and is not used
    extra = {} if "exit_eig" in kw else {"exit_eig": kind not in ("main",)}
    rec = run_one(p, method, x0=x0, **extra, **kw)
    rec["wall"] = time.perf_counter() - t
    rec["kind"] = kind
    rec.update(meta)
    rec["params"] = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in kw.items()}
    return rec


def _load(path: str) -> List[Dict]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def _save(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(records, fh)
    os.replace(tmp, path)


def _key(rec: Dict) -> str:
    return f"{rec['kind']}|{rec['problem']}|{rec['method']}|{json.dumps(rec.get('params', {}), sort_keys=True)}"


def run_jobs(jobs, path: str, workers: int = 2) -> List[Dict]:
    records = _load(path)
    done = {_key(r) for r in records}
    todo = []
    for name, method, kind, kw in jobs:
        probe = {"kind": kind, "problem": name, "method": method,
                 "params": {k: v for k, v in kw.items()
                            if not k.startswith("_") and not k.startswith("meta_") and not isinstance(v, np.ndarray)}}
        if _key(probe) not in done:
            todo.append((name, method, kind, dict(kw)))
    print(f"{path}: {len(records)} done, {len(todo)} to run", flush=True)
    if not todo:
        return records
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(_job, todo):
                rec["params"] = {k: v for k, v in rec["params"].items() if not k.startswith("_")}
                records.append(rec)
                _save(path, records)
                print(f"  {rec['problem']:18s} {rec['method']:9s} {rec['status']:12s} "
                      f"NI={rec['outer']:6d} cost={rec['cost']:9.0f} wall={rec['wall']:6.1f}s", flush=True)
    else:
        for job in todo:
            rec = _job(job)
            rec["params"] = {k: v for k, v in rec["params"].items() if not k.startswith("_")}
            records.append(rec)
            _save(path, records)
            print(f"  {rec['problem']:18s} {rec['method']:9s} {rec['status']:12s} "
                  f"NI={rec['outer']:6d} cost={rec['cost']:9.0f} wall={rec['wall']:6.1f}s", flush=True)
    return records


# ---------------------------------------------------------------------------- experiments ---


def starts(path: str) -> Dict:
    """lambda_min(H(u_0)) for every instance, eigensolve to relative tolerance 1e-10 (Section 7.2),
    and the number of Hessian-vector products that the solver's own first eigensolve (tolerance
    1e-8, seeded start, Section 7.1) needs at u_0; neither is charged to any run."""
    out = _load(path) if os.path.exists(path) else []
    out = out if isinstance(out, dict) else {}
    for p in all_instances():
        if p.name in out and "lanczos_start_hvp" in out[p.name]:
            continue
        oracle = Oracle(p)
        lam, _ = _eig_uncounted(oracle, p.x0)
        with oracle.uncounted():
            g0 = float(np.linalg.norm(oracle.g(p.x0)))
            count = [0]

            def mv(v, _c=count, _o=oracle, _x=p.x0):
                _c[0] += 1
                return _o.hvp(_x, v)
            lam8, _ = smallest_eigenpair(mv, p.n, tol=1e-8)
        out[p.name] = {"lambda_min_0": lam, "grad_norm_0": g0, "n": p.n, "group": p.group,
                       "lanczos_start_hvp": count[0], "lambda_min_0_tol8": lam8}
        print(f"  {p.name:18s} lambda_min(H(u0)) = {lam:+.4e}  ||grad f(u0)|| = {g0:.4e}  "
              f"first eigensolve: {count[0]} HVPs", flush=True)
        _save(path, out)
    return out


def main_campaign(path: str, workers: int) -> List[Dict]:
    jobs = [(p.name, m, "main", {}) for p in all_instances() for m in METHODS]
    return run_jobs(jobs, path, workers)


def repeat_main(path: str, workers: int) -> List[Dict]:
    """Section 7.1 (reproducibility): a second, independent run of the whole main campaign with
    the same code and settings, compared run by run with results/main.json."""
    jobs = [(p.name, m, "repeat", {"exit_eig": False}) for p in all_instances() for m in METHODS]
    return run_jobs(jobs, path, workers)


def escape(path: str, workers: int) -> List[Dict]:
    """Table 6: start at 1e-8 v_1 on the quartic, gradient test disabled, budget 2e4."""
    jobs = []
    for n in (100, 1000):
        p = quartic(n)
        x0 = 1e-8 * p.extra["escape_direction"]
        for m in METHODS:
            jobs.append((p.name, m, "escape", {"budget": 2.0e4, "gradient_test": False,
                                                "exit_eig": False, "_x0": x0}))
    return run_jobs(jobs, path, workers)


def sensitivity(path: str, workers: int) -> List[Dict]:
    """Table 8: SS-INK on extended Rosenbrock n = 100, one parameter varied at a time."""
    jobs = []
    for eta_max in (0.9, 0.5, 0.1, 1e-3, 1e-6, 1e-10):
        jobs.append(("ROSENBROCK-100", "SS-INK", "sensitivity", {"eta_max": eta_max, "exit_eig": False}))
    for ratio in (1.2, 2.0, 5.0, 20.0):
        jobs.append(("ROSENBROCK-100", "SS-INK", "sensitivity", {"K": ratio, "gamma": 1.0, "exit_eig": False}))
    return run_jobs(jobs, path, workers)


RETRY_INSTANCES = ("ROSENBROCK-100", "QUARTIC-100", "QUARTIC-1000", "ALLENCAHN-1024")


def retry_ablation(path: str, workers: int) -> List[Dict]:
    """Section 4.3: SS-INK without the retry loop of Algorithm 2 (r_max = 1, i.e. one GMRES
    solve per inner iteration and no tightening of the forcing term), on the classical instances
    that SS-INK solves in the main campaign; all other settings are the defaults."""
    jobs = [(name, "SS-INK", "retry_ablation", {"retry_max": 1, "exit_eig": False}) for name in RETRY_INSTANCES]
    return run_jobs(jobs, path, workers)


def cosine_truncated(path: str, workers: int) -> List[Dict]:
    """Section 7.3: SS-INK-2 with the inner solve truncated at a relative reduction of 1e-2."""
    jobs = [(f"COSINE-{n}", "SS-INK-2", "cosine_trunc", {"inner_rtol": 1e-2}) for n in (1000, 10000)]
    return run_jobs(jobs, path, workers)


def mlp(path: str, workers: int) -> List[Dict]:
    """Table 9 and the variants of Section 7.9 (third-derivative terms omitted, truncated
    inner solve, eps reduced by two and four orders of magnitude at n = 241)."""
    jobs = []
    for d, h in ((10, 20), (20, 50)):
        p = mlp_problem(d, h)
        for m in METHODS:
            jobs.append((p.name, m, "mlp", {"_dh": (d, h)}))
        jobs.append((p.name, "SS-INK-2", "mlp_trunc", {"_dh": (d, h), "inner_rtol": 1e-2}))
    p = mlp_problem(10, 20)
    for factor in (1e-4, 1e-6):
        jobs.append((p.name, "SS-INK", "mlp_eps", {"_dh": (10, 20), "eps_factor": factor}))
    return run_jobs(jobs, path, workers)


def mlp_escape(path: str, workers: int) -> List[Dict]:
    """Table 10: 1e-8 v_1 away from the symmetric saddle, gradient test disabled, budget 2e4."""
    jobs = []
    for d, h in ((10, 20), (20, 50)):
        p = mlp_problem(d, h)
        oracle = Oracle(p)
        saddle = p.extra["saddle"]
        lam, v1 = _eig_uncounted(oracle, saddle)
        with oracle.uncounted():
            f_saddle = oracle.f(saddle)
            g_saddle = float(np.linalg.norm(oracle.g(saddle)))
        x0 = saddle + 1e-8 * v1
        for m in METHODS:
            jobs.append((p.name, m, "mlp_escape", {"_dh": (d, h), "budget": 2.0e4, "gradient_test": False,
                                                    "exit_eig": False, "_x0": x0,
                                                    "meta_f_saddle": f_saddle, "meta_lambda_min_saddle": lam,
                                                    "meta_grad_norm_saddle": g_saddle}))
    return run_jobs(jobs, path, workers)


def mlp_saddle_info(path: str) -> Dict:
    """lambda_min and the eigenvalue of smallest modulus at the symmetric saddle (Section 7.9)."""
    out = {}
    for d, h in ((10, 20), (20, 50)):
        p = mlp_problem(d, h)
        oracle = Oracle(p)
        saddle = p.extra["saddle"]
        n = p.n
        with oracle.uncounted():
            H = np.column_stack([oracle.hvp(saddle, np.eye(n)[:, j]) for j in range(n)])
            gnorm = float(np.linalg.norm(oracle.g(saddle)))
        w = np.linalg.eigvalsh(0.5 * (H + H.T))
        lam_min = float(w[0])
        mult = int(np.sum(np.abs(w - lam_min) <= 1e-8 * max(1.0, abs(lam_min))))
        out[p.name] = {"lambda_min": lam_min, "multiplicity": mult, "smallest_modulus": float(np.min(np.abs(w))),
                       "grad_norm": gnorm, "n": n, "d": d, "h": h}
        print(f"  {p.name}: lambda_min = {lam_min:.4g} (multiplicity {mult}), min |lambda| = {np.min(np.abs(w)):.4g}, ||grad|| = {gnorm:.2e}", flush=True)
    _save(path, out)
    return out


def run_campaign(which: str = "all", workers: int = 2) -> None:
    from .verify import save_report, verify_all
    R = RESULTS_DIR
    out = verify_all(all_instances() + [mlp_problem(10, 20), mlp_problem(20, 50)], verbose=False)
    save_report(out, os.path.join(R, "verify.json"))
    print("derivative checks passed", flush=True)
    if which in ("all", "starts"):
        starts(os.path.join(R, "starts.json"))
    if which in ("all", "main"):
        main_campaign(os.path.join(R, "main.json"), workers)
    if which in ("all", "escape"):
        escape(os.path.join(R, "escape.json"), workers)
    if which in ("all", "sensitivity"):
        sensitivity(os.path.join(R, "sensitivity.json"), workers)
    if which in ("all", "repeat"):
        repeat_main(os.path.join(R, "repeat.json"), workers)
    if which in ("all", "retry"):
        retry_ablation(os.path.join(R, "retry_ablation.json"), workers)
    if which in ("all", "cosine"):
        cosine_truncated(os.path.join(R, "cosine_trunc.json"), workers)
    if which in ("all", "mlp"):
        mlp_saddle_info(os.path.join(R, "mlp_saddle.json"))
        mlp(os.path.join(R, "mlp.json"), workers)
        mlp_escape(os.path.join(R, "mlp_escape.json"), workers)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_campaign(which, workers)
