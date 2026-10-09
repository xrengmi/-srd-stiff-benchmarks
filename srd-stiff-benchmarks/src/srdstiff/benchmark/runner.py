"""Deney yürütücü — tek (problem, solver, seed) üçlüsünü çalıştırır."""
from __future__ import annotations

import json
import platform
import socket
import subprocess
try:
    import resource
except ImportError:
    resource = None

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from srdstiff.core.types import SolverResult
from srdstiff.problems.cutest_loader import load_cutest_problem
from srdstiff.solvers.base import BaseSolver
from srdstiff.utils.logging import get_logger
from srdstiff.utils.reproducibility import capture_environment, set_all_seeds

logger = get_logger(__name__)


@dataclass(slots=True)
class RunMetadata:
    """Çalıştırma meta-verileri — reproducibility için."""

    hostname: str
    cpu_info: str
    python_version: str
    numpy_version: str
    scipy_version: str
    timestamp_utc: str
    git_commit: str
    env_hash: str
    peak_memory_mb: float


def _get_git_commit() -> str:
    """Git commit SHA'sını al."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _get_cpu_info() -> str:
    """CPU bilgisi."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except OSError:
        pass
    return platform.processor()


def _collect_metadata() -> RunMetadata:
    """Tam çevre meta-verisini topla."""
    import scipy

    env = capture_environment()
    env_json = json.dumps(env, sort_keys=True)
    import hashlib
    env_hash = hashlib.sha256(env_json.encode()).hexdigest()[:16]

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_mb = usage.ru_maxrss / 1024  # KB -> MB (Linux)
    else:
        peak_mb = 0.0


    return RunMetadata(
        hostname=socket.gethostname(),
        cpu_info=_get_cpu_info(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_get_git_commit(),
        env_hash=env_hash,
        peak_memory_mb=peak_mb,
    )


def run_single_experiment(
    problem_name: str,
    solver: BaseSolver,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Tek (problem, solver, seed) deneyini çalıştır.

    Reproducibility:
      - Tüm RNG tohumları sabitlenir
      - Çevre snapshot alınır
      - Zaman damgası + git commit kaydedilir
      - Çıktılar JSON + Parquet olarak kaydedilir
    """
    set_all_seeds(seed)
    logger.info(
        "experiment_start",
        problem=problem_name, solver=solver.name, seed=seed,
    )

    try:
        problem = load_cutest_problem(problem_name)
    except Exception as e:
        logger.error("problem_load_failed", problem=problem_name, error=str(e))
        return {
            "problem": problem_name,
            "solver": solver.name,
            "seed": seed,
            "status": "LOAD_FAILED",
            "error": str(e),
        }

    try:
        result: SolverResult = solver.solve(problem)
    except Exception as e:
        logger.exception("solver_crashed", solver=solver.name, problem=problem_name)
        return {
            "problem": problem_name,
            "solver": solver.name,
            "seed": seed,
            "status": "CRASHED",
            "error": str(e),
        }

    metadata = _collect_metadata()

    output_record = {
        "problem": problem_name,
        "solver": solver.name,
        "seed": seed,
        "result": result.to_dict(),
        "metadata": asdict(metadata),
    }

    # JSON kaydet
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{solver.name}__{problem_name}__seed{seed}.json"
    output_file = output_dir / filename
    with open(output_file, "w") as f:
        json.dump(output_record, f, indent=2, default=str)

    logger.info(
        "experiment_done",
        problem=problem_name, solver=solver.name, seed=seed,
        status=result.status.value, iters=result.iterations,
        time=result.time_total,
    )
    return output_record
