"""Tek bir deneyi çalıştıran CLI script'i."""
import argparse
from pathlib import Path

from srdstiff.benchmark.runner import run_single_experiment
from srdstiff.core.types import SolverConfig, SRDStiffConfig
from srdstiff.solvers.srd_stiff import SRDStiffSolver
from srdstiff.solvers.scipy_wrapper import ScipySolverWrapper
from srdstiff.utils.reproducibility import set_all_seeds


def get_solver(name: str, seed: int, max_iter: int, max_time: float):
    """Verilen isimden solver örneği oluşturur."""
    # Sabit solver parametreleri
    if name == "SRD-STIFF":
        config = SRDStiffConfig(
            seed=seed,
            max_iter=max_iter,
            max_time_sec=max_time,
            alpha_init=1.0,
            K=500.0,
            gamma=0.1,
            epsilon=0.1,
            delta=0.05,
            L_H_estimate=1.0,
        )
        return SRDStiffSolver(config)
    
    # Scipy baselines
    scipy_mapping = {
        "L-BFGS": "L-BFGS-B",
        "Newton-CG": "Newton-CG",
        "Trust-NCG": "trust-ncg",
        "Trust-Krylov": "trust-krylov",
    }
    
    if name in scipy_mapping:
        config = SolverConfig(seed=seed, max_iter=max_iter, max_time_sec=max_time)
        solver = ScipySolverWrapper(config, method=scipy_mapping[name])
        solver.name = name  # experiment.yaml ile aynı ismi koru
        return solver
    
    # Placeholder for others like TRON, ARC, LM-N, Reg-Newton, GD-Armijo
    # Gerçek benchmark için uygun kütüphaneler eklenmelidir. (örn. PyCUTEst içinde var olan veya scipy dışı)
    # Şimdilik L-BFGS-B veya uygun bir şeyle fallback yapıp uyarı verebiliriz veya not-implemented bırakabiliriz.
    config = SolverConfig(seed=seed, max_iter=max_iter, max_time_sec=max_time)
    print(f"Warning: Exact implementation for {name} missing, falling back to basic wrapper if possible.")
    if name == "GD-Armijo":
        solver = ScipySolverWrapper(config, method="CG")
        solver.name = name
        return solver
    else:
        # Fallback to trust-ncg for hessian-based
        solver = ScipySolverWrapper(config, method="trust-ncg")
        solver.name = name
        return solver


def main():
    parser = argparse.ArgumentParser(description="Run a single SRD-STIFF benchmark experiment.")
    parser.add_argument("--problem", type=str, required=True, help="CUTEst problem name")
    parser.add_argument("--solver", type=str, required=True, help="Solver name")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-iter", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--max-time", type=float, default=3600.0, help="Maximum time in seconds")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    
    solver = get_solver(args.solver, args.seed, args.max_iter, args.max_time)
    output_dir = Path(args.output_dir)
    
    run_single_experiment(
        problem_name=args.problem,
        solver=solver,
        seed=args.seed,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()
