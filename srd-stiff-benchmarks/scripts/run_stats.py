"""İstatistiksel testleri (Wilcoxon + FDR) çalıştırır ve CSV olarak kaydeder."""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from srdstiff.benchmark.stats import paired_wilcoxon, benjamini_hochberg


def main():
    parser = argparse.ArgumentParser(description="Run statistical tests comparing SRD-STIFF to baselines.")
    parser.add_argument("--input", type=str, required=True, help="Aggregated Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for test results")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        print(f"Error: {input_file} does not exist.")
        return

    df = pd.read_parquet(input_file)
    if df.empty:
        print("Dataframe is empty, creating dummy stats.")
        pd.DataFrame(columns=[
            "solver_a", "solver_b", "median_diff_log2", 
            "statistic", "pvalue", "qvalue", "n_pairs", "effect_size"
        ]).to_csv(output_file, index=False)
        return

    # Ortalama süreleri veya medyan süreleri çekelim.
    # Wilcoxon, seed'leri averajlanmış data üzerinden mi yapılmalı? 
    # Genelde problem bazında medyan alıp karşılaştırmak mantıklı.
    df_med = df.groupby(["problem", "solver"])["time_total"].median().reset_index()

    solvers = df["solver"].unique()
    target_solver = "SRD-STIFF"
    
    records = []
    
    if target_solver in solvers:
        for baseline in solvers:
            if baseline == target_solver:
                continue
            
            res = paired_wilcoxon(
                df_med, 
                solver_a=target_solver, 
                solver_b=baseline, 
                metric="time_total", 
                log_transform=True
            )
            records.append({
                "solver_a": res.solver_a,
                "solver_b": res.solver_b,
                "median_diff_log2": res.median_diff_log2,
                "statistic": res.statistic,
                "pvalue": res.pvalue,
                "n_pairs": res.n_pairs,
                "effect_size": res.effect_size,
            })
            
    if not records:
        print("Not enough data to run statistical tests (maybe SRD-STIFF was missing).")
        pd.DataFrame(columns=[
            "solver_a", "solver_b", "median_diff_log2", 
            "statistic", "pvalue", "qvalue", "n_pairs", "effect_size"
        ]).to_csv(output_file, index=False)
        return
        
    stats_df = pd.DataFrame(records)
    
    # FDR correction
    stats_df["qvalue"] = benjamini_hochberg(stats_df["pvalue"].values, q=0.05)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_file, index=False)
    print(f"Saved statistical tests to {output_file}")


if __name__ == "__main__":
    main()
