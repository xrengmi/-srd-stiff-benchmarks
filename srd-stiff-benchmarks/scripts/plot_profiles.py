"""Performans profillerini çizip kaydeder."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from srdstiff.benchmark.metrics import compute_performance_ratio, dolan_more_profile


def main():
    parser = argparse.ArgumentParser(description="Plot performance profiles.")
    parser.add_argument("--input", type=str, required=True, help="Aggregated Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output image file (PNG)")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: {input_file} does not exist.")
        return
        
    df = pd.read_parquet(input_file)
    if df.empty:
        print("Dataframe is empty, creating dummy plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file)
        return

    # Ortalama süreyi bulmak için seed üzerinden grupla (aynı problem & solver için)
    # Performans metriklerinde genellikle "zaman" (time_total) veya "nhev" kullanılır.
    # Burada time_total'i kullanacağız.
    df_mean = df.groupby(["problem", "solver"])["time_total"].mean().reset_index()
    
    # Zaman 0 veya 0'a çok yakınsa 1e-6 yap ki sıfıra bölme olmasın
    df_mean["time_total"] = df_mean["time_total"].clip(lower=1e-6)

    # Performans oranını hesapla
    df_ratio = compute_performance_ratio(df_mean, metric="time_total")
    
    # Dolan-Moré profilini hesapla
    tau_grid = np.logspace(0, 3, 100, base=2.0)
    profile_df = dolan_more_profile(df_ratio, tau_grid=tau_grid)
    
    # Çiz
    plt.figure(figsize=(8, 6))
    solvers = profile_df["solver"].unique()
    
    for solver in solvers:
        sub = profile_df[profile_df["solver"] == solver]
        plt.plot(sub["tau"], sub["rho"], label=solver, linewidth=2)
        
    plt.xscale("log", base=2)
    plt.xlabel(r"$\tau$ (Performance Ratio, time)")
    plt.ylabel(r"$\rho(\tau)$ (Fraction of problems)")
    plt.title("Dolan-Moré Performance Profile (Time)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved performance profile to {output_file}")


if __name__ == "__main__":
    main()
