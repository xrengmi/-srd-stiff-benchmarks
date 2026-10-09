"""Deney sonuçlarını toplar (aggregate) ve parquet olarak kaydeder."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment JSON files into a Parquet file.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    
    records = []
    
    for json_file in input_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                
            # Düz (flat) bir yapı oluştur
            row = {
                "problem": data.get("problem"),
                "solver": data.get("solver"),
                "seed": data.get("seed"),
            }
            
            # Result kısmı (başarılı olanlar vs.)
            res = data.get("result", {})
            if "status" in res:
                 row["status"] = res.get("status")
            elif "status" in data:
                 row["status"] = data.get("status")
                 
            row["iterations"] = res.get("iterations", 0)
            row["time_total"] = res.get("time_total", 0.0)
            row["f_final"] = res.get("f_final", float("nan"))
            row["grad_norm_final"] = res.get("grad_norm_final", float("nan"))
            row["nfev"] = res.get("nfev", 0)
            row["ngev"] = res.get("ngev", 0)
            row["nhev"] = res.get("nhev", 0)
            row["failure_mode"] = res.get("failure_mode", "NONE")
            
            records.append(row)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            
    if not records:
        print("No valid JSON records found. Creating empty DataFrame.")
        df = pd.DataFrame(columns=[
            "problem", "solver", "seed", "status", "iterations", 
            "time_total", "f_final", "grad_norm_final", "nfev", "ngev", "nhev", "failure_mode"
        ])
    else:
        df = pd.DataFrame(records)
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Aggregated {len(df)} records into {output_file}")


if __name__ == "__main__":
    main()
