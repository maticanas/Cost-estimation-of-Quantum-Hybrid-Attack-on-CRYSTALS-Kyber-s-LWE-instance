"""Add the revised Clifford+T / NIST columns to existing Section 7 results.

This script is the fastest way to update a repository that already contains
``section7_results/best_cost_log2_results_*.csv`` from the exhaustive search.
It does *not* rerun the (r,beta) optimization.  Instead it reads the saved
coarse TD/TC/GD/GC values at each optimum, applies ``FT_gate_cost.py``, updates
the per-instance CSV/JSON files, and writes the three Section 7.4 summary CSVs.

Examples
--------
From the ``Cost esimation`` directory::

    python update_ft_results.py

or for a separate results directory::

    python update_ft_results.py --results-dir my_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import FT_gate_cost as QFT


COARSE_KEYS = {
    "toffoli_depth": "QSearch_Toffoli_depth_log2",
    "toffoli_count": "QSearch_Toffoli_count_log2",
    "gate_depth": "QSearch_Gate_depth_log2",
    "gate_count": "QSearch_Gate_count_log2",
}


def update_row(row: dict) -> dict:
    n = int(row["n"])
    r = int(row["r"])
    prob_log2 = float(row["prob_log2"])
    coarse = {key: float(row[column]) for key, column in COARSE_KEYS.items()}

    ft = QFT.compute_ft_resources_from_log2(n, r, coarse)
    per_success = QFT.add_success_probability(ft, prob_log2)

    maxdepth_fields: dict[str, float | int | None] = {}
    for comparison in QFT.maxdepth_comparison(n, ft, prob_log2):
        h = int(comparison["h"])
        maxdepth_fields[f"FT_MAXDEPTH_h{h}_gate_count_log2"] = float(
            comparison["FT_MAXDEPTH_gate_count_log2"]
        )
        maxdepth_fields[f"NIST_AES_reference_h{h}_log2"] = comparison[
            "NIST_AES_reference_log2"
        ]

    return {**row, **ft, **per_success, **maxdepth_fields}


def write_summary_tables(results: dict[int, dict], results_dir: Path) -> None:
    rotation_rows = []
    ft_rows = []
    maxdepth_rows = []

    for n in sorted(results):
        result = results[n]
        label = QFT.instance_label(n)
        rotation_rows.append(
            {
                "Parameter Set": label,
                "n": n,
                "r": int(result["r"]),
                "log2_L": float(result["log2_L_rotation_budget"]),
                "log2_N_R": float(result["rotation_count_log2"]),
                "b_rot": int(result["rotation_precision_bits"]),
                "T_R_per_rotation": int(result["rotation_T_per_gate"]),
                "log2_N_R_T_R": float(result["rotation_T_count_log2"]),
            }
        )
        ft_rows.append(
            {
                "Parameter Set": label,
                "n": n,
                "r": int(result["r"]),
                "log2_D_FT": float(result["QSearch_FT_depth_log2"]),
                "log2_G_FT": float(result["QSearch_FT_gate_count_log2"]),
                "log2_T_FT": float(result["QSearch_FT_T_count_log2"]),
                "log2_D_FT_per_success": float(
                    result["Hybrid_FT_depth_per_success_log2"]
                ),
                "log2_G_FT_per_success": float(
                    result["Hybrid_FT_gate_count_per_success_log2"]
                ),
                "log2_T_FT_per_success": float(
                    result["Hybrid_FT_T_count_per_success_log2"]
                ),
            }
        )
        for h in QFT.MAXDEPTH_EXPONENTS:
            maxdepth_rows.append(
                {
                    "Parameter Set": label,
                    "n": n,
                    "h": h,
                    "log2_G_MD_FT": float(
                        result[f"FT_MAXDEPTH_h{h}_gate_count_log2"]
                    ),
                    "NIST_AES_reference_log2": result.get(
                        f"NIST_AES_reference_h{h}_log2"
                    ),
                }
            )

    pd.DataFrame(rotation_rows).to_csv(
        results_dir / "rotation_synthesis_budget.csv", index=False
    )
    pd.DataFrame(ft_rows).to_csv(
        results_dir / "ft_gate_resources.csv", index=False
    )
    pd.DataFrame(maxdepth_rows).to_csv(
        results_dir / "maxdepth_gate_comparison.csv", index=False
    )


def update_results(results_dir: str | Path = "section7_results") -> dict[int, dict]:
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    results: dict[int, dict] = {}
    for n in QFT.QTC.SUPPORTED_DIMENSIONS:
        csv_path = results_path / f"best_cost_log2_results_{n}.csv"
        if not csv_path.exists():
            continue

        frame = pd.read_csv(csv_path)
        if len(frame) != 1:
            raise ValueError(f"expected one optimum row in {csv_path}")
        updated = update_row(frame.iloc[0].to_dict())
        results[n] = updated

        pd.DataFrame([updated]).to_csv(csv_path, index=False)
        json_path = results_path / f"best_cost_log2_results_{n}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2, sort_keys=True)

    if not results:
        raise FileNotFoundError(
            f"no best_cost_log2_results_*.csv files found in {results_path}"
        )

    # Keep the convenience combined file in sync with the individual files.
    combined = pd.DataFrame([results[n] for n in sorted(results)])
    combined.to_csv(
        results_path / "best_cost_log2_results_256,512,768,1024.csv",
        index=False,
    )
    write_summary_tables(results, results_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="section7_results")
    args = parser.parse_args()
    results = update_results(args.results_dir)

    for n in sorted(results):
        row = results[n]
        print(
            f"{QFT.instance_label(n)}: "
            f"b_rot={int(row['rotation_precision_bits'])}, "
            f"T_R={int(row['rotation_T_per_gate'])}, "
            f"log2(G_FT/p_success)="
            f"{float(row['Hybrid_FT_gate_count_per_success_log2']):.6f}"
        )
    print(f"Updated FT/NIST results in {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
