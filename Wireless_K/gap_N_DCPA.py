"""
Created on : ------
@brief: Plot gap percentages from saved DCPA inference results for all N*_K5 runs.
@author: Ariel_Kantorovich
"""

from pathlib import Path
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt


def _extract_n_from_dirname(name: str) -> int | None:
    match = re.match(r"N(\d+)_K\d+", name)
    if match is None:
        return None
    return int(match.group(1))


def load_all_inference_results(base_dir: Path, k_value: int = 5) -> dict[str, np.ndarray]:
    """
    Load all files matching:
      base_dir / f"N*_K{k_value}" / "results" / "inference_results.npz"
    Returns a dictionary with sorted arrays by N:
      { "N", "NE", "alpha", "alphaBeta" }
    """
    pattern = f"N*_K{k_value}"
    files = sorted(base_dir.glob(f"{pattern}/results/inference_results.npz"))

    if not files:
        raise FileNotFoundError(
            f"No inference_results.npz files found under: {base_dir} with pattern {pattern}"
        )

    records = []
    for file_path in files:
        run_dir = file_path.parents[1].name
        n_from_dir = _extract_n_from_dirname(run_dir)

        with np.load(file_path, allow_pickle=False) as data:
            n_val = int(data["N"]) if "N" in data.files else n_from_dir
            if n_val is None:
                raise ValueError(f"Could not determine N for file: {file_path}")

            record = {
                "N": n_val,
                "NE": float(data["NE"]),
                "alpha": float(data["alpha"]),
                "alphaBeta": float(data["alphaBeta"]),
            }
            records.append(record)

    records.sort(key=lambda r: r["N"])
    return {
        key: np.array([r[key] for r in records], dtype=float)
        for key in ("N", "NE", "alpha", "alphaBeta")
    }


def plot_gap_scatter(results: dict[str, np.ndarray], output_path: Path) -> None:
    """Scatter plot of gap(%) vs number of agents N for all techniques."""
    n_vals = results["N"]

    plt.figure(figsize=(10, 6))
    plt.scatter(n_vals, results["NE"], label="NE", s=90, marker="o")
    plt.scatter(n_vals, results["alpha"], label=r"DCPA-($\alpha_n$)", s=90, marker="s")
    plt.scatter(
        n_vals,
        results["alphaBeta"],
        label=r"DCPA-($\alpha_n,\beta_n$)",
        s=90,
        marker="^",
    )

    for n_val, ne_val in zip(n_vals, results["NE"]):
        plt.annotate(f"N={int(n_val)}", (n_val, ne_val), textcoords="offset points", xytext=(4, 6), fontsize=8)

    plt.xlabel("Number of agents (N)")
    plt.ylabel("Gap from optimal (%)")
    plt.title("Gap Comparison Across Techniques")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot DCPA gap scatter for all N*_K5 inference results.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Training_Data",
        help="Base directory containing N*_K*/results/inference_results.npz files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value to filter folders (N*_K<k>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Training_Data" / "gap_N_DCPA_scatter.png",
        help="Output image path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = load_all_inference_results(args.base_dir, k_value=args.k)
    plot_gap_scatter(results, args.output)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
