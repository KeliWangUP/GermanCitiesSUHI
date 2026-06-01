import argparse
from pathlib import Path

from shap_pipeline import run_shap_from_results_csv


def _discover_default_jobs() -> list[tuple[Path, Path]]:
    base_results_dir = Path("../../../data/results")
    mode_roots = {
        "lgbm_results_selected": "selected",
    }

    jobs: list[tuple[Path, Path]] = []
    for lgbm_root_name, mode_name in mode_roots.items():
        root = base_results_dir / lgbm_root_name
        if not root.exists():
            continue

        csv_paths = sorted(root.glob("final_lgbm_results*.csv"))
        for csv_path in csv_paths:
            output_dir = base_results_dir / "shap_results" / mode_name
            jobs.append((csv_path, output_dir))

    return jobs


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for one or multiple LGBM result CSV files"
    )
    parser.add_argument(
        "--results-csv",
        action="append",
        default=[],
        help=(
            "Path to an LGBM summary CSV. Repeat this option for multiple files. "
            "If omitted, jobs are auto-discovered from lgbm_results_all_feat and lgbm_results_selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        action="append",
        default=[],
        help=(
            "Output directory matching each --results-csv. "
            "Must have the same number of entries as --results-csv."
        ),
    )
    parser.add_argument("--top-n-dependence", type=int, default=8)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--class-index", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel SHAP tasks per results CSV. Use -1 to use all CPUs.",
    )
    parser.add_argument(
        "--parallel-prefer",
        type=str,
        default="processes",
        choices=["processes", "threads"],
        help="Parallel backend preference for joblib.",
    )
    parser.add_argument("--check-additivity", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _resolve_jobs(args) -> list[tuple[Path, Path]]:
    if args.results_csv:
        if len(args.results_csv) != len(args.output_dir):
            raise ValueError(
                "When using --results-csv, provide exactly one matching --output-dir per CSV"
            )
        return list(zip([Path(p) for p in args.results_csv], [Path(p) for p in args.output_dir]))

    jobs = _discover_default_jobs()
    if not jobs:
        raise FileNotFoundError(
            "No default jobs found. Expected CSV files under data/results/lgbm_results_all_feat and "
            "data/results/lgbm_results_selected."
        )
    return jobs


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    jobs = _resolve_jobs(args)
    verbose = 0 if args.quiet else 1

    print(f"[RUN SHAP] Total result tables to process: {len(jobs)}")

    for results_csv, output_dir in jobs:
        if not results_csv.exists():
            print(f"[RUN SHAP] Skip missing results CSV: {results_csv}")
            continue

        print(f"[RUN SHAP] Processing: {results_csv}")
        print(f"[RUN SHAP] Output dir: {output_dir}")

        run_shap_from_results_csv(
            results_csv_path=results_csv,
            output_root_dir=output_dir,
            top_n_dependence=args.top_n_dependence,
            max_display=args.max_display,
            sample_size=args.sample_size,
            class_index=args.class_index,
            check_additivity=args.check_additivity,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
