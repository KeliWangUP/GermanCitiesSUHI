import argparse
import json
from pathlib import Path
import re
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _none_if_na(value):
	if value is None:
		return None
	if isinstance(value, float) and np.isnan(value):
		return None
	if isinstance(value, str) and value.strip().lower() in {'', 'nan', 'none', 'null'}:
		return None
	return value


def _load_shap_module():
	try:
		import shap
	except ImportError as exc:
		raise ImportError(
			'shap is required for SHAP analysis. Install it with: pip install shap'
		) from exc
	return shap


def load_lgbm_artifacts(
	final_model_path: str | Path,
	x_test_path: str | Path,
	y_test_path: str | Path | None = None,
) -> tuple[Any, pd.DataFrame, pd.Series | None]:
	"""Load a fitted LGBM model and the saved holdout feature matrix."""
	model_path = Path(final_model_path)
	features_path = Path(x_test_path)

	if not model_path.exists():
		raise FileNotFoundError(f'Model path does not exist: {model_path}')
	if not features_path.exists():
		raise FileNotFoundError(f'X_test path does not exist: {features_path}')

	model = joblib.load(model_path)
	x_test = pd.read_parquet(features_path)

	y_test = None
	if y_test_path is not None:
		target_path = Path(y_test_path)
		if not target_path.exists():
			raise FileNotFoundError(f'y_test path does not exist: {target_path}')
		y_test_df = pd.read_parquet(target_path)
		if y_test_df.shape[1] == 0:
			raise ValueError(f'y_test parquet has no columns: {target_path}')
		y_test = y_test_df.iloc[:, 0]

	return model, x_test, y_test


def _normalize_shap_values(shap_values_raw, *, class_index: int = 0) -> np.ndarray:
	"""Return a 2D SHAP matrix (n_samples, n_features)."""
	if isinstance(shap_values_raw, list):
		if not shap_values_raw:
			raise ValueError('Received an empty SHAP values list')
		if class_index >= len(shap_values_raw):
			raise ValueError(
				f'class_index={class_index} is out of range for {len(shap_values_raw)} outputs'
			)
		return np.asarray(shap_values_raw[class_index])

	values = np.asarray(shap_values_raw)
	if values.ndim == 3:
		if class_index >= values.shape[2]:
			raise ValueError(
				f'class_index={class_index} is out of range for shape {values.shape}'
			)
		return values[:, :, class_index]
	if values.ndim != 2:
		raise ValueError(f'Unexpected SHAP value shape: {values.shape}')
	return values


def _resolve_dependence_features(
	importance_df: pd.DataFrame,
	dependence_features: list[str] | None,
	top_n_dependence: int,
) -> list[str]:
	if dependence_features:
		return dependence_features
	return importance_df['feature'].head(top_n_dependence).tolist()


def compute_relative_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
	"""Compute SHAP mean absolute contribution and relative percent importance."""
	if shap_values.shape[1] != len(feature_names):
		raise ValueError(
			f'Feature count mismatch: shap_values has {shap_values.shape[1]} columns, '
			f'feature_names has {len(feature_names)} entries'
		)

	mean_abs = np.abs(shap_values).mean(axis=0)
	total = float(mean_abs.sum())
	if total <= 0:
		relative = np.zeros_like(mean_abs)
	else:
		relative = mean_abs / total * 100.0

	importance_df = pd.DataFrame(
		{
			'feature': feature_names,
			'mean_abs_shap': mean_abs,
			'relative_importance_pct': relative,
		}
	).sort_values('mean_abs_shap', ascending=False)
	importance_df['rank'] = np.arange(1, len(importance_df) + 1)
	importance_df['cumulative_importance_pct'] = importance_df['relative_importance_pct'].cumsum()
	return importance_df.reset_index(drop=True)


def compute_and_save_shap_for_model(
	*,
	final_model_path: str | Path,
	x_test_path: str | Path,
	output_dir: str | Path,
	y_test_path: str | Path | None = None,
	metadata_path: str | Path | None = None,
	dependence_features: list[str] | None = None,
	top_n_dependence: int = 8,
	max_display: int = 20,
	sample_size: int | None = None,
	random_state: int = 42,
	class_index: int = 0,
	check_additivity: bool = False,
	verbose: int = 1,
) -> dict[str, str]:
	"""
	Compute SHAP values for one trained model and save artifacts:
	- shap_values.parquet
	- shap_relative_importance.csv
	- summary dot/bar plots
	- dependence plots for selected features
	"""
	shap = _load_shap_module()
	output_root = Path(output_dir)
	plots_dir = output_root / 'plots'
	tables_dir = output_root / 'tables'
	arrays_dir = output_root / 'arrays'
	for directory in (output_root, plots_dir, tables_dir, arrays_dir):
		directory.mkdir(parents=True, exist_ok=True)

	model, x_test, y_test = load_lgbm_artifacts(
		final_model_path=final_model_path,
		x_test_path=x_test_path,
		y_test_path=y_test_path,
	)

	if sample_size is not None and sample_size > 0 and len(x_test) > sample_size:
		sampled_idx = x_test.sample(n=sample_size, random_state=random_state).index
		x_test = x_test.loc[sampled_idx].reset_index(drop=True)
		if y_test is not None:
			y_test = y_test.loc[sampled_idx].reset_index(drop=True)

	if verbose:
		print(f'[SHAP] Computing SHAP values for {len(x_test)} rows and {x_test.shape[1]} features')

	explainer = shap.TreeExplainer(model)
	shap_values_raw = explainer.shap_values(x_test, check_additivity=check_additivity)
	shap_values = _normalize_shap_values(shap_values_raw, class_index=class_index)

	shap_values_df = pd.DataFrame(shap_values, columns=x_test.columns)
	shap_values_path = arrays_dir / 'shap_values.parquet'
	shap_values_df.to_parquet(shap_values_path, index=False)

	importance_df = compute_relative_importance(shap_values=shap_values, feature_names=x_test.columns.tolist())
	importance_path = tables_dir / 'shap_relative_importance.csv'
	importance_df.to_csv(importance_path, index=False)

	summary_dot_path = plots_dir / 'shap_summary_dot.png'
	summary_bar_path = plots_dir / 'shap_summary_bar.png'

	plt.figure(figsize=(10, 6))
	shap.summary_plot(shap_values, x_test, max_display=max_display, show=False, plot_type='dot')
	plt.tight_layout()
	plt.savefig(summary_dot_path, dpi=300, bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(10, 6))
	shap.summary_plot(shap_values, x_test, max_display=max_display, show=False, plot_type='bar')
	plt.tight_layout()
	plt.savefig(summary_bar_path, dpi=300, bbox_inches='tight')
	plt.close()

	dependence_dir = plots_dir / 'dependence'
	dependence_dir.mkdir(parents=True, exist_ok=True)

	selected_dependence_features = _resolve_dependence_features(
		importance_df=importance_df,
		dependence_features=dependence_features,
		top_n_dependence=top_n_dependence,
	)

	dependence_paths: dict[str, str] = {}
	for feature in selected_dependence_features:
		if feature not in x_test.columns:
			continue
		feature_slug = re.sub(r'[^A-Za-z0-9._-]+', '_', str(feature))
		feature_path = dependence_dir / f'shap_dependence_{feature_slug}.png'
		plt.figure(figsize=(8, 6))
		shap.dependence_plot(
			ind=feature,
			shap_values=shap_values,
			features=x_test,
			interaction_index='auto',
			show=False,
		)
		plt.tight_layout()
		plt.savefig(feature_path, dpi=300, bbox_inches='tight')
		plt.close()
		dependence_paths[feature] = str(feature_path)

	shap_metadata = {
		'final_model_path': str(final_model_path),
		'x_test_path': str(x_test_path),
		'y_test_path': str(y_test_path) if y_test_path else None,
		'metadata_path': str(metadata_path) if metadata_path else None,
		'n_rows_used': int(len(x_test)),
		'n_features': int(x_test.shape[1]),
		'top_n_dependence': int(top_n_dependence),
		'max_display': int(max_display),
		'sample_size': int(sample_size) if sample_size else None,
		'class_index': int(class_index),
		'check_additivity': bool(check_additivity),
		'dependence_features': selected_dependence_features,
		'generated_files': {
			'shap_values_path': str(shap_values_path),
			'importance_path': str(importance_path),
			'summary_dot_path': str(summary_dot_path),
			'summary_bar_path': str(summary_bar_path),
			'dependence_paths': dependence_paths,
		},
	}
	shap_metadata_path = output_root / 'shap_metadata.json'
	with open(shap_metadata_path, 'w') as fh:
		json.dump(shap_metadata, fh, indent=2)

	if verbose:
		print(f'[SHAP] Saved SHAP artifacts to: {output_root}')

	return {
		'shap_values_path': str(shap_values_path),
		'importance_path': str(importance_path),
		'summary_dot_path': str(summary_dot_path),
		'summary_bar_path': str(summary_bar_path),
		'shap_metadata_path': str(shap_metadata_path),
	}


def run_shap_from_results_csv(
	results_csv_path: str | Path,
	output_root_dir: str | Path,
	*,
	top_n_dependence: int = 8,
	max_display: int = 20,
	sample_size: int | None = None,
	class_index: int = 0,
	check_additivity: bool = False,
	verbose: int = 1,
) -> pd.DataFrame:
	"""Batch SHAP generation using a results summary CSV from the LGBM runner."""
	results_path = Path(results_csv_path)
	if not results_path.exists():
		raise FileNotFoundError(f'Results CSV not found: {results_path}')

	output_root = Path(output_root_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	df_results = pd.read_csv(results_path)
	required_cols = ['final_model_path', 'X_test_path']
	missing_cols = [col for col in required_cols if col not in df_results.columns]
	if missing_cols:
		raise KeyError(f'Results CSV is missing required columns: {missing_cols}')

	run_rows = []
	for idx, row in df_results.iterrows():
		climate_zone = row.get('climate_zone', 'NA')
		scale_m = row.get('scale_m', 'NA')
		model_tag = f'CZ_{climate_zone}_scale_{scale_m}m'
		run_dir = output_root / str(model_tag)

		if verbose:
			print(f'[SHAP] ({idx + 1}/{len(df_results)}) Processing {model_tag}')

		paths = compute_and_save_shap_for_model(
			final_model_path=row['final_model_path'],
			x_test_path=row['X_test_path'],
			y_test_path=_none_if_na(row.get('y_test_path')),
			metadata_path=_none_if_na(row.get('metadata_path')),
			output_dir=run_dir,
			top_n_dependence=top_n_dependence,
			max_display=max_display,
			sample_size=sample_size,
			class_index=class_index,
			check_additivity=check_additivity,
			verbose=verbose,
		)

		run_rows.append(
			{
				'climate_zone': climate_zone,
				'scale_m': scale_m,
				'feature_mode': row.get('feature_mode'),
				'run_dir': str(run_dir),
				**paths,
			}
		)

	run_summary = pd.DataFrame(run_rows)
	summary_path = output_root / 'shap_run_summary.csv'
	run_summary.to_csv(summary_path, index=False)
	if verbose:
		print(f'[SHAP] Batch summary saved to: {summary_path}')
	return run_summary


def _build_cli_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='SHAP analysis pipeline for saved LGBM artifacts')
	parser.add_argument('--results-csv', type=str, required=True, help='Path to LGBM summary CSV')
	parser.add_argument('--output-dir', type=str, required=True, help='Directory to save SHAP artifacts')
	parser.add_argument('--top-n-dependence', type=int, default=8)
	parser.add_argument('--max-display', type=int, default=20)
	parser.add_argument('--sample-size', type=int, default=None)
	parser.add_argument('--class-index', type=int, default=0)
	parser.add_argument('--check-additivity', action='store_true')
	parser.add_argument('--quiet', action='store_true')
	return parser


def main():
	parser = _build_cli_parser()
	args = parser.parse_args()

	run_shap_from_results_csv(
		results_csv_path=args.results_csv,
		output_root_dir=args.output_dir,
		top_n_dependence=args.top_n_dependence,
		max_display=args.max_display,
		sample_size=args.sample_size,
		class_index=args.class_index,
		check_additivity=args.check_additivity,
		verbose=0 if args.quiet else 1,
	)


if __name__ == '__main__':
	main()
