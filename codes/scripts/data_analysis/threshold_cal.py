from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_SHAP = Path("/home/GermanCitiesSUHI/data/results/shap_results/selected")
LGBM_BASE = Path("/home/GermanCitiesSUHI/data/results/lgbm_results_selected")
OUTPUT_DIR = Path("/home/GermanCitiesSUHI/data/results/paper_figures/ed_shap_pland_bins_all_scales")

SCALES = [100, 250, 500, 750, 1000]
SCOPE_SPECS = [
	("global", None),
	("by_climate_zone", 15),
	("by_climate_zone", 26),
]

TARGET_FEATURE = "ED_tree_cover"
INTERACT_FEATURE = "PLAND_tree_cover"
N_BINS = 10
RANDOM_SEED = 42
MAX_POINTS_PER_BIN = 1500
MIN_POINTS_FOR_TREND = 20
TREND_N_BINS_X = 25


def _paths(scale: int, scope: str, climate_zone: int | None) -> tuple[Path, Path]:
	if scope == "global":
		shap_path = BASE_SHAP / f"Global_scale_{scale}m" / "arrays" / "shap_values.parquet"
		xtest_path = LGBM_BASE / "global" / f"scale_{scale}m" / "folds" / "X_test.parquet"
	else:
		shap_path = BASE_SHAP / f"CZ_{climate_zone}_scale_{scale}m" / "arrays" / "shap_values.parquet"
		xtest_path = (
			LGBM_BASE
			/ "by_climate_zone"
			/ f"CZ_{climate_zone}"
			/ f"scale_{scale}m"
			/ "folds"
			/ "X_test.parquet"
		)
	return shap_path, xtest_path


def _load(scale: int, scope: str, climate_zone: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
	shap_path, xtest_path = _paths(scale, scope, climate_zone)
	shap_df = pd.read_parquet(shap_path)
	x_df = pd.read_parquet(xtest_path)
	return shap_df, x_df


def _scope_name(scope: str, climate_zone: int | None) -> str:
	if scope == "global":
		return "global"
	return f"cz_{climate_zone}"


def _trend_from_binned_x(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
	tmp = sub[["x_ed", "shap_ed"]].dropna().copy()
	if len(tmp) < MIN_POINTS_FOR_TREND:
		return np.array([]), np.array([])

	# Bin ED values along x-axis and use median SHAP per x-bin as a robust trend.
	tmp["x_bin"] = pd.qcut(tmp["x_ed"], q=min(TREND_N_BINS_X, len(tmp)), duplicates="drop")
	grp = tmp.groupby("x_bin", observed=False).agg(x_mid=("x_ed", "median"), y_med=("shap_ed", "median")).dropna()
	if grp.empty:
		return np.array([]), np.array([])

	grp = grp.sort_values("x_mid")
	return grp["x_mid"].to_numpy(dtype=float), grp["y_med"].to_numpy(dtype=float)


def _sample_bin(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
	if len(df) <= max_points:
		return df
	return df.sample(n=max_points, random_state=seed)


def plot_ed_shap_by_pland_bins(scale: int, scope: str, climate_zone: int | None) -> None:
	shap_df, x_df = _load(scale, scope, climate_zone)

	required_cols = [TARGET_FEATURE, INTERACT_FEATURE]
	for col in required_cols:
		if col not in x_df.columns:
			raise KeyError(f"Feature '{col}' not found in X_test columns.")
		if col not in shap_df.columns:
			raise KeyError(f"Feature '{col}' not found in SHAP columns.")

	df = pd.DataFrame(
		{
			"x_ed": x_df[TARGET_FEATURE].astype(float).values,
			"shap_ed": shap_df[TARGET_FEATURE].astype(float).values,
			"pland": x_df[INTERACT_FEATURE].astype(float).values,
		}
	).dropna()

	# Use quantile bins so each panel has comparable sample size.
	bin_codes, bin_edges = pd.qcut(df["pland"], q=N_BINS, labels=False, retbins=True, duplicates="drop")
	df = df.assign(pland_bin=bin_codes)
	actual_bins = int(df["pland_bin"].nunique())

	ncols = 5
	nrows = int(np.ceil(actual_bins / ncols))
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.3 * ncols, 2.8 * nrows), squeeze=False)
	axes_flat = axes.ravel()

	rng = np.random.default_rng(RANDOM_SEED)
	panel_idx = 0
	for b in sorted(df["pland_bin"].dropna().unique().astype(int)):
		ax = axes_flat[panel_idx]
		panel_idx += 1

		sub = df[df["pland_bin"] == b]
		sub = _sample_bin(sub, MAX_POINTS_PER_BIN, int(rng.integers(0, 1_000_000)))

		ax.scatter(
			sub["x_ed"],
			sub["shap_ed"],
			s=6,
			alpha=0.55,
			linewidths=0,
			color="#1f77b4",
			rasterized=True,
		)

		trend_x, trend_y = _trend_from_binned_x(sub)
		if len(trend_x) > 0:
			ax.plot(trend_x, trend_y, color="#d62728", linewidth=1.4, alpha=0.95)

		ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.6)

		lo = bin_edges[b]
		hi = bin_edges[b + 1]
		ax.set_title(f"Bin {b + 1}: [{lo:.3f}, {hi:.3f}]\n(n={len(sub)})", fontsize=9)
		ax.set_xlabel("ED_tree_cover value", fontsize=9)
		ax.set_ylabel("SHAP(ED_tree_cover)", fontsize=9)
		ax.tick_params(axis="both", labelsize=8)

	for j in range(panel_idx, len(axes_flat)):
		axes_flat[j].axis("off")

	scope_name = _scope_name(scope, climate_zone)
	fig.suptitle(
		f"{scope_name.upper()} | {scale}m: ED_tree_cover SHAP by PLAND_tree_cover bins",
		fontsize=12,
		y=0.995,
	)
	fig.tight_layout()

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	base_name = f"{scope_name}_scale_{scale}m_ed_shap_by_pland_bins"
	out_png = OUTPUT_DIR / f"{base_name}.png"
	out_pdf = OUTPUT_DIR / f"{base_name}.pdf"
	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved: {out_png}")
	print(f"Saved: {out_pdf}")
	print(f"Used bins: {actual_bins}")


def plot_binwise_shap_boxplot(scale: int, scope: str, climate_zone: int | None) -> None:
	shap_df, x_df = _load(scale, scope, climate_zone)

	for col in [TARGET_FEATURE, INTERACT_FEATURE]:
		if col not in x_df.columns:
			raise KeyError(f"Feature '{col}' not found in X_test columns.")
		if col not in shap_df.columns:
			raise KeyError(f"Feature '{col}' not found in SHAP columns.")

	df = pd.DataFrame(
		{
			"shap_ed": shap_df[TARGET_FEATURE].astype(float).values,
			"pland": x_df[INTERACT_FEATURE].astype(float).values,
		}
	).dropna()

	bin_codes, bin_edges = pd.qcut(df["pland"], q=N_BINS, labels=False, retbins=True, duplicates="drop")
	df = df.assign(pland_bin=bin_codes)
	bins = sorted(df["pland_bin"].dropna().unique().astype(int))

	box_data: list[np.ndarray] = []
	x_labels: list[str] = []
	for b in bins:
		sub = df[df["pland_bin"] == b]["shap_ed"].to_numpy(dtype=float)
		box_data.append(sub)
		lo = bin_edges[b]
		hi = bin_edges[b + 1]
		x_labels.append(f"B{b + 1}\n[{lo:.2f},{hi:.2f}]")

	fig, ax = plt.subplots(figsize=(max(9.5, 1.15 * len(box_data)), 4.8))
	bp = ax.boxplot(
		box_data,
		patch_artist=True,
		showfliers=False,
		medianprops={"color": "#d62728", "linewidth": 1.4},
	)

	for patch in bp["boxes"]:
		patch.set_facecolor("#9ecae1")
		patch.set_edgecolor("#3182bd")
		patch.set_alpha(0.8)

	ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.7)
	ax.set_xticks(np.arange(1, len(x_labels) + 1))
	ax.set_xticklabels(x_labels, fontsize=8)
	ax.set_ylabel("SHAP(ED_tree_cover)", fontsize=10)
	ax.set_xlabel("PLAND_tree_cover bins (quantiles)", fontsize=10)

	scope_name = _scope_name(scope, climate_zone)
	ax.set_title(f"{scope_name.upper()} | {scale}m: SHAP(ED_tree_cover) distribution across PLAND bins", fontsize=11)
	ax.tick_params(axis="y", labelsize=9)
	fig.tight_layout()

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	base_name = f"{scope_name}_scale_{scale}m_ed_shap_boxplot_by_pland_bins"
	out_png = OUTPUT_DIR / f"{base_name}.png"
	out_pdf = OUTPUT_DIR / f"{base_name}.pdf"
	fig.savefig(out_png, dpi=300, bbox_inches="tight")
	fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved: {out_png}")
	print(f"Saved: {out_pdf}")


if __name__ == "__main__":
	print("[threshold_cal] Plotting ED_tree_cover SHAP diagnostics for all scales and scopes...")
	for scale in SCALES:
		for scope, climate_zone in SCOPE_SPECS:
			name = _scope_name(scope, climate_zone)
			print(f"Processing {name}, scale={scale}m...")
			plot_ed_shap_by_pland_bins(scale=scale, scope=scope, climate_zone=climate_zone)
			plot_binwise_shap_boxplot(scale=scale, scope=scope, climate_zone=climate_zone)
	print("Done!")
