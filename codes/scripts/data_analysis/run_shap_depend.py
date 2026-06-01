from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

BASE_SHAP  = Path("../../../data/results/shap_results/selected")
LGBM_BASE  = Path("../../../data/results/lgbm_results_selected")
OUTPUT_DIR = Path("../../../data/results/paper_figures/shap_dependence")

SCALES = [100, 250, 500, 750, 1000]

COLUMN_SPECS = [
    ("global",          None),
    ("by_climate_zone", 15),
    ("by_climate_zone", 26),
]
COL_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

SAMPLE_SIZE = 3000

FIG_WIDTH  = 22
FIG_HEIGHT = 36

FEAT_LABELS: dict[str, str] = {
    "isa_fraction":           "ISA fraction",
    "PLAND_tree_cover":       "PLAND tree cover",
    "building_height_mean":   "Building Height",
    "BCR":                    "BCR",
    "LPI_water":              "LPI water",
    "PD_built_up":            "PD built-up",
    "PD_tree_cover":          "PD tree cover",
    "PD_grassland":           "PD grassland",
    "ED_tree_cover":          "ED tree cover",
    "PD_water":               "PD water",
    "CONTAG":                 "CONTAG",
    "LPI_grassland":          "LPI grassland",
    "pop_sum":                "Population",
    "CZ_median":              "Climate zone",
    "SHDI":                   "SHDI",
    "PD_cropland":            "PD cropland",
    "LPI_cropland":           "LPI cropland",
    "LPI_bare_land":          "LPI bareland",
    "LPI_wetland":            "LPI wetland",
    "PD_bare_land":           "PD bareland",
    "ED_grassland":           "ED grassland",
    "ED_cropland":            "ED cropland",
    "ED_water":               "ED water",
    "PD_wetland":             "PD wetland",
    "LPI_shrubland":          "LPI shrubland",
    "PD_shrubland":           "PD shrubland",
    "ED_shrubland":           "ED shrubland",
    "LSI_shrubland":          "LSI shrubland",
    "PLAND_shrubland":        "PLAND shrubland",
    "ED_bare_land":           "ED bareland",
    "ED_wetland":             "ED wetland",
}

def get_all_top_features(top_n: int = 10) -> list[str]:
    features = set()
    for scale in SCALES:
        for scope, cz in COLUMN_SPECS:
            try:
                p = _paths(scale, scope, cz)
                imp_df = pd.read_csv(p["imp"])
                top_feats = imp_df.head(top_n)["feature"].tolist()
                features.update(top_feats)
            except Exception as e:
                print(f"Warning: could not process importance for {scale}m {scope} {cz}: {e}")
    return sorted(list(features))

def _label(feat: str) -> str:
    return FEAT_LABELS.get(feat, feat)

def _paths(scale: int, scope: str, cz: int | None) -> dict[str, Path]:
    if scope == "global":
        shap_dir = BASE_SHAP / f"Global_scale_{scale}m"
        xtest    = LGBM_BASE / "global" / f"scale_{scale}m" / "folds" / "X_test.parquet"
    else:
        shap_dir = BASE_SHAP / f"CZ_{cz}_scale_{scale}m"
        xtest    = LGBM_BASE / "by_climate_zone" / f"CZ_{cz}" / f"scale_{scale}m" / "folds" / "X_test.parquet"
    return {
        "shap": shap_dir / "arrays" / "shap_values.parquet",
        "imp":  shap_dir / "tables" / "shap_relative_importance.csv",
        "xtest": xtest,
    }

def _load(scale: int, scope: str, cz: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _paths(scale, scope, cz)
    return (pd.read_parquet(p["shap"]), pd.read_parquet(p["xtest"]))

def _norm_feature(vals: np.ndarray) -> np.ndarray:
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))
    if hi == lo:
        return np.full(len(vals), 0.5)
    normed = (vals - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0).astype(float)

def _plot_depend_cell(
    ax: plt.Axes,
    shap_df: pd.DataFrame,
    x_df: pd.DataFrame,
    target_feature: str,
    *,
    panel_label: str,
    seed: int = 0,
) -> None:
    if target_feature not in x_df.columns:
        ax.text(0.5, 0.5, f"'{_label(target_feature)}'\nnot selected by VIF", 
                ha='center', va='center', fontsize=16, color='#888888')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top", ha="left")
        ax.set_facecolor("#f9f9f9")
        return

    rng = np.random.default_rng(seed)
    n = len(shap_df)
    if n > SAMPLE_SIZE:
        idx = rng.choice(n, SAMPLE_SIZE, replace=False)
        shap_sub = shap_df.iloc[idx].reset_index(drop=True)
        x_sub    = x_df.iloc[idx].reset_index(drop=True)
    else:
        shap_sub, x_sub = shap_df.reset_index(drop=True), x_df.reset_index(drop=True)

    ind = x_df.columns.get_loc(target_feature)
    int_ind = shap.utils.approximate_interactions(ind, shap_sub.values, x_sub.values)[0]
    int_feature = x_sub.columns[int_ind]

    sv = shap_sub[target_feature].values.astype(float)
    fv = x_sub[target_feature].values.astype(float)
    int_v = x_sub[int_feature].values.astype(float)

    cmap_obj = shap.plots.colors.red_blue
    norm_obj = Normalize(vmin=0, vmax=1)
    colors = cmap_obj(norm_obj(_norm_feature(int_v)))

    # jitter discrete values slightly in x if few unique values
    unique_vals = len(np.unique(fv))
    if unique_vals < 15:
        width = (np.max(fv) - np.min(fv)) if unique_vals > 1 else 1.0
        fv_jit = fv + rng.uniform(-0.02 * width, 0.02 * width, size=len(fv))
    else:
        fv_jit = fv

    ax.scatter(
        fv_jit, sv,
        c=colors, s=12, alpha=1.0,
        linewidths=0, rasterized=True,
        zorder=3,
    )

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="-", zorder=2)
    
    # Text annotation for interaction
    ax.text(0.96, 0.95, f"Interaction:\n{_label(int_feature)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color="#333333")

    ax.set_xlabel(f"{_label(target_feature)} value", fontsize=16)
    ax.set_ylabel("SHAP value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left")

    ax.set_facecolor("white")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)


def make_figure_for_feature(target_feature: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(SCALES)
    n_cols = len(COLUMN_SPECS)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    main_gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.1, right=0.96,
        top=0.97,  bottom=0.02,
        width_ratios=[0.94, 0.015],
        wspace=0.025
    )

    axes_gs = main_gs[0, 0].subgridspec(
        n_rows, n_cols,
        hspace=0.25, 
        wspace=0.35 
    )

    cbar_gs = main_gs[0, 1].subgridspec(
        n_rows, 1,
        hspace=0.25
    )

    curr_plot_idx = 0
    for ri, scale in enumerate(SCALES):
        for ci, (scope, cz) in enumerate(COLUMN_SPECS):
            shap_df, x_df = _load(scale, scope, cz)

            ax = fig.add_subplot(axes_gs[ri, ci])

            if ri == 0:
                ax.set_title(
                    COL_TITLES[ci],
                    fontsize=18, fontweight="bold",
                    pad=30,
                )

            panel_lbl = chr(ord('a') + curr_plot_idx)
            curr_plot_idx += 1

            _plot_depend_cell(
                ax,
                shap_df, x_df, target_feature,
                panel_label=panel_lbl,
                seed=ri * 10 + ci,
            )

        ss_row   = axes_gs[ri, 0]
        bbox_row = ss_row.get_position(fig)
        y_mid    = (bbox_row.y0 + bbox_row.y1) / 2
        x_pos    = bbox_row.x0 - 0.09
        fig.text(
            x_pos, y_mid,
            f"{scale} m",
            ha="right", va="center",
            fontsize=18, fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

        cbar_ax = fig.add_subplot(cbar_gs[ri, 0])
        sm = ScalarMappable(cmap=shap.plots.colors.red_blue, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.outline.set_visible(False)
        cbar.set_label("Interaction\nFeature value", fontsize=16, rotation=270, labelpad=25)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"], fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    safe_name = target_feature.replace("_", "-")
    out_pdf = OUTPUT_DIR / f"shap_depend_{safe_name}_5x3.pdf"
    out_png = OUTPUT_DIR / f"shap_depend_{safe_name}_5x3.png"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    print("[run_shap_depend] Identifying top features across all 15 models...")
    all_top_features = get_all_top_features(10)
    print(f"Found {len(all_top_features)} unique features in the top 10 across all models:")
    print(all_top_features)
    
    print(f"\n[run_shap_depend] Generating {len(all_top_features)} dependence figures (5x3)...")
    for i, feat in enumerate(all_top_features):
        print(f"Processing ({i+1}/{len(all_top_features)}): {feat}")
        make_figure_for_feature(feat)
    print("Done!")
