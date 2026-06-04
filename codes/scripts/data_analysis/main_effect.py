from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

BASE_SHAP  = Path("../../../data/results/shap_results/selected")
LGBM_BASE  = Path("../../../data/results/lgbm_results_selected")
OUTPUT_DIR = Path("../../../data/results/paper_figures/shap_dependence_targeted")

SCALES = [100, 250, 500, 750, 1000]

COLUMN_SPECS = [
    ("global",          None),
    ("by_climate_zone", 15),
    ("by_climate_zone", 26),
]
COL_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

SAMPLE_SIZE = 3000

FIG_WIDTH  = 6.5
FIG_HEIGHT = 10

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

def _label(feat: str) -> str:
    return FEAT_LABELS.get(feat, feat)

def _paths(scale: int, scope: str, cz: int | None) -> dict[str, Path]:
    if scope == "global":
        shap_dir = BASE_SHAP / f"Global_scale_{scale}m"
        xtest    = LGBM_BASE / "global" / f"scale_{scale}m" / "folds" / "X_test.parquet"
        model    = LGBM_BASE / "global" / f"scale_{scale}m" / "models" / "final_model.pkl"
    else:
        shap_dir = BASE_SHAP / f"CZ_{cz}_scale_{scale}m"
        xtest    = LGBM_BASE / "by_climate_zone" / f"CZ_{cz}" / f"scale_{scale}m" / "folds" / "X_test.parquet"
        model    = LGBM_BASE / "by_climate_zone" / f"CZ_{cz}" / f"scale_{scale}m" / "models" / "final_model.pkl"
    return {
        "shap": shap_dir / "arrays" / "shap_values.parquet",
        "xtest": xtest,
        "model": model,
        "shap_dir": shap_dir
    }

def _plot_main_effect_cell(
    ax: plt.Axes,
    int_vals: np.ndarray,
    x_sub: pd.DataFrame,
    target_feature: str,
    *,
    panel_label: str,
    show_ylabel: bool = True,
) -> None:
    if target_feature not in x_sub.columns:
        ax.text(0.5, 0.5, f"Feature missing\nnot selected in model", 
                ha='center', va='center', fontsize=9, color='#888888')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left")
        ax.set_facecolor("#f9f9f9")
        return

    # Determine index of the target feature in the dataset used for SHAP
    target_idx = x_sub.columns.get_loc(target_feature)

    # Main effect is the diagonal of interaction values
    main_effect = int_vals[:, target_idx, target_idx].astype(float)
    fv = x_sub[target_feature].values.astype(float)

    # Use shap's default blue color for consistency
    colors = "#1e88e5"

    unique_vals = len(np.unique(fv))
    if unique_vals < 15:
        width = (np.max(fv) - np.min(fv)) if unique_vals > 1 else 1.0
        rng = np.random.default_rng(42)
        fv_jit = fv + rng.uniform(-0.02 * width, 0.02 * width, size=len(fv))
    else:
        fv_jit = fv

    ax.scatter(
        fv_jit, main_effect,
        c=colors, s=2, alpha=0.8,
        linewidths=0, rasterized=True,
        zorder=3,
    )

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":", alpha=0.5, zorder=2)
    
    ax.set_xlabel(f"{_label(target_feature)} value", fontsize=9)
    
    if show_ylabel:
        ax.set_ylabel("SHAP Main Effect", fontsize=9)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=8, labelleft=show_ylabel) 

    ax.text(0.98, 0.02, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")

    ax.set_facecolor("white")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

def make_main_effect_figure(target_feature: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(SCALES)
    n_cols = len(COLUMN_SPECS)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    main_gs = gridspec.GridSpec(
        1, 1,
        figure=fig,
        left=0.1, right=0.96,
        top=0.97,  bottom=0.02,
    )

    axes_gs = main_gs[0, 0].subgridspec(
        n_rows, n_cols,
        hspace=0.25, 
        wspace=0.35 
    )

    curr_plot_idx = 0
    first_col_axes = []

    for ri, scale in enumerate(SCALES):
        for ci, (scope, cz) in enumerate(COLUMN_SPECS):
            paths = _paths(scale, scope, cz)
            x_df = pd.read_parquet(paths["xtest"])
            
            n = len(x_df)
            seed = ri * 10 + ci
            rng = np.random.default_rng(seed)
            if n > SAMPLE_SIZE:
                idx = rng.choice(n, SAMPLE_SIZE, replace=False)
                x_sub = x_df.iloc[idx].reset_index(drop=True)
            else:
                x_sub = x_df.reset_index(drop=True)
            
            if not paths["model"].exists():
                print(f"Model missing: {paths['model']}. Stopping execution.")
                sys.exit(0)
            
            print(f"Computing interaction values for scale={scale}m, scope={scope} (n={len(x_sub)})...")
            model = joblib.load(paths["model"])
            explainer = shap.TreeExplainer(model)
            if not hasattr(explainer, "shap_interaction_values"):
                print("Explainer does not support shap_interaction_values. Stopping execution.")
                sys.exit(0)
                
            int_vals = explainer.shap_interaction_values(x_sub)
            
            if int_vals is None:
                print(f"Failed to compute interaction values. Stopping execution.")
                sys.exit(0)

            ax = fig.add_subplot(axes_gs[ri, ci])

            if ci == 0:
                first_col_axes.append(ax)

            if ri == 0:
                ax.set_title(
                    COL_TITLES[ci],
                    fontsize=10, fontweight="bold",
                    pad=18,
                )

            panel_lbl = chr(ord('a') + curr_plot_idx)
            curr_plot_idx += 1

            _plot_main_effect_cell(
                ax,
                int_vals, x_sub, target_feature,
                panel_label=panel_lbl,
                show_ylabel=(ci == 0)
            )

            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.8)
            ax.tick_params(axis="both", direction="in", top=True, right=True, 
                           labelsize=8, width=0.8)

        ss_row   = axes_gs[ri, 0]
        bbox_row = ss_row.get_position(fig)
        y_mid    = (bbox_row.y0 + bbox_row.y1) / 2
        x_pos    = bbox_row.x0 - 0.09
        fig.text(
            x_pos, y_mid,
            f"{scale} m",
            ha="right", va="center",
            fontsize=10, fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

    fig.align_ylabels(first_col_axes)

    safe_t = target_feature.replace("_", "-")
    out_pdf = OUTPUT_DIR / f"shap_main_effect_{safe_t}_5x3.pdf"
    out_png = OUTPUT_DIR / f"shap_main_effect_{safe_t}_5x3.png"
    
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved main effect analysis:\n  -> {out_pdf}\n  -> {out_png}")

if __name__ == "__main__":
    TARGET = "ED_tree_cover"
    
    print(f"[main_effect] Generating main effect figure:")
    print(f"  Target feature: {TARGET}")
    
    make_main_effect_figure(TARGET)
    print("Done!")
