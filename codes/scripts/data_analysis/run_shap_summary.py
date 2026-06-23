"""
paper_vis_shap.py
=================
Composite 5×3 SHAP summary figure for the GermanCitiesSUHI paper.

Layout
------
Rows  : 5 spatial scales  [100, 250, 500, 750, 1000] m  (top → bottom)
Cols  : 3 model scopes    [Global | Climate Zone 15 | Climate Zone 26]

Each grid cell contains two vertically stacked panels
  ┌─────────────────────────────────────────────┐
  │  BAR   mean |SHAP|  →  (x-axis on top)      │
  ├─────────────────────────────────────────────┤
  │  DOT   SHAP value   →  (coloured by feat)   │
  └─────────────────────────────────────────────┘

One feature-value colour bar is shared across each row's three cells.
"""

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

# ── configuration ─────────────────────────────────────────────────────────────

BASE_SHAP  = Path("/home/GermanCitiesSUHI/data/results/shap_results/selected")
LGBM_BASE  = Path("/home/GermanCitiesSUHI/data/results/lgbm_results_selected")
OUTPUT_DIR = Path("/home/GermanCitiesSUHI/data/results/paper_figures/shap_summary")

# SCALES = [100, 250, 500, 750, 1000]   # rows top → bottom
SCALES = [250] 

COLUMN_SPECS = [                       # (scope_dir, climate_zone_or_None)
    ("global",          None),
    ("by_climate_zone", 15),
    ("by_climate_zone", 26),
]
COL_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

TOP_N       = 10     # top features shown per subplot
SAMPLE_SIZE = 3000   # scatter sub-sample per subplot

# FIG_WIDTH  = 8.5   # inches
# FIG_HEIGHT = 14   # inches

FIG_WIDTH  = 8.5  # inches
FIG_HEIGHT = 2.5   # inches

# ── optional readable feature-name map ────────────────────────────────────────

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


# ── path / IO helpers ─────────────────────────────────────────────────────────

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


def _load(scale: int, scope: str, cz: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = _paths(scale, scope, cz)
    return (
        pd.read_parquet(p["shap"]),
        pd.read_csv(p["imp"]),
        pd.read_parquet(p["xtest"]),
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _shap_jitter(shaps: np.ndarray, max_width: float = 0.4) -> np.ndarray:
    """Exactly replicates SHAP's native swarm point density algorithm."""
    shaps = np.asarray(shaps)
    if len(shaps) < 2:
        return np.zeros(len(shaps))
    nbins = 100
    quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)).astype(int)
    inds = np.argsort(shaps + np.random.randn(len(shaps)) * 1e-6)
    layer = 0
    last_bin = -1
    ys = np.zeros(len(shaps))
    for ind in inds:
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    
    max_y = np.max(ys + 1)
    if max_y > 0:
        ys *= max_width / max_y
    return ys


def _norm_feature(vals: np.ndarray) -> np.ndarray:
    """Percentile-clip and normalise feature values to [0, 1] for colouring."""
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))
    if hi == lo:
        return np.full(len(vals), 0.5)
    normed = (vals - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0).astype(float)


# ── per-cell plot ─────────────────────────────────────────────────────────────

def _plot_cell(
    ax: plt.Axes,
    shap_df: pd.DataFrame,
    x_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    *,
    panel_label: str,
    seed: int = 0,
) -> None:
    """
    Overlay bar (background, top x-axis) and beeswarm dots (foreground,
    bottom x-axis) on a single axes.  The two x-axes are independent via
    twiny(), but share the y-axis (feature rows).
    """
    rng = np.random.default_rng(seed)
    features = imp_df["feature"].head(TOP_N).tolist()
    n_feat = len(features)

    # ── sub-sample for dot plot ───────────────────────────────────────────────
    n = len(shap_df)
    if n > SAMPLE_SIZE:
        idx = rng.choice(n, SAMPLE_SIZE, replace=False)
        shap_sub = shap_df.iloc[idx].reset_index(drop=True)
        x_sub    = x_df.iloc[idx].reset_index(drop=True)
    else:
        shap_sub, x_sub = shap_df.reset_index(drop=True), x_df.reset_index(drop=True)

    cmap_obj = shap.plots.colors.red_blue
    norm_obj = Normalize(vmin=0, vmax=1)

    # Y positions: feature[0] (most important) at top
    y_pos = np.arange(n_feat - 1, -1, -1)  # [n_feat-1, …, 0]
    y_lim = (-0.7, n_feat - 0.3)

    # ── BAR layer — twin top x-axis (background, zorder=1) ───────────────────
    ax_bar = ax.twiny()          # shares y-axis with ax
    mean_abs = imp_df.set_index("feature").loc[features, "mean_abs_shap"].values
    ax_bar.barh(
        y_pos, mean_abs,
        height=0.6,
        color="#cccccc",  # neutral, lighter grey for background bar
        alpha=0.4,
        linewidth=0,
        zorder=1,
    )
    ax_bar.set_xlim(left=0)
    ax_bar.set_ylim(*y_lim)
    ax_bar.xaxis.tick_top()
    ax_bar.xaxis.set_label_position("top")
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=8, labelpad=4)
    ax_bar.tick_params(axis="x", labelsize=6, top=True)
    ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_bar.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
    ax_bar.spines["right"].set_visible(False)
    # hide the bottom spine of twiny (it sits on top anyway)
    ax_bar.spines["bottom"].set_visible(False)
    # also hide left and top for ax_bar, so the main ax box frames everything
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)

    # ── DOT layer — primary bottom x-axis (foreground, zorder=3) ────────────
    for fi, feat in enumerate(features):
        y_base = n_feat - 1 - fi
        sv     = shap_sub[feat].values.astype(float)
        fv     = (x_sub[feat].values.astype(float)
                  if feat in x_sub.columns else np.zeros(len(sv)))
        colors = cmap_obj(norm_obj(_norm_feature(fv)))
        # Standard SHAP native density beeswarm layout
        y_jit  = y_base + _shap_jitter(sv, max_width=0.45)
        ax.scatter(
            sv, y_jit,
            c=colors, s=3, alpha=1.0,  # SHAP style
            linewidths=0, rasterized=True,
            zorder=3,
        )

    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="-", zorder=2)

    # ── shared y-axis styling ─────────────────────────────────────────────────
    ax.set_ylim(*y_lim)
    ax.set_yticks(y_pos)
    # Always display y-axis labels for each subplot
    ax.set_yticklabels([_label(f) for f in features], fontsize=8)
    ax.tick_params(axis="y", length=0)

    # primary bottom x-axis (SHAP value)
    ax.set_xlabel("SHAP value", fontsize=8, labelpad=2)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    
    # We want a closed bounding box around the axes, so don't hide spines here.
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)

    # Add subplot label (a), (b), (c)... without a bounding box 
    # placed just right at the top left interior
    ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left",
            bbox=None)

    # background colour so bars read against white
    ax.set_facecolor("white")


# ── main ──────────────────────────────────────────────────────────────────────

def make_figure() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(SCALES)
    n_cols = len(COLUMN_SPECS)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # 1. Top-level GridSpec: Separate the canvas into a Main Plot Area and a Colorbar Area
    # main_gs[0, 0] is for all SHAP plots, main_gs[0, 1] is dedicated to the colorbars
    main_gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.1, right=0.9,  # Leave enough left margin for row labels
        top=0.97,  bottom=0.02,
        width_ratios=[0.94, 0.015], # 94% width for plots, 1.5% for colorbars 
        wspace=0.025                # [CRITICAL] This strictly controls the distance between the last plot column and the colorbar
    )

    # 2. Nested GridSpec for the main SHAP plots
    # We can control hspace and wspace internally without affecting the colorbar position
    axes_gs = main_gs[0, 0].subgridspec(
        n_rows, n_cols,
        hspace=0.35, 
        wspace=0.75  # Keep optimized spacing between main columns for y-labels
    )

    # 3. Nested GridSpec for the row colorbars (aligned vertically with the main plots)
    cbar_gs = main_gs[0, 1].subgridspec(
        n_rows, 1,
        hspace=0.35
    )

    # ── iterate rows and columns ───────────────────────────────────────────────
    curr_plot_idx = 0
    for ri, scale in enumerate(SCALES):

        for ci, (scope, cz) in enumerate(COLUMN_SPECS):
            label = f"Scale {scale} m | {scope}" + (f" CZ={cz}" if cz else "")
            print(f"  Loading {label} …", end=" ", flush=True)

            shap_df, imp_df, x_df = _load(scale, scope, cz)

            # Use the nested axes_gs instead of outer_gs
            ax = fig.add_subplot(axes_gs[ri, ci])

            # Column title: only first row, placed as axes title with padding
            if ri == 0:
                ax.set_title(
                    COL_TITLES[ci],
                    fontsize=8, fontweight="bold",
                    pad=18,  # push above the top x-axis tick labels
                )

            panel_lbl = chr(ord('a') + curr_plot_idx)
            curr_plot_idx += 1

            _plot_cell(
                ax,
                shap_df, x_df, imp_df,
                panel_label=panel_lbl,
                seed=ri * 10 + ci,
            )
            print("done")

        # Row label: centred vertically across both panels using the subgridspec bbox
        ss_row   = axes_gs[ri, 0]
        bbox_row = ss_row.get_position(fig)
        y_mid    = (bbox_row.y0 + bbox_row.y1) / 2
        
        # Using bbox_row.x0 dynamically ensures row labels move automatically if margins change
        # Changed ha to "right" for elegant alignment of text ends
        x_pos    = bbox_row.x0 - 0.12
        fig.text(
            x_pos, y_mid,
            f"{scale} m",
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

        # ── shared feature-value colorbar for this row ────────────────────────
        # Use the nested cbar_gs instead of outer_gs
        cbar_ax = fig.add_subplot(cbar_gs[ri, 0])
        sm = ScalarMappable(cmap=shap.plots.colors.red_blue, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.outline.set_visible(False)
        
        # Adjusted labelpad to avoid overlap between "High/Low" and "Feature value"
        cbar.set_label("Feature value", fontsize=8, rotation=270, labelpad=-12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"], fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    # ── save ──────────────────────────────────────────────────────────────────
    out_pdf = OUTPUT_DIR / "shap_composite_250_5x3.pdf"
    out_png = OUTPUT_DIR / "shap_composite_250_5x3.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[paper_vis_shap] Saved → {out_pdf}")
    print(f"[paper_vis_shap] Saved → {out_png}")


if __name__ == "__main__":
    make_figure()
