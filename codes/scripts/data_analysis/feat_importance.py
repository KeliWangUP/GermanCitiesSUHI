"""
paper_vis_importance_evolution.py
=================================
1x3 Composite 100% Stacked Area plot showing how the relative importance 
of different feature categories (Composition, Configuration, 3D/Socio) 
evolves across spatial scales.

Layout
------
Rows  : 1
Cols  : 3 model scopes [Global | Climate Zone 15 | Climate Zone 26]
X-axis: Grid Scale (100m -> 1000m)
Y-axis: Relative SHAP Importance (%)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── configuration ─────────────────────────────────────────────────────────────

BASE_SHAP  = Path("/home/GermanCitiesSUHI/data/results/shap_results/selected")
OUTPUT_DIR = Path("/home/GermanCitiesSUHI/data/results/paper_figures/shap_summary")

SCALES = [100, 250, 500, 750, 1000]

COLUMN_SPECS = [
    ("global",          None),
    ("by_climate_zone", 15),
    ("by_climate_zone", 26),
]
COL_TITLES = ["Global", "Oceanic (Zone 15)", "Continental (Zone 26)"]

# Publication Style
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['axes.unicode_minus'] = False 

# ── feature categorization logic ──────────────────────────────────────────────

def categorize_feature(feat_name: str) -> str:
    """Classify raw features into macro theoretical categories."""
    if "PLAND" in feat_name or feat_name in ["isa_fraction", "BCR"]:
        return "Composition (Quantity)"
    elif any(x in feat_name for x in ["ED_", "PD_", "LSI_", "LPI_", "SHDI", "CONTAG"]):
        return "Configuration (Heterogeneity)"
    elif feat_name in ["building_height_mean", "pop_sum"]:
        return "3D Structure & Socio-economic"
    else:
        return "Other"

# Category colors (Accessible, high-contrast palette)
CAT_COLORS = {
    "Composition (Quantity)": "#1f77b4",          # Classic Blue
    "Configuration (Heterogeneity)": "#d62728",   # Hero Color: Red
    "3D Structure & Socio-economic": "#7f7f7f",   # Neutral Grey
    "Other": "#e377c2"
}
CAT_ORDER = ["Composition (Quantity)", "3D Structure & Socio-economic", "Configuration (Heterogeneity)"]

# ── path helpers ──────────────────────────────────────────────────────────────

def _get_imp_path(scale: int, scope: str, cz: int | None) -> Path:
    if scope == "global":
        return BASE_SHAP / f"Global_scale_{scale}m" / "tables" / "shap_relative_importance.csv"
    else:
        return BASE_SHAP / f"CZ_{cz}_scale_{scale}m" / "tables" / "shap_relative_importance.csv"

# ── main ──────────────────────────────────────────────────────────────────────

def make_evolution_plot() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data Collection
    records = []
    for scope, cz in COLUMN_SPECS:
        for scale in SCALES:
            imp_path = _get_imp_path(scale, scope, cz)
            if not imp_path.exists():
                print(f"Warning: Missing {imp_path}")
                continue
            
            df_imp = pd.read_csv(imp_path)
            
            # Use 'mean_abs_shap' to calculate true relative importance
            if "mean_abs_shap" not in df_imp.columns:
                continue
                
            df_imp["Category"] = df_imp["feature"].apply(categorize_feature)
            
            # Aggregate SHAP by Category
            cat_sum = df_imp.groupby("Category")["mean_abs_shap"].sum()
            total_sum = cat_sum.sum()
            
            # Convert to percentages
            for cat, val in cat_sum.items():
                records.append({
                    "Scope": f"{scope}_{cz}",
                    "Scale": scale,
                    "Category": cat,
                    "Relative_Importance_Pct": (val / total_sum) * 100
                })

    df_plot = pd.DataFrame(records)
    if df_plot.empty:
        print("No data collected. Check paths.")
        return

    # 2. Plotting Setup (1x3 Panel)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, dpi=300)
    sns.set_style("ticks")

    for i, (scope, cz) in enumerate(COLUMN_SPECS):
        ax = axes[i]
        scope_key = f"{scope}_{cz}"
        subset = df_plot[df_plot["Scope"] == scope_key]
        
        if subset.empty:
            continue
            
        # Pivot table for stackplot: index=Scale, columns=Category
        pivot_df = subset.pivot(index="Scale", columns="Category", values="Relative_Importance_Pct").fillna(0)
        
        # Ensure consistent column order
        plot_cols = [c for c in CAT_ORDER if c in pivot_df.columns]
        y_data = [pivot_df[c].values for c in plot_cols]
        colors = [CAT_COLORS[c] for c in plot_cols]
        
        # Draw 100% Stacked Area
        ax.stackplot(pivot_df.index, y_data, labels=plot_cols, colors=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # Formatting the axes
        ax.set_xlim(min(SCALES), max(SCALES))
        ax.set_ylim(0, 100)
        
        # Ticks: actual physical distance (100 -> 1000)
        ax.set_xticks(SCALES)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        
        # Title and Labels
        ax.set_title(f"({chr(97+i)}) {COL_TITLES[i]}", loc='left', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Grid Scale (m)", fontsize=11, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel("Relative SHAP Importance (%)", fontsize=11, fontweight='bold')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            
        # Refine borders
        sns.despine(ax=ax, top=True, right=True)

    # 3. Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Reverse order for legend to match the physical stacking (bottom to top)
    fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.1) # Leave space for the global legend
    
    # 4. Save
    out_pdf = OUTPUT_DIR / "shap_category_evolution.pdf"
    out_png = OUTPUT_DIR / "shap_category_evolution.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[paper_vis_importance_evolution] Saved → {out_pdf}")

if __name__ == "__main__":
    make_evolution_plot()