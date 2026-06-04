import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import string

matplotlib.use("Agg")

# ── 1. Configuration & Paths ──────────────────────────────────────────────────
BASE_DIR = Path("../../../data/results")
OUTPUT_DIR = Path("../../../data/results/paper_figures")

# 注意：请确保这里的路径与你的实际 SHAP 结果路径一致
BASE_SHAP = BASE_DIR / "shap_results/selected"  

SCALES = [100, 250, 500, 750, 1000]
SCOPES = ["Global", "CZ15", "CZ26"]
SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

# ── 2. Data Loading Functions ─────────────────────────────────────────────────
def get_metrics_path(model, scope, scale):
    if model == "MLR":
        if scope == "Global":
            return BASE_DIR / f"mlr_results/Global/scale_{scale}m/metrics/test_metrics.json"
        elif scope == "CZ15":
            return BASE_DIR / f"mlr_results/Zone_15/scale_{scale}m/metrics/test_metrics.json"
        elif scope == "CZ26":
            return BASE_DIR / f"mlr_results/Zone_26/scale_{scale}m/metrics/test_metrics.json"
    elif model == "LightGBM":
        if scope == "Global":
            return BASE_DIR / f"lgbm_results_selected/global/scale_{scale}m/metrics/test_metrics.json"
        elif scope == "CZ15":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_15/scale_{scale}m/metrics/test_metrics.json"
        elif scope == "CZ26":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_26/scale_{scale}m/metrics/test_metrics.json"

def categorize_feature(feat_name: str) -> str:
    """Classify raw features into macro theoretical categories."""
    if "PLAND" in feat_name or feat_name in ["isa_fraction"]:
        return "Composition"
    elif any(x in feat_name for x in ["ED_", "PD_", "LSI_", "LPI_", "SHDI", "CONTAG"]):
        return "Configuration"
    elif feat_name in ["building_height_mean", "pop_sum", "BCR"]:
        return "3D & Socio-economic"
    else:
        return "Other"

def get_shap_path(scope, scale):
    if scope == "Global":
        return BASE_SHAP / f"Global_scale_{scale}m/tables/shap_relative_importance.csv"
    elif scope == "CZ15":
        return BASE_SHAP / f"CZ_15_scale_{scale}m/tables/shap_relative_importance.csv"
    elif scope == "CZ26":
        return BASE_SHAP / f"CZ_26_scale_{scale}m/tables/shap_relative_importance.csv"

# ── 3. Extract Data ───────────────────────────────────────────────────────────
# 3.1 Metrics Data
records_metrics = []
for model in ["MLR", "LightGBM"]:
    for scope in SCOPES:
        for scale in SCALES:
            path = get_metrics_path(model, scope, scale)
            if path and path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    records_metrics.append({
                        "Model": model, "Scope": scope, "Scale": scale,
                        "R2": data.get("r2"), "RMSE": data.get("rmse")
                    })
df_metrics = pd.DataFrame(records_metrics)

# 3.2 SHAP Importance Data
records_shap = []
for scope in SCOPES:
    for scale in SCALES:
        path = get_shap_path(scope, scale)
        if path and path.exists():
            df_imp = pd.read_csv(path)
            if "mean_abs_shap" in df_imp.columns:
                df_imp["Category"] = df_imp["feature"].apply(categorize_feature)
                cat_sum = df_imp.groupby("Category")["mean_abs_shap"].sum()
                total_sum = cat_sum.sum()
                for cat, val in cat_sum.items():
                    records_shap.append({
                        "Scope": scope, "Scale": scale, "Category": cat,
                        "Rel_Imp": (val / total_sum) * 100
                    })
df_shap = pd.DataFrame(records_shap)

# ── 4. Plotting Setup ─────────────────────────────────────────────────────────
sns.set_theme(style="ticks")
# plt.rcParams['font.family'] = 'Arial'

PALETTE_MODELS = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}
MARKERS_MODELS = {"LightGBM": "o", "MLR": "s"}

CAT_COLORS = {
    "Composition": "#5E81ACFF",
    "3D & Socio-economic": "#A3BE8CFF",
    "Configuration": "#D08670FF"
}
CAT_ORDER = ["Composition", "3D & Socio-economic", "Configuration"]

Y_LABELS = [r"$\mathbf{R^2}$", r"$\mathbf{RMSE\ (^\circ C)}$", r"$\mathbf{Relative\ SHAP\ (\%)}$"]
METRICS = ["R2", "RMSE"]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.5, 7.5), sharex=True, dpi=300)
sub_labels = list(string.ascii_lowercase)

# ── 5. Main Plotting Loop ─────────────────────────────────────────────────────
for col_idx, scope in enumerate(SCOPES):
    
    # --- Row 0 & 1: Line Plots (R2 and RMSE) ---
    for row_idx, metric in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        subset = df_metrics[df_metrics["Scope"] == scope]
        
        if not subset.empty:
            sns.lineplot(
                data=subset, x="Scale", y=metric, hue="Model", style="Model",
                markers=MARKERS_MODELS, dashes={"LightGBM": "", "MLR": (4, 1.5)},
                palette=PALETTE_MODELS, linewidth=1.5, markersize=4, ax=ax,
                legend=False # Legend handles extracted later
            )
            
        # Optimal scale highlight
        ax.axvline(250, color='gray', linestyle=':', linewidth=1.2, alpha=0.6, zorder=0)

        # Y-axis limits
        if metric == "R2":
            ax.set_ylim(0.45, 0.85)
        else:
            ax.set_ylim(0.9, 2.2)

    # --- Row 2: Stacked Area Plot (SHAP Categories) ---
    ax_area = axes[2, col_idx]
    subset_shap = df_shap[df_shap["Scope"] == scope]
    
    if not subset_shap.empty:
        pivot_df = subset_shap.pivot(index="Scale", columns="Category", values="Rel_Imp").fillna(0)
        plot_cols = [c for c in CAT_ORDER if c in pivot_df.columns]
        y_data = [pivot_df[c].values for c in plot_cols]
        colors = [CAT_COLORS[c] for c in plot_cols]
        
        ax_area.stackplot(pivot_df.index, y_data, labels=plot_cols, colors=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    
    ax_area.axvline(250, color='black', linestyle=':', linewidth=1.5, alpha=0.8, zorder=4)
    ax_area.set_ylim(0, 100)
    ax_area.set_xlim(min(SCALES), max(SCALES))

# ── 6. Formatting & Aesthetics ────────────────────────────────────────────────
FRAME_WIDTH = 0.8
for row_idx in range(3):
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        
        # Column Titles (Top row only)
        if row_idx == 0:
            ax.set_title(SCOPE_TITLES[col_idx], fontsize=11, fontweight="bold", pad=12)
            
        # Y-axis Labels (Left column only)
        ax.set_ylabel(Y_LABELS[row_idx] if col_idx == 0 else "", fontsize=10, fontweight="bold")
        
        # X-axis Labels (Bottom row only)
        ax.set_xlabel("Grid Scale (m)" if row_idx == 2 else "", fontsize=10, fontweight="bold")
        ax.set_xticks(SCALES)
        
        # Full borders and ticks
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(FRAME_WIDTH)
        ax.tick_params(axis="both", labelsize=9, width=FRAME_WIDTH, length=4)
        
        # Subplot letters
        l_idx = row_idx * 3 + col_idx
        ax.text(0.04, 0.96 if row_idx != 2 else 0.04, f"({sub_labels[l_idx]})", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', va='top' if row_idx != 2 else 'bottom', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# ── 7. Custom Dual Legends at Bottom (Single Row Layout) ──────────────────────
# Extract lines legend
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

legend_elements_line = [
    Line2D([0], [0], color=PALETTE_MODELS["MLR"], lw=1.5, ls="--", marker="s", markersize=4, label="MLR"),
    Line2D([0], [0], color=PALETTE_MODELS["LightGBM"], lw=1.5, ls="-", marker="o", markersize=4, label="LightGBM")
]

legend_elements_area = [
    mpatches.Patch(facecolor=CAT_COLORS[c], label=c, edgecolor='white', lw=0.5) for c in CAT_ORDER
]

# 第一个图例 (Model Performance) 锚定在左下角
leg1 = fig.legend(handles=legend_elements_line, loc='upper left', bbox_to_anchor=(0.1, 0.01), 
                  ncol=2, frameon=False, fontsize=9, title=r"$\mathbf{Model\ Performance}$")
leg1.get_title().set_fontsize('9')

# 第二个图例 (Feature Category) 锚定在右下角
# 使用 columnspacing 和 handletextpad 稍微收缩间距，确保 8.5 英寸画布内不拥挤
leg2 = fig.legend(handles=legend_elements_area, loc='upper right', bbox_to_anchor=(0.9, 0.01), 
                  ncol=3, frameon=False, fontsize=9, title=r"$\mathbf{Feature\ Category\ (LightGBM)}$",
                  columnspacing=1.0, handletextpad=0.5)
leg2.get_title().set_fontsize('9')

# ── 8. Save workflow ──────────────────────────────────────────────────────────
# 调整 rect 保护底部的图例空间 (bottom 设置为 0.1 即可，因为在同一行)
plt.tight_layout(rect=[0, 0, 1, 1])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_pdf = OUTPUT_DIR / "Master_Scale_Selection_3x3.pdf"
out_png = OUTPUT_DIR / "Master_Scale_Selection_3x3.png"

# 使用 bbox_inches="tight" 让 matplotlib 自动计算边缘包围盒，确保不切断任何文本
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved {out_pdf} and {out_png}")