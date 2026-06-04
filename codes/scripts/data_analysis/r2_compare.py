import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import string

matplotlib.use("Agg")

BASE_DIR = Path("../../../data/results")
OUTPUT_DIR = Path("../../../data/results/paper_figures")

SCALES = [100, 250, 500, 750, 1000]

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

records = []
for model in ["MLR", "LightGBM"]:
    for scope in ["Global", "CZ15", "CZ26"]:
        for scale in SCALES:
            path = get_metrics_path(model, scope, scale)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    records.append({
                        "Model": model,
                        "Scope": scope,
                        "Scale": scale,
                        "R2": data.get("r2"),
                        "RMSE": data.get("rmse")
                    })
            else:
                print(f"Warning: {path} not found.")

df = pd.DataFrame(records)

# Seaborn global canvas default configuration
sns.set_theme(style="ticks")

PALETTE = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}
MARKERS = {"LightGBM": "o", "MLR": "s"}
SCOPES = ["Global", "CZ15", "CZ26"]
SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]
METRICS = ["R2", "RMSE"]

# FIX 1: Use math bold (\mathbf) for R^2 to prevent LaTeX engine from bypassing fontweight="bold"
Y_LABELS = [r"$\mathbf{R^2}$", r"$\mathbf{RMSE\ (^\circ C)}$"]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6.5, 5), dpi=300)
sub_labels = list(string.ascii_lowercase)

# 1. Main Plotting Grid Loop
for row_idx, metric in enumerate(METRICS):
    for col_idx, scope in enumerate(SCOPES):
        ax = axes[row_idx, col_idx]
        subset = df[df["Scope"] == scope]
        
        if not subset.empty:
            sns.lineplot(
                data=subset, x="Scale", y=metric, hue="Model", style="Model",
                markers=MARKERS, dashes={"LightGBM": "", "MLR": (4, 1.5)},
                palette=PALETTE, linewidth=1.5, markersize=4, ax=ax,
                legend=(row_idx == 0 and col_idx == 0)
            )
            
        # Title and Labels setup
        if row_idx == 0:
            ax.set_title(SCOPE_TITLES[col_idx], fontsize=10, fontweight="bold", pad=15)
            
        ax.set_ylabel(Y_LABELS[row_idx] if col_idx == 0 else "", fontsize=10, fontweight="bold")
        ax.set_xlabel("Scale (m)" if row_idx == 1 else "", fontsize=10, fontweight="bold")
        ax.set_xticks(SCALES)
        
        if row_idx == 0:
            ax.tick_params(axis="x", labelbottom=False) # Safely hide top-row labels without resetting ticks
            
        ax.set_ylim((0.55, 0.85) if metric == "R2" else (1.0, 2.2))
        
        # Subplot text tracking
        l_idx = row_idx * 3 + col_idx
        ax.text(0.04, 0.96, f"({sub_labels[l_idx]})", transform=ax.transAxes, 
                fontsize=10, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# 2. Extract global legend and strip the temporary subplot legend object
handles, labels = axes[0, 0].get_legend_handles_labels()
if axes[0, 0].get_legend():
    axes[0, 0].get_legend().remove()

# 3. FIX 2: Post-processing block to enforce full-frame borders & refresh axis labels
FRAME_WIDTH = 0.6  # Control your box frame thickness here
for r in range(2):
    for c in range(3):
        ax = axes[r, c]
        # Re-verify and lock the axis label properties to patch any matplotlib lifecycle bugs
        ax.set_ylabel(Y_LABELS[r] if c == 0 else "", fontsize=10, fontweight="bold")
        
        # Draw explicit full borders (box style)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(FRAME_WIDTH)
            
        # Inward-pointing ticks mirrored to top and right axes
        ax.tick_params(axis="both", labelsize=9, width=FRAME_WIDTH, length=3)

# Place the unified legend outside the plot grid frame
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10, 
           bbox_to_anchor=(0.5, -0.02), frameon=False)

# 4. Save workflow (rect constraints secure the legend margin)
plt.tight_layout(rect=[0, 0.05, 1, 1])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_pdf = OUTPUT_DIR / "model_comparison_r2_rmse.pdf"
out_png = OUTPUT_DIR / "model_comparison_r2_rmse.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_pdf} and {out_png}")

# sns.set_theme(style="ticks")
# PALETTE = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}
# MARKERS = {"LightGBM": "o", "MLR": "s"}

# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6.5, 5), dpi=300)
# sub_labels = list(string.ascii_lowercase)

# SCOPES = ["Global", "CZ15", "CZ26"]
# SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]
# METRICS = ["R2", "RMSE"]
# Y_LABELS = ["$R^2$", "RMSE ($^\circ$C)"]

# for row_idx, metric in enumerate(METRICS):
#     for col_idx, scope in enumerate(SCOPES):
#         ax = axes[row_idx, col_idx]
        
#         subset = df[df["Scope"] == scope].copy()
        
#         if not subset.empty:
#             sns.lineplot(
#                 data=subset,
#                 x="Scale",
#                 y=metric,
#                 hue="Model",
#                 style="Model",
#                 markers=MARKERS,
#                 dashes={"LightGBM": "", "MLR": (4, 1.5)},
#                 palette=PALETTE,
#                 linewidth=1.5,
#                 markersize=4,
#                 ax=ax,
#                 legend=(row_idx==0 and col_idx==0)
#             )
            
#         if row_idx == 0:
#             ax.set_title(SCOPE_TITLES[col_idx], fontsize=10, fontweight="bold", pad=15)
            
#         if col_idx == 0:
#             ax.set_ylabel(Y_LABELS[row_idx], fontsize=10, fontweight="bold")
#         else:
#             ax.set_ylabel("")
        
#         if row_idx == 1:
#             ax.set_xlabel("Scale (m)", fontsize=10, fontweight="bold")
#         else:
#             ax.set_xlabel("")
#             ax.set_xticklabels([])
            
#         ax.set_xticks(SCALES)
        
#         if metric == "R2":
#             ax.set_ylim(0.55, 0.85)
#         else:
#             ax.set_ylim(1.0, 2.2)

#         ax.tick_params(axis="both", labelsize=9, width=0.4, length=3)
        
#         label_idx = row_idx * 3 + col_idx
#         ax.text(0.04, 0.96, f"({sub_labels[label_idx]})", 
#                 transform=ax.transAxes, 
#                 fontsize=11, 
#                 fontweight='bold', 
#                 va='top', 
#                 ha='left',
#                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# handles, labels = axes[0,0].get_legend_handles_labels()
# axes[0,0].get_legend().remove()

# fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05), frameon=False)

# # sns.despine(fig)
# plt.tight_layout()

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# out_pdf = OUTPUT_DIR / "model_comparison_r2_rmse.pdf"
# out_png = OUTPUT_DIR / "model_comparison_r2_rmse.png"

# fig.savefig(out_pdf, bbox_inches="tight")
# fig.savefig(out_png, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved {out_pdf} and {out_png}")



# Ensure global Times New Roman style and Stix math layout
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
#     'mathtext.fontset': 'stix'
# })