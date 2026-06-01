import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import string

matplotlib.use("Agg")

BASE_DIR = Path("../../../data/results")
OUTPUT_DIR = Path("../../../data/results/paper_figures")

SCALE = 250

def get_predictions_path(model, scope):
    if model == "MLR":
        if scope == "Global":
            return BASE_DIR / f"mlr_results/Global/scale_{SCALE}m/folds/test_predictions.parquet"
        elif scope == "CZ15":
            return BASE_DIR / f"mlr_results/Zone_15/scale_{SCALE}m/folds/test_predictions.parquet"
        elif scope == "CZ26":
            return BASE_DIR / f"mlr_results/Zone_26/scale_{SCALE}m/folds/test_predictions.parquet"
    elif model == "LightGBM":
        if scope == "Global":
            return BASE_DIR / f"lgbm_results_selected/global/scale_{SCALE}m/folds/test_predictions.parquet"
        elif scope == "CZ15":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_15/scale_{SCALE}m/folds/test_predictions.parquet"
        elif scope == "CZ26":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_26/scale_{SCALE}m/folds/test_predictions.parquet"

df_list = []
for model in ["MLR", "LightGBM"]:
    for scope in ["Global", "CZ15", "CZ26"]:
        path = get_predictions_path(model, scope)
        if path and path.exists():
            df = pd.read_parquet(path)
            if "residual" in df.columns:
                subset = df[["residual"]].copy()
                subset["Model"] = model
                subset["Scope"] = scope
                df_list.append(subset)
            elif "y_true" in df.columns and "y_pred" in df.columns:
                subset = pd.DataFrame()
                subset["residual"] = df["y_true"] - df["y_pred"]
                subset["Model"] = model
                subset["Scope"] = scope
                df_list.append(subset)
            else:
                print(f"Warning: {path} has no residual column.")
        else:
            print(f"Warning: {path} not found.")

if not df_list:
    print("No data found.")
    exit(1)

df_all = pd.concat(df_list, ignore_index=True)

sns.set_theme(style="ticks")
PALETTE = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
sub_labels = list(string.ascii_lowercase)

SCOPES = ["Global", "CZ15", "CZ26"]
SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

# Calculate the index of the rightmost plot
last_idx = len(SCOPES) - 1

for i, scope in enumerate(SCOPES):
    ax = axes[i]
    df_scope = df_all[df_all["Scope"] == scope]
    
    if not df_scope.empty:
        # Plot KDEs with fill and transparency so they overlap nicely
        sns.kdeplot(
            data=df_scope, 
            x="residual", 
            hue="Model", 
            fill=True, 
            alpha=0.5, 
            linewidth=2,
            palette=PALETTE,
            common_norm=False,
            ax=ax,
            # 🟢 修正 1：仅在最后一张图（最右边）生成图例
            legend=(i == last_idx)  
        )
    
    ax.set_title(SCOPE_TITLES[i], fontsize=14, fontweight="bold", pad=15)
    
    # Label formatting depending on column position
    if i == 0:
        ax.set_ylabel("Density", fontsize=13)
    else:
        ax.set_ylabel("")
        
    # 🟢 修正 2：仅在最后一张图上配置和移动图例
    if i == last_idx:
        # Move the legend to the upper right of the rightmost plot
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.02, 1), title=None, frameon=False, fontsize=12)
        
    ax.set_xlabel("Residual (°C)", fontsize=13) # Removed LaTeX math mode for simple unit text
    ax.tick_params(axis="both", labelsize=11)
    
    # 0 degree reference line
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    
    ax.text(0.04, 0.96, f"({sub_labels[i]})", 
            transform=ax.transAxes, 
            fontsize=16, 
            fontweight='bold', 
            va='top', 
            ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

sns.despine(fig)
plt.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_pdf = OUTPUT_DIR / f"residual_distribution_{SCALE}m.pdf"
out_png = OUTPUT_DIR / f"residual_distribution_{SCALE}m.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_pdf} and {out_png}")