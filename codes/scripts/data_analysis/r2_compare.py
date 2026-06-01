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

sns.set_theme(style="ticks")
PALETTE = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}
MARKERS = {"LightGBM": "o", "MLR": "s"}

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), dpi=150)
sub_labels = list(string.ascii_lowercase)

SCOPES = ["Global", "CZ15", "CZ26"]
SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]
METRICS = ["R2", "RMSE"]
Y_LABELS = ["$R^2$", "RMSE ($^\circ$C)"]

for row_idx, metric in enumerate(METRICS):
    for col_idx, scope in enumerate(SCOPES):
        ax = axes[row_idx, col_idx]
        
        subset = df[df["Scope"] == scope].copy()
        
        if not subset.empty:
            sns.lineplot(
                data=subset,
                x="Scale",
                y=metric,
                hue="Model",
                style="Model",
                markers=MARKERS,
                dashes={"LightGBM": "", "MLR": (4, 1.5)},
                palette=PALETTE,
                linewidth=2.5,
                markersize=8,
                ax=ax,
                legend=(row_idx==0 and col_idx==0)
            )
            
        if row_idx == 0:
            ax.set_title(SCOPE_TITLES[col_idx], fontsize=14, fontweight="bold", pad=15)
            
        if col_idx == 0:
            ax.set_ylabel(Y_LABELS[row_idx], fontsize=13)
        else:
            ax.set_ylabel("")
        
        if row_idx == 1:
            ax.set_xlabel("Scale (m)", fontsize=13)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            
        ax.set_xticks(SCALES)
        
        if metric == "R2":
            ax.set_ylim(0.4, 0.9)
        else:
            ax.set_ylim(1.0, 2.5)
            
        ax.tick_params(axis="both", labelsize=11)
        
        label_idx = row_idx * 3 + col_idx
        ax.text(0.04, 0.96, f"({sub_labels[label_idx]})", 
                transform=ax.transAxes, 
                fontsize=16, 
                fontweight='bold', 
                va='top', 
                ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,0].get_legend().remove()

fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.05), frameon=False)

sns.despine(fig)
plt.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_pdf = OUTPUT_DIR / "model_comparison_r2_rmse.pdf"
out_png = OUTPUT_DIR / "model_comparison_r2_rmse.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_pdf} and {out_png}")
