import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import string

matplotlib.use("Agg")

BASE_DIR = Path("../../../data/results")
OUTPUT_DIR = Path("../../../data/results/paper_figures")

SCALES = [100, 250, 500, 750, 1000]

def get_predictions_path(model, scope, scale):
    if model == "MLR":
        if scope == "Global":
            return BASE_DIR / f"mlr_results/Global/scale_{scale}m/folds/test_predictions.parquet"
        elif scope == "CZ15":
            return BASE_DIR / f"mlr_results/Zone_15/scale_{scale}m/folds/test_predictions.parquet"
        elif scope == "CZ26":
            return BASE_DIR / f"mlr_results/Zone_26/scale_{scale}m/folds/test_predictions.parquet"
    elif model == "LightGBM":
        if scope == "Global":
            return BASE_DIR / f"lgbm_results_selected/global/scale_{scale}m/folds/test_predictions.parquet"
        elif scope == "CZ15":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_15/scale_{scale}m/folds/test_predictions.parquet"
        elif scope == "CZ26":
            return BASE_DIR / f"lgbm_results_selected/by_climate_zone/CZ_26/scale_{scale}m/folds/test_predictions.parquet"

def calculate_morans_i_knn(coords_xy, values, k=8):
    if len(values) <= k:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords_xy)
    _, indices = nn.kneighbors(coords_xy)
    neighbors_indices = indices[:, 1:]
    
    x = np.asarray(values)
    dx = x - np.mean(x)
    var = np.sum(dx**2)
    if var == 0: 
        return np.nan
    
    dx_j = dx[neighbors_indices]
    spatial_lag = np.sum(dx_j, axis=1) / k
    I = np.sum(dx * spatial_lag) / var
    return float(I)

records = []

for model in ["MLR", "LightGBM"]:
    for scope in ["Global", "CZ15", "CZ26"]:
        for scale in SCALES:
            path = get_predictions_path(model, scope, scale)
            if path and path.exists():
                print(f"Processing {model} | {scope} | {scale}m")
                df = pd.read_parquet(path)
                
                # Identify residual column
                if "residual" in df.columns:
                    resid_col = "residual"
                elif "y_true" in df.columns and "y_pred" in df.columns:
                    df["computed_residual"] = df["y_true"] - df["y_pred"]
                    resid_col = "computed_residual"
                else:
                    print(f"No residual column found for {path}")
                    continue
                
                # Convert geometry to shapely logic
                if isinstance(df['geometry'].iloc[0], str):
                    try:
                        geoms = gpd.GeoSeries.from_wkb(df['geometry'].apply(bytes.fromhex))
                    except Exception:
                        geoms = gpd.GeoSeries.from_wkt(df['geometry'])
                else:
                    geoms = gpd.GeoSeries(df['geometry'])
                
                df['x'] = geoms.centroid.x
                df['y'] = geoms.centroid.y
                
                for city_id, group in df.groupby("city_id"):
                    coords = group[['x', 'y']].values
                    vals = group[resid_col].values
                    
                    if len(vals) > 8:
                        mi = calculate_morans_i_knn(coords, vals, k=8)
                        records.append({
                            "Model": model,
                            "Scope": scope,
                            "Scale": scale,
                            "City_ID": city_id,
                            "Morans_I": mi
                        })

# Save the tabular data
out_df = pd.DataFrame(records).dropna()
out_csv = BASE_DIR / "residual_morans_i_city_level.csv"
out_df.to_csv(out_csv, index=False)
print(f"\nCalculated Moran's I, saved to {out_csv}")

# Visualization 
sns.set_theme(style="ticks")
PALETTE = {"LightGBM": "#BF616A", "MLR": "#5E81AC"}

# 2x3 Plot similar to R2/RMSE
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
sub_labels = list(string.ascii_lowercase)

SCOPES = ["Global", "CZ15", "CZ26"]
SCOPE_TITLES = ["Global", "Climate Zone 15", "Climate Zone 26"]

for i, scope in enumerate(SCOPES):
    ax = axes[i]
    subset = out_df[out_df["Scope"] == scope]
    
    if not subset.empty:
        sns.boxplot(
            data=subset,
            x="Scale",
            y="Morans_I",
            hue="Model",
            palette=PALETTE,
            ax=ax,
            showmeans=True,
            meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6},
            fliersize=2,
            linewidth=1.2,
            legend=(i==0)
        )
    
    ax.set_title(SCOPE_TITLES[i], fontsize=14, fontweight="bold", pad=15)
    
    if i == 0:
        ax.set_ylabel("Moran's $I$ (Residuals)", fontsize=13)
        sns.move_legend(ax, "upper right", title=None, frameon=False, fontsize=12)
    else:
        ax.set_ylabel("")
        
    ax.set_xlabel("Scale (m)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    
    # Optional baseline 0 autocorrelation reference
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    
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
out_pdf = OUTPUT_DIR / "residual_morans_i_boxplots.pdf"
out_png = OUTPUT_DIR / "residual_morans_i_boxplots.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to {out_png}")
