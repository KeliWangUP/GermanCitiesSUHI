import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import string

matplotlib.use("Agg")

UNIFIED_DIR = Path("../../../data/unified_scale_matrices").resolve()
OUTPUT_DIR = Path("../../../data/results/paper_figures").resolve()

SCALES = [100, 250, 500, 750, 1000]

df_list = []
for scale in SCALES:
    file_path = UNIFIED_DIR / f"merged_metrics_{scale}m.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
        if "SUHI" in df.columns and "CZ_median" in df.columns:
            subset = df[["SUHI", "CZ_median"]].dropna()
            
            # Global
            df_global = subset.copy()
            df_global["Scope"] = "Global"
            
            # CZ 15
            df_cz15 = subset[subset["CZ_median"] == 15].copy()
            df_cz15["Scope"] = "CZ15"
            
            # CZ 26
            df_cz26 = subset[subset["CZ_median"] == 26].copy()
            df_cz26["Scope"] = "CZ26"
            
            scale_df = pd.concat([df_global, df_cz15, df_cz26], ignore_index=True)
            scale_df["Scale"] = scale
            df_list.append(scale_df)

if not df_list:
    print("No data found.")
    exit(1)

df_all = pd.concat(df_list, ignore_index=True)

fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=150)
sns.set_theme(style="ticks")

sub_labels = list(string.ascii_lowercase)

y_min = df_all["SUHI"].min() - 0.5
y_max = df_all["SUHI"].max() + 0.5

# We can use a palette for the 3 scopes
color_palette = ["#B48EAD", "#EBCB8B", "#88C0D0"]

for i, scale in enumerate(SCALES):
    ax = axes[i]
    df_scale = df_all[df_all["Scale"] == scale]
    
    sns.violinplot(
        data=df_scale, 
        x="Scope", 
        y="SUHI",
        hue="Scope",
        ax=ax, 
        palette=color_palette, 
        inner="box", 
        linewidth=1.2,
        cut=0,
        density_norm="area",
        legend=False
    )
    
    ax.set_title(f"{scale} m", fontsize=14, fontweight="bold", pad=15)
    
    if i == 0:
        ax.set_ylabel("SUHI ($^\circ$C)", fontsize=13)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(y_min, y_max)
    
    # 0 degree reference line
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
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
out_pdf = OUTPUT_DIR / "suhi_distribution_by_scale.pdf"
out_png = OUTPUT_DIR / "suhi_distribution_by_scale.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_pdf} and {out_png}")
