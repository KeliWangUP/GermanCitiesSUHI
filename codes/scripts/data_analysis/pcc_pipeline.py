import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import math
import string

GRID_SIZES = [100, 250, 500, 750, 1000]
UNIFIED_DIR = Path("../../../data/unified_scale_matrices").resolve()
TARGET_VAR = "SUHI"

name_mapping_dict = {'10': 'tree_cover', 
                     '20': 'shrubland',
                     '30': 'grassland',
                     '40': 'cropland',
                     '50': 'built_up',
                     '60': 'bare_land',
                     '70': 'snow_and_ice',
                     '80': 'water',
                     '90': 'wetland'}

SELECTED_FEATURES = [
    "PLAND_tree_cover", "PLAND_built_up", "PLAND_grassland", "PLAND_water",
    "ED_tree_cover", "ED_built_up", "ED_grassland", "ED_water",
    "PD_tree_cover", "PD_built_up", "PD_grassland", "PD_water",
    "LSI_tree_cover", "LSI_built_up", "LSI_grassland", "LSI_water",
    "LPI_tree_cover", "LPI_built_up", "LPI_grassland", "LPI_water",
    "SHDI", "CONTAG", 
    "isa_fraction", "building_height_mean", "pop_sum", "BCR"
]

FEAT_LABELS: dict[str, str] = {
    "isa_fraction":           "ISA fraction",
    "PLAND_tree_cover":       "PLAND tree cover",
    "PLAND_built_up":         "PLAND built-up",
    "PLAND_grassland":        "PLAND grassland",
    "PLAND_water":            "PLAND water",
    "ED_tree_cover":          "ED tree cover",
    "ED_built_up":            "ED built-up",
    "ED_grassland":           "ED grassland",
    "ED_water":               "ED water",
    "PD_tree_cover":          "PD tree cover",
    "PD_built_up":            "PD built-up",
    "PD_grassland":           "PD grassland",
    "PD_water":               "PD water",
    "LSI_tree_cover":         "LSI tree cover",
    "LSI_built_up":           "LSI built-up",
    "LSI_grassland":          "LSI grassland",
    "LSI_water":              "LSI water",
    "LPI_tree_cover":         "LPI tree cover",
    "LPI_built_up":           "LPI built-up",
    "LPI_grassland":          "LPI grassland",
    "LPI_water":              "LPI water",
    "building_height_mean":   "Building Height",
    "BCR":                    "BCR",
    "CONTAG":                 "CONTAG",
    "pop_sum":                "Population",
    "SHDI":                   "SHDI",
}

def get_label(feat: str) -> str:
    return FEAT_LABELS.get(feat, feat)

feature_groups = {"Composition (PLAND)": ["PLAND_tree_cover", "PLAND_built_up", "PLAND_grassland", "PLAND_water"],
                  "Configuration (ED)": ["ED_tree_cover", "ED_built_up", "ED_grassland", "ED_water"],
                  "Configuration (PD)": ["PD_tree_cover", "PD_built_up", "PD_grassland", "PD_water"],
                  "Configuration (LSI)": ["LSI_tree_cover", "LSI_built_up", "LSI_grassland", "LSI_water"],
                  "Configuration (LPI)": ["LPI_tree_cover", "LPI_built_up", "LPI_grassland", "LPI_water"],
                  "Diversity & Complexity": ["SHDI", "CONTAG"],
                  "Construction & 3D": ["isa_fraction", "BCR", "building_height_mean"],
                  "Socio-economics": ["pop_sum"]}

results_global = []
results_cz15 = []
results_cz26 = []

for scale in GRID_SIZES:
    file_path = UNIFIED_DIR / f"merged_metrics_{scale}m.parquet"
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        continue
        
    df = pd.read_parquet(file_path)
    new_column_names = {}
    for old_name in df.columns:
        if '_cls_' in old_name:
            suffix = old_name.split('_cls_')[-1]
            new_suffix = name_mapping_dict.get(suffix, suffix)
            new_name = old_name.replace(f'_cls_{suffix}', f'_{new_suffix}')
            new_column_names[old_name] = new_name

    df = df.rename(columns=new_column_names)
    
    cols_to_use = [TARGET_VAR] + (["CZ_median"] if "CZ_median" in df.columns else []) + [f for f in SELECTED_FEATURES if f in df.columns]
    
    df_clean = df[cols_to_use].dropna(subset=[TARGET_VAR])
    
    df_global = df_clean
    df_cz15 = df_clean[df_clean["CZ_median"] == 15] if "CZ_median" in df_clean.columns else pd.DataFrame()
    df_cz26 = df_clean[df_clean["CZ_median"] == 26] if "CZ_median" in df_clean.columns else pd.DataFrame()
    
    for feature in SELECTED_FEATURES:
        if feature in df_clean.columns:
            # Global
            sub_global = df_global[[feature, TARGET_VAR]].dropna()
            if len(sub_global) > 2:
                r_val, p_val = pearsonr(sub_global[feature], sub_global[TARGET_VAR])
                results_global.append({"Scale": scale, "Feature": feature, "Correlation": r_val, "P_value": p_val, "Is_Significant": p_val < 0.05, "Scope": "Global"})
            
            # CZ15
            sub_cz15 = df_cz15[[feature, TARGET_VAR]].dropna()
            if len(sub_cz15) > 2:
                r_val, p_val = pearsonr(sub_cz15[feature], sub_cz15[TARGET_VAR])
                results_cz15.append({"Scale": scale, "Feature": feature, "Correlation": r_val, "P_value": p_val, "Is_Significant": p_val < 0.05, "Scope": "CZ15"})
            
            # CZ26
            sub_cz26 = df_cz26[[feature, TARGET_VAR]].dropna()
            if len(sub_cz26) > 2:
                r_val, p_val = pearsonr(sub_cz26[feature], sub_cz26[TARGET_VAR])
                results_cz26.append({"Scale": scale, "Feature": feature, "Correlation": r_val, "P_value": p_val, "Is_Significant": p_val < 0.05, "Scope": "CZ26"})

MORANDI_PALETTE = ["#A3BE8C", "#BF616A", "#B48EAD", "#5E81AC", "#D08770"]

def plot_pcc_results(results_list, scope_name):
    if not results_list:
        print(f"No results to plot for {scope_name}")
        return
        
    plot_df = pd.DataFrame(results_list)
    # Apply friendly labels
    plot_df["FeatureLabel"] = plot_df["Feature"].apply(get_label)
    
    sns.set_palette(MORANDI_PALETTE)
    cols = 4
    rows = math.ceil(len(feature_groups) / cols)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(13, 5 * rows), dpi=150)
    sns.set_style("ticks")

    axes_flat = axes.flatten()
    sub_labels = list(string.ascii_lowercase)

    for i, (group_name, features) in enumerate(feature_groups.items()):
        ax = axes_flat[i]
        row_idx = i // cols
        col_idx = i % cols
        
        group_data = plot_df[plot_df["Feature"].isin(features)]
        
        if not group_data.empty:
            df_sig = group_data[group_data['Is_Significant'] == True]
            df_nonsig = group_data[group_data['Is_Significant'] == False]

            if not df_sig.empty:
                sns.lineplot(data=df_sig, x="Scale", y="Correlation", hue="FeatureLabel", 
                             marker="o", ax=ax, palette=MORANDI_PALETTE[:len(df_sig["FeatureLabel"].unique())], linewidth=2, markersize=5.5)

            if not df_nonsig.empty:
                for feat, sub in df_nonsig.groupby("FeatureLabel"):
                    ax.plot(sub['Scale'], sub['Correlation'], linestyle='--', marker='o', markersize=4, color='0.6', alpha=0.7)

            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(group_name, fontsize=13, pad=10)
            ax.set_xticks(GRID_SIZES)
            ax.set_ylim(-1, 1)
            
            ax.text(0.03, 0.98, f"({sub_labels[i]})", 
                    transform=ax.transAxes, 
                    fontsize=16, 
                    fontweight='bold', 
                    va='top', 
                    ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            if col_idx == 0:
                ax.set_ylabel("Pearson r", fontsize=11)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                
            is_last_row = (row_idx == rows - 1)
            has_no_sub_plot_below = (i + cols >= len(feature_groups))
            
            if is_last_row or has_no_sub_plot_below:
                ax.set_xlabel("Scale (m)", fontsize=11)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([]) 
                
            leg = ax.legend(fontsize=9, loc='lower right', frameon=False)
            if leg is not None:
                for handle in leg.legend_handles:
                    handle.set_marker('None')
                leg.set_bbox_to_anchor((1.01, -0.01))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    sns.despine()
    fig.suptitle(f"Pearson Correlation with SUHI - {scope_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_pdf = f'../../../data/results/paper_figures/pcc_{scope_name}.pdf'
    out_png = f'../../../data/results/paper_figures/pcc_{scope_name}.png'
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', bbox_inches='tight')
    plt.close()
    print(f"Saved {out_pdf} and {out_png}")

print("Plotting Global...")
plot_pcc_results(results_global, "Global")
print("Plotting Climate Zone 15...")
plot_pcc_results(results_cz15, "Climate-Zone-15")
print("Plotting Climate Zone 26...")
plot_pcc_results(results_cz26, "Climate-Zone-26")

