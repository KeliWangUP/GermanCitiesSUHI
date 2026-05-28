import pandas as pd
from pathlib import Path
from mlr_pipeline import compute_clean_ols
from vif import filter_features_by_vif_with_protection


def _parse_model_sort_key(model_key):
    subset_label, scale_label = model_key
    scale_value = int(scale_label.rstrip("m"))

    if subset_label == "Global":
        subset_rank = 0
        subset_value = 0
    elif subset_label.startswith("Zone_"):
        subset_rank = 1
        subset_value = subset_label.split("Zone_", 1)[1]
        try:
            subset_value = int(subset_value)
        except ValueError:
            subset_value = str(subset_value)
    else:
        subset_rank = 2
        subset_value = subset_label

    return (scale_value, subset_rank, subset_value)


def _ordered_feature_index(index_values, protected_features):
    protected_order = [feature for feature in protected_features if feature in index_values]
    remainder = [feature for feature in index_values if feature not in protected_order and feature != "_Model_R2"]
    remainder = sorted(remainder)

    ordered = []
    if "const" in index_values:
        ordered.append("const")
    ordered.extend(protected_order)
    ordered.extend(feature for feature in remainder if feature != "const")
    if "_Model_R2" in index_values:
        ordered.append("_Model_R2")
    return [feature for feature in ordered if feature in index_values]


def main():
    SCALES = [100, 250, 500, 750, 1000]
    UNIFIED_DIR = Path("../../../data/unified_scale_matrices").resolve()
    name_mapping_dict = {'10': 'tree_cover', 
                         '20': 'shrubland',
                         '30': 'grassland',
                         '40': 'cropland',
                         '50': 'built_up',
                         '60': 'bare_land',
                         '70': 'snow_and_ice',
                         '80': 'water',
                         '90': 'wetland'}
    names_to_drop = ['geometry',
                     'city_id',
                     'lst_mean',
                     'PLAND_cls_90',
                     'PD_cls_90',
                     'ED_cls_90',
                     'LPI_cls_90',
                     'LSI_cls_90',
                     'PLAND_cls_20',
                     'PD_cls_20',
                     'ED_cls_20',
                     'LPI_cls_20',
                     'LSI_cls_20',
                     'LST_Rural_mean']
    CORE_STORY_VARIABLES = ["PLAND_tree_cover", "building_height_mean", "pop_sum", "ED_tree_cover"]
    TARGET_VAR = "SUHI"

    

    # These dictionaries store aligned tables: {(Subset_Level, Scale_Level): {Feature: Value}}
    master_ols_dict = {}
    master_vif_dict = {}

    for scale in SCALES:
        print(f"\nProcessing scale: {scale}m")
        # Load scale-specific integrated dataframe
        file_path = UNIFIED_DIR / f"merged_metrics_{scale}m.parquet"
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue
        df_scale = pd.read_parquet(file_path)
        df_scale['CZ_median'] = df_scale['CZ_median'].dropna().astype(float).astype(int).astype(str)

        new_column_names = {}
        for old_name in df_scale.columns:
            if '_cls_' in old_name:
                suffix = old_name.split('_cls_')[-1]
                new_suffix = name_mapping_dict.get(suffix, suffix)  # 如果找不到映射，保留原样
                new_name = old_name.replace(f'_cls_{suffix}', f'_{new_suffix}')
                new_column_names[old_name] = new_name
        
        df_scale = df_scale.rename(columns=new_column_names)
        RAW_FEATURES = [f for f in df_scale.columns if f not in names_to_drop]
        # drop columns of CZ_median
        RAW_FEATURES = [f for f in RAW_FEATURES if f != 'CZ_median']
        df_clean = df_scale[RAW_FEATURES].dropna()

        if len(df_clean) > 10000:
            df_for_vif = df_clean.sample(n=10000, random_state=42)
            df_for_vif = df_for_vif[RAW_FEATURES].drop(columns=[TARGET_VAR], errors='ignore')
        else:
            df_for_vif = df_clean[RAW_FEATURES].drop(columns=[TARGET_VAR], errors='ignore')

        vif_candidates = [f for f in RAW_FEATURES if f != "SUHI"]

        # ------------------------------------------------------------------
        # 设定 1：全集 (Global)
        # ------------------------------------------------------------------
        # Dynamic VIF filtration (Assume your custom filter function is 'vif_filter')
        global_vif = filter_features_by_vif_with_protection(
            df_for_vif,
            vif_candidates,
            protected_features=CORE_STORY_VARIABLES,
            thresh=5.0,
        )
        features_global = list(global_vif.keys())
        global_res = compute_clean_ols(
            df_clean,
            features_global,
            target_col=TARGET_VAR,
            standardize_x=True,
        )

        if global_res:
            # Construct the multi-level tuple column key: (Subset_Level, Scale_Level)
            col_key = ("Global", f"{scale}m")
            master_ols_dict[col_key] = global_res
            master_vif_dict[col_key] = global_vif

        # ------------------------------------------------------------------
        # 设定 2 & 3：各气候区子集 (Subsets)
        # ------------------------------------------------------------------
        climate_zones = df_scale["CZ_median"].unique()
        for zone in climate_zones:
            print(f"\n  Processing climate zone: {zone} within scale: {scale}m")
            df_zone = df_scale[df_scale["CZ_median"] == zone].copy()
            df_zone = df_zone[RAW_FEATURES].dropna()
            if len(df_zone) > 10000:
                df_for_vif = df_zone.sample(n=10000, random_state=42)
                df_for_vif = df_for_vif[RAW_FEATURES].drop(columns=[TARGET_VAR], errors='ignore')
            else:
                df_for_vif = df_zone[RAW_FEATURES].drop(columns=[TARGET_VAR], errors='ignore')
            
            vif_candidates = [f for f in RAW_FEATURES if f != "SUHI"]

            # Re-run VIF independently since collinearity shifts across spatial zones!
            zone_vif = filter_features_by_vif_with_protection(
                df_for_vif,
                vif_candidates,
                protected_features=CORE_STORY_VARIABLES,
                thresh=5.0,
            )
            features_zone = list(zone_vif.keys())
            zone_res = compute_clean_ols(
                df_zone,
                features_zone,
                target_col=TARGET_VAR,
                standardize_x=True,
            )

            if zone_res:
                # Construct the multi-level tuple column key
                col_key = (f"Zone_{zone}", f"{scale}m")
                master_ols_dict[col_key] = zone_res
                master_vif_dict[col_key] = zone_vif
            print(f"  Completed processing for climate zone: {zone} within scale: {scale}m. Results stored.")
        print(f"Completed processing for scale: {scale}m. Global and zone-specific results stored.")

    # ==================================================================
    # THE WRAPPING STEP: One-shot construction of aligned OLS and VIF tables
    # ==================================================================
    final_matrix_df = pd.DataFrame(master_ols_dict).sort_index()
    final_vif_df = pd.DataFrame(master_vif_dict).sort_index()

    ordered_model_columns = sorted(master_ols_dict.keys(), key=_parse_model_sort_key)
    final_matrix_df = final_matrix_df.reindex(columns=ordered_model_columns)
    final_vif_df = final_vif_df.reindex(columns=ordered_model_columns)

    final_matrix_df.columns = pd.MultiIndex.from_tuples(
        final_matrix_df.columns, names=["Subset_Scale", "Grid_Resolution"]
    )
    final_vif_df.columns = pd.MultiIndex.from_tuples(
        final_vif_df.columns, names=["Subset_Scale", "Grid_Resolution"]
    )

    ordered_matrix_index = _ordered_feature_index(final_matrix_df.index.tolist(), CORE_STORY_VARIABLES)
    ordered_vif_index = _ordered_feature_index(final_vif_df.index.tolist(), CORE_STORY_VARIABLES)
    final_matrix_df = final_matrix_df.reindex(ordered_matrix_index)
    final_vif_df = final_vif_df.reindex(ordered_vif_index)

    final_matrix_df.to_csv("../../../data/results/final_mlr_results.csv", index=True)
    final_vif_df.to_csv("../../../data/results/final_mlr_results_vif.csv", index=True)
    print("Master OLS matrix and VIF table generated successfully.")

if __name__ == "__main__":
    main()