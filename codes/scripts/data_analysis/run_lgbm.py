import json
from pathlib import Path

import pandas as pd

from lgbm_pipeline import train_lgbm_pipeline
from pipeline_utils import prepare_scale_dataframe


def _load_partition_vif_features(scale: int, partition: str, mlr_results_dir: Path) -> list[str]:
    """Read the MLR VIF-selected feature list for a given partition and scale."""
    vif_meta_path = mlr_results_dir / partition / f'scale_{scale}m' / 'artifacts' / 'vif_metadata.json'
    if not vif_meta_path.exists():
        raise FileNotFoundError(f"MLR VIF metadata not found for partition={partition}: {vif_meta_path}")

    with open(vif_meta_path, 'r') as fh:
        metadata = json.load(fh)

    selected_features = metadata.get('selected_features', [])
    if not selected_features:
        raise ValueError(f"No selected_features found in {vif_meta_path}")
    return selected_features


def _select_input_features(
    *,
    feature_mode: str,
    scale: int,
    df_scale: pd.DataFrame,
    pre_vif_features: list[str],
    stratify_col: str,
    mlr_results_dir: Path,
    vif_partition: str | None = None,
    include_stratify_feature: bool = True,
) -> list[str]:
    if feature_mode == 'pre_vif_all':
        features = [feature for feature in pre_vif_features if feature in df_scale.columns]
    elif feature_mode == 'vif_selected':
        if not vif_partition:
            raise ValueError('vif_partition is required when feature_mode="vif_selected"')
        vif_features = _load_partition_vif_features(
            scale=scale,
            partition=vif_partition,
            mlr_results_dir=mlr_results_dir,
        )
        features = [feature for feature in vif_features if feature in df_scale.columns]
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    if include_stratify_feature and stratify_col not in features:
        features.append(stratify_col)
    if not include_stratify_feature:
        features = [feature for feature in features if feature != stratify_col]
    return features


def main():
    SCALES = [100, 250, 500, 750, 1000]
    UNIFIED_DIR = Path("../../../data/unified_scale_matrices").resolve()
    FEATURE_MODE = 'pre_vif_all'
    RESULTS_DIR = Path("../../../data/results/lgbm_results_by_climate_zone").resolve()
    MLR_RESULTS_DIR = Path("../../../data/results/mlr_results").resolve()
    MODE_RESULTS_DIR = RESULTS_DIR / FEATURE_MODE

    group_col = 'city_id'
    stratify_col = 'CZ_median'
    size_col = '__sample_size__'
    target_col = 'SUHI'

    name_mapping_dict = {
        '10': 'tree_cover',
        '20': 'shrubland',
        '30': 'grassland',
        '40': 'cropland',
        '50': 'built_up',
        '60': 'bare_land',
        '70': 'snow_and_ice',
        '80': 'water',
        '90': 'wetland',
    }
    names_to_drop = [
        'geometry',
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
        'LST_Rural_mean',
    ]

    MODE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for scale in SCALES:
        print(f"\n[LGBM RUN] Processing scale: {scale}m")
        file_path = UNIFIED_DIR / f"merged_metrics_{scale}m.parquet"
        if not file_path.exists():
            print(f"[LGBM RUN] Warning: {file_path} not found.")
            continue

        df_scale, pre_vif_features = prepare_scale_dataframe(
            file_path=file_path,
            name_mapping_dict=name_mapping_dict,
            group_col=group_col,
            target_col=target_col,
            stratify_col=stratify_col,
            size_col=size_col,
            names_to_drop=names_to_drop,
            keep_stratify_as_feature=False,
        )

        climate_zones = sorted(df_scale[stratify_col].dropna().astype(str).unique().tolist())
        print(f"[LGBM RUN] Climate zones found at {scale}m: {climate_zones}")
        print(f"[LGBM RUN] Feature mode: {FEATURE_MODE}")

        for climate_zone in climate_zones:
            zone_df = df_scale[df_scale[stratify_col].astype(str) == str(climate_zone)].copy()
            n_cities = zone_df[group_col].nunique()
            zone_partition = f"Zone_{climate_zone}"

            features = _select_input_features(
                feature_mode=FEATURE_MODE,
                scale=scale,
                df_scale=zone_df,
                pre_vif_features=pre_vif_features,
                stratify_col=stratify_col,
                mlr_results_dir=MLR_RESULTS_DIR,
                vif_partition=zone_partition,
                include_stratify_feature=False,
            )

            print(
                f"[LGBM RUN] Zone={climate_zone} features selected "
                f"(from {zone_partition}, without {stratify_col}): {len(features)}"
            )
            print("[LGBM RUN] Input feature names:")
            for feature_name in features:
                print(f"  - {feature_name}")

            if n_cities < 5:
                print(
                    f"[LGBM RUN] Skip scale={scale}m, zone={climate_zone}: "
                    f"need at least 5 unique {group_col} for n_splits=5, got {n_cities}."
                )
                continue

            print(
                f"[LGBM RUN] Training scale={scale}m, zone={climate_zone}, "
                f"rows={len(zone_df)}, {group_col}s={n_cities}"
            )
            save_dir = MODE_RESULTS_DIR / f"CZ_{climate_zone}" / f"scale_{scale}m"

            result = train_lgbm_pipeline(
                df=zone_df,
                features=features,
                target_col=target_col,
                group_col=group_col,
                stratify_cols=[],
                categorical_features=[],
                size_col=size_col,
                random_state=42,
                n_splits=5,
                test_fold=0,
                use_optuna=True,
                n_trials=50,
                save_dir=str(save_dir),
                model_n_jobs=4,
                cv_jobs=8,
                search_jobs=1,
                verbose=1,
            )

            metrics = result['metrics']
            summary_rows.append(
                {
                    'climate_zone': climate_zone,
                    'scale_m': scale,
                    'feature_mode': FEATURE_MODE,
                    'n_features': len(features),
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'cv_best_score': metrics['cv_best_score'],
                    'final_model_path': result['final_model_path'],
                    'X_test_path': result['X_test_path'],
                    'y_test_path': result['y_test_path'],
                    'y_pred_path': result['y_pred_path'],
                    'test_output_path': result['test_output_path'],
                    'params_path': result['params_path'],
                    'metrics_path': result['metrics_path'],
                    'metadata_path': result['metadata_path'],
                    'best_params_json': json.dumps(result['best_params'], ensure_ascii=False),
                }
            )

            print(
                f"[LGBM RUN] Completed scale={scale}m, zone={climate_zone}: "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}"
            )
            print(f"[LGBM RUN] Test predictions: {result['test_output_path']}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(['climate_zone', 'scale_m']).reset_index(drop=True)
        summary_csv = MODE_RESULTS_DIR / f'final_lgbm_results_by_climate_zone_{FEATURE_MODE}.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[LGBM RUN] Summary saved to: {summary_csv}")
    else:
        print("[LGBM RUN] No scales were processed successfully.")


if __name__ == '__main__':
    main()
