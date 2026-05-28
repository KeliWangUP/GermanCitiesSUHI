import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlr_pipeline import train_mlr_pipeline
from pipeline_utils import prepare_scale_dataframe


def _partition_configs(cz_values: list[str]) -> list[dict]:
    configs = [
        {
            'partition_label': 'Global',
            'split_strategy': 'stratified_group',
            'filter_value': None,
        }
    ]
    for value in cz_values:
        configs.append(
            {
                'partition_label': f'Zone_{value}',
                'split_strategy': 'group',
                'filter_value': value,
            }
        )
    return configs


def _append_model_to_matrices(
    coef_matrix: dict,
    vif_matrix: dict,
    model_id: str,
    coef_table_path: str,
    vif_table_path: str,
    metrics: dict,
):
    coef_table = pd.read_csv(coef_table_path)
    for _, row in coef_table.iterrows():
        feature = row['feature']
        # Use the display string so the exported matrix keeps significance stars.
        coef_matrix.setdefault(feature, {})[model_id] = row.get('display', row['coefficient'])

    vif_table = pd.read_csv(vif_table_path)
    for _, row in vif_table.iterrows():
        feature = row['feature']
        vif_matrix.setdefault(feature, {})[model_id] = row.get('vif', np.nan)

    metric_rows = {
        '_RMSE': metrics['rmse'],
        '_MAE': metrics['mae'],
        '_R2': metrics['r2'],
        '_N_FEATURES': metrics['n_features_selected'],
    }
    for metric_name, metric_value in metric_rows.items():
        coef_matrix.setdefault(metric_name, {})[model_id] = metric_value


def _matrix_dict_to_dataframe(matrix_dict: dict, ordered_columns: list[str]) -> pd.DataFrame:
    matrix_df = pd.DataFrame.from_dict(matrix_dict, orient='index')
    matrix_df.index.name = 'feature_or_metric'
    matrix_df = matrix_df.reindex(columns=ordered_columns)
    return matrix_df.reset_index()


def main():
    scales = [100, 250, 500, 750, 1000]
    unified_dir = Path("../../../data/unified_scale_matrices").resolve()
    results_dir = Path("../../../data/results/mlr_results").resolve()

    group_col = 'city_id'
    stratify_col = 'CZ_median'
    size_col = '__sample_size__'
    target_col = 'SUHI'
    vif_threshold = 5.0
    protected_features = ["PLAND_tree_cover", "building_height_mean", "pop_sum", "ED_tree_cover"]

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

    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    coef_matrix = {}
    vif_matrix = {}
    model_columns = []

    for scale in scales:
        print(f"\n[MLR RUN] Processing scale: {scale}m")
        file_path = unified_dir / f"merged_metrics_{scale}m.parquet"
        if not file_path.exists():
            print(f"[MLR RUN] Warning: {file_path} not found.")
            continue

        df_scale, features = prepare_scale_dataframe(
            file_path=file_path,
            name_mapping_dict=name_mapping_dict,
            group_col=group_col,
            target_col=target_col,
            stratify_col=stratify_col,
            size_col=size_col,
            names_to_drop=names_to_drop,
            keep_stratify_as_feature=False,
        )

        cz_values = sorted(df_scale[stratify_col].dropna().astype(str).unique().tolist())
        partition_configs = _partition_configs(cz_values)

        for partition_cfg in partition_configs:
            partition_label = partition_cfg['partition_label']
            filter_value = partition_cfg['filter_value']
            split_strategy = partition_cfg['split_strategy']

            if filter_value is None:
                df_model = df_scale.copy()
            else:
                df_model = df_scale[df_scale[stratify_col].astype(str) == str(filter_value)].copy()

            if df_model.empty:
                print(f"[MLR RUN] Skip {partition_label} at {scale}m: empty subset")
                continue

            model_id = f"{partition_label}_{scale}m"
            print(f"[MLR RUN] Model: {model_id} | rows={len(df_model)}")

            save_dir = results_dir / partition_label / f"scale_{scale}m"
            result = train_mlr_pipeline(
                df=df_model,
                features=features,
                target_col=target_col,
                group_col=group_col,
                stratify_cols=[stratify_col],
                split_strategy=split_strategy,
                size_col=size_col,
                random_state=42,
                n_splits=5,
                test_fold=0,
                save_dir=str(save_dir),
                vif_threshold=vif_threshold,
                protected_features=protected_features,
                vif_sample_size=10000,
                vif_random_state=42,
                verbose=1,
            )

            metrics = result['metrics']
            model_columns.append(model_id)

            summary_rows.append(
                {
                    'model_id': model_id,
                    'partition': partition_label,
                    'scale_m': scale,
                    'subset_rows': len(df_model),
                    'n_features_candidate': metrics['n_features_candidate'],
                    'n_features_selected': metrics['n_features_selected'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'model_path': result['model_path'],
                    'test_output_path': result['test_output_path'],
                    'coef_table_path': result['coef_table_path'],
                    'vif_table_path': result['vif_table_path'],
                    'model_table_path': result['model_table_path'],
                    'metrics_path': result['metrics_path'],
                    'metadata_path': result['metadata_path'],
                }
            )

            _append_model_to_matrices(
                coef_matrix=coef_matrix,
                vif_matrix=vif_matrix,
                model_id=model_id,
                coef_table_path=result['coef_table_path'],
                vif_table_path=result['vif_table_path'],
                metrics=metrics,
            )

            print(
                f"[MLR RUN] Completed {model_id}: RMSE={metrics['rmse']:.4f}, "
                f"MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}"
            )

    if not summary_rows:
        print("[MLR RUN] No models were processed successfully.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['partition', 'scale_m']).reset_index(drop=True)
    summary_csv = results_dir / 'final_mlr_results.csv'
    summary_df.to_csv(summary_csv, index=False)

    for feature_name in vif_matrix.keys():
        coef_matrix.setdefault(feature_name, {})

    unique_model_columns = []
    seen = set()
    for model_id in model_columns:
        if model_id not in seen:
            unique_model_columns.append(model_id)
            seen.add(model_id)

    coef_matrix_df = _matrix_dict_to_dataframe(coef_matrix, unique_model_columns)
    coef_matrix_csv = results_dir / 'final_mlr_coef_matrix.csv'
    coef_matrix_df.to_csv(coef_matrix_csv, index=False)

    vif_matrix_df = _matrix_dict_to_dataframe(vif_matrix, unique_model_columns)
    vif_matrix_csv = results_dir / 'final_mlr_vif_matrix.csv'
    vif_matrix_df.to_csv(vif_matrix_csv, index=False)

    print(f"\n[MLR RUN] Summary saved to: {summary_csv}")
    print(f"[MLR RUN] Coef matrix saved to: {coef_matrix_csv}")
    print(f"[MLR RUN] VIF matrix saved to: {vif_matrix_csv}")


if __name__ == '__main__':
    main()
