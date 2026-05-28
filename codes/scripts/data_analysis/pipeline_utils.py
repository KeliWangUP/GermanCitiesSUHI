from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def rename_cls_columns(df: pd.DataFrame, name_mapping_dict: dict) -> pd.DataFrame:
    """Normalize class-suffix column names once so all pipelines see the same inputs."""
    new_column_names = {}
    for old_name in df.columns:
        if '_cls_' in old_name:
            suffix = old_name.split('_cls_')[-1]
            new_suffix = name_mapping_dict.get(suffix, suffix)
            new_column = old_name.replace(f'_cls_{suffix}', f'_{new_suffix}')
            new_column_names[old_name] = new_column
    return df.rename(columns=new_column_names)


def build_feature_list(df: pd.DataFrame, exclude_cols: list) -> list:
    """Keep the original column order; only remove explicit non-features."""
    return [col for col in df.columns if col not in exclude_cols]


def prepare_scale_dataframe(
    file_path: Path,
    name_mapping_dict: dict,
    group_col: str,
    target_col: str,
    stratify_col: str,
    size_col: str,
    names_to_drop: list,
    keep_stratify_as_feature: bool = False,
) -> tuple[pd.DataFrame, list]:
    """Load one scale and return a cleaned dataframe plus its model feature list."""
    df_scale = pd.read_parquet(file_path)

    if stratify_col not in df_scale.columns:
        raise KeyError(f"Missing required stratify column '{stratify_col}' in {file_path.name}")

    df_scale = rename_cls_columns(df_scale, name_mapping_dict)
    # Clean CZ_median (or the chosen stratify column) once here so both MLR and LGBM use the same group labels.
    df_scale[stratify_col] = pd.to_numeric(df_scale[stratify_col], errors='coerce')
    df_scale = df_scale.dropna(subset=[group_col, target_col, stratify_col]).copy()
    df_scale[stratify_col] = df_scale[stratify_col].astype(int).astype(str)
    df_scale[size_col] = 1

    # MLR keeps the stratify variable as a feature in the global model input; LGBM does not.
    exclude_cols = set(names_to_drop) | {group_col, target_col, size_col}
    if not keep_stratify_as_feature:
        exclude_cols.add(stratify_col)

    features = build_feature_list(df_scale, list(exclude_cols))
    return df_scale, features


def ensure_output_dirs(save_dir: Path) -> Dict[str, Path]:
    models_dir = save_dir / 'models'
    folds_dir = save_dir / 'folds'
    artifacts_dir = save_dir / 'artifacts'
    metrics_dir = save_dir / 'metrics'

    for directory in (models_dir, folds_dir, artifacts_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        'models_dir': models_dir,
        'folds_dir': folds_dir,
        'artifacts_dir': artifacts_dir,
        'metrics_dir': metrics_dir,
    }


def geometry_value_to_text(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    geometry_text = getattr(value, 'wkt', None)
    if geometry_text is not None:
        return geometry_text
    return str(value)


def make_strata_label(
    df: pd.DataFrame,
    stratify_cols: list,
    size_col: str,
    group_col: str,
    n_bins: int = 3,
) -> pd.Series:
    group_meta = (
        df.groupby(group_col)
        .agg({**{col: 'first' for col in stratify_cols}, size_col: 'count'})
        .rename(columns={size_col: 'pixel_count'})
    )

    try:
        labels = ['S', 'M', 'L'][:n_bins]
        group_meta['size_bin'] = pd.qcut(group_meta['pixel_count'], q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        group_meta['size_bin'] = 'M'

    group_meta['strata_label'] = (
        group_meta[stratify_cols].astype(str).agg('_'.join, axis=1) + '_' + group_meta['size_bin'].astype(str)
    )
    return df[group_col].map(group_meta['strata_label'])


def split_outer_train_test_by_stratified_group(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stratify_cols: list,
    size_col: str,
    n_splits: int,
    random_state: int,
    test_fold: int = 0,
    verbose: bool = True,
):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise ImportError('StratifiedGroupKFold is required for the outer split') from exc

    df_split = df.copy()
    df_split['strata_label'] = make_strata_label(
        df_split,
        stratify_cols=stratify_cols,
        size_col=size_col,
        group_col=group_col,
        n_bins=3,
    )
    df_split = df_split.dropna(subset=['strata_label', group_col, target_col])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df_split, df_split['strata_label'], groups=df_split[group_col]))
    if test_fold < 0 or test_fold >= len(splits):
        raise ValueError(f'test_fold must be in [0, {len(splits) - 1}], got {test_fold}')

    train_idx, test_idx = splits[test_fold]
    train_df = df_split.iloc[train_idx].copy()
    test_df = df_split.iloc[test_idx].copy()

    if verbose:
        print(f"[Outer Split] train rows={len(train_df)}, test rows={len(test_df)}")
        print(f"[Outer Split] train {group_col}s={train_df[group_col].nunique()}, test {group_col}s={test_df[group_col].nunique()}")
        print(f"[Outer Split] test strata distribution:\n{test_df['strata_label'].value_counts().sort_index()}")

    return train_df, test_df


def split_outer_train_test_by_group(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_splits: int,
    test_fold: int = 0,
    verbose: bool = True,
):
    from sklearn.model_selection import GroupKFold

    df_split = df.dropna(subset=[group_col, target_col]).copy()
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df_split, groups=df_split[group_col]))
    if test_fold < 0 or test_fold >= len(splits):
        raise ValueError(f'test_fold must be in [0, {len(splits) - 1}], got {test_fold}')

    train_idx, test_idx = splits[test_fold]
    train_df = df_split.iloc[train_idx].copy()
    test_df = df_split.iloc[test_idx].copy()

    if verbose:
        print(f"[Group Split] train rows={len(train_df)}, test rows={len(test_df)}")
        print(f"[Group Split] train {group_col}s={train_df[group_col].nunique()}, test {group_col}s={test_df[group_col].nunique()}")

    return train_df, test_df


def build_holdout_predictions_frame(
    base_df: pd.DataFrame,
    y_true,
    y_pred,
    geometry_col: str = 'geometry',
) -> pd.DataFrame:
    output_df = base_df.copy()
    output_df['y_true'] = np.asarray(y_true)
    output_df['y_pred'] = np.asarray(y_pred)
    output_df['residual'] = output_df['y_true'] - output_df['y_pred']
    if geometry_col in output_df.columns:
        output_df[geometry_col] = output_df[geometry_col].map(geometry_value_to_text)
    return output_df
