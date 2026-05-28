import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_utils import (
    build_holdout_predictions_frame,
    ensure_output_dirs,
    split_outer_train_test_by_group,
    split_outer_train_test_by_stratified_group,
)
from vif import filter_features_by_vif_with_audit


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _standardize_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    feature_std = X_train.std(ddof=0)
    zero_var_features = feature_std[feature_std == 0].index.tolist()
    if zero_var_features and verbose:
        print(f"[MLR] Dropping zero-variance features before standardization: {zero_var_features}")

    kept_features = [feature for feature in X_train.columns if feature not in zero_var_features]
    X_train_kept = X_train[kept_features].copy()
    X_test_kept = X_test[kept_features].copy()

    train_mean = X_train_kept.mean()
    train_std = X_train_kept.std(ddof=0).replace(0, np.nan)
    X_train_std = (X_train_kept - train_mean) / train_std
    X_test_std = (X_test_kept - train_mean) / train_std

    X_train_std = X_train_std.replace([np.inf, -np.inf], np.nan).dropna()
    X_test_std = X_test_std.replace([np.inf, -np.inf], np.nan).dropna()

    standardization_info = {
        'kept_features': kept_features,
        'zero_var_features': zero_var_features,
        'train_mean': train_mean.to_dict(),
        'train_std': train_std.fillna(0).to_dict(),
    }
    return X_train_std, X_test_std, standardization_info


def _fit_ols_model(X_train: pd.DataFrame, y_train: pd.Series):
    X_train_const = sm.add_constant(X_train, has_constant='add')
    model = sm.OLS(y_train, X_train_const).fit()
    return model


def _evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred = model.predict(X_test_const)
    return {
        'y_pred': y_pred,
        'rmse': _rmse(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }


def _coef_table_from_model(model, feature_order: list | None = None) -> pd.DataFrame:
    coeffs = model.params
    p_values = model.pvalues
    table = pd.DataFrame(
        {
            'feature': coeffs.index,
            'coefficient': coeffs.values,
            'p_value': p_values.values,
        }
    )
    table['stars'] = table['p_value'].apply(get_significance_stars)
    table['display'] = table.apply(
        lambda row: f"{row['coefficient']:.3f}{row['stars']}" if row['feature'] != 'const' else f"{row['coefficient']:.3f}",
        axis=1,
    )
    if feature_order:
        order_map = {feature: index for index, feature in enumerate(feature_order)}
        table['sort_rank'] = table['feature'].map(lambda feature: 0 if feature == 'const' else 1)
        table['feature_rank'] = table['feature'].map(lambda feature: order_map.get(feature, len(order_map) + 1))
        table = table.sort_values(['sort_rank', 'feature_rank', 'feature']).drop(columns=['sort_rank', 'feature_rank'])
    return table.reset_index(drop=True)


def _maybe_sample_for_vif(
    X: pd.DataFrame,
    max_rows: int = 10000,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    if len(X) > max_rows:
        if verbose:
            print(f"[MLR] Sampling {max_rows} rows from {len(X)} rows for VIF computation")
        return X.sample(n=max_rows, random_state=random_state)
    return X


def _save_artifacts(
    model,
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    X_test_raw: pd.DataFrame,
    test_output_df: pd.DataFrame,
    y_test: pd.Series,
    coefficient_table: pd.DataFrame,
    vif_table: pd.DataFrame,
    model_table: pd.DataFrame,
    dirs: Dict[str, Path],
):
    model_path = dirs['models_dir'] / 'final_model.pkl'
    model.save(model_path)

    raw_test_path = dirs['folds_dir'] / 'X_test_raw.parquet'
    y_test_path = dirs['folds_dir'] / 'y_test.parquet'
    test_output_path = dirs['folds_dir'] / 'test_predictions.parquet'

    X_test_raw.reset_index(drop=True).to_parquet(raw_test_path)
    pd.DataFrame(y_test.reset_index(drop=True), columns=[y_test.name or 'y_test']).to_parquet(y_test_path)
    test_output_df.reset_index(drop=True).to_parquet(test_output_path)

    coef_table_path = dirs['artifacts_dir'] / 'coef_table.csv'
    coefficient_table.to_csv(coef_table_path, index=False)

    vif_table_path = dirs['artifacts_dir'] / 'vif_table.csv'
    vif_table.to_csv(vif_table_path, index=False)

    model_table_path = dirs['artifacts_dir'] / 'model_table.csv'
    model_table.to_csv(model_table_path, index=False)

    summary_path = dirs['artifacts_dir'] / 'model_summary.txt'
    with open(summary_path, 'w') as fh:
        fh.write(model.summary().as_text())

    metrics_path = dirs['metrics_dir'] / 'test_metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)

    metadata_path = dirs['artifacts_dir'] / 'metadata.json'
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    vif_meta_path = dirs['artifacts_dir'] / 'vif_metadata.json'
    with open(vif_meta_path, 'w') as fh:
        json.dump(
            {
                'vif_threshold': metadata.get('vif_threshold'),
                'vif_sample_size': metadata.get('vif_sample_size'),
                'vif_random_state': metadata.get('vif_random_state'),
                'protected_features': metadata.get('protected_features', []),
                'selected_features': metadata.get('selected_features', []),
            },
            fh,
            indent=2,
        )

    return {
        'model_path': str(model_path),
        'X_test_raw_path': str(raw_test_path),
        'y_test_path': str(y_test_path),
        'test_output_path': str(test_output_path),
        'coef_table_path': str(coef_table_path),
        'vif_table_path': str(vif_table_path),
        'model_table_path': str(model_table_path),
        'summary_path': str(summary_path),
        'metrics_path': str(metrics_path),
        'metadata_path': str(metadata_path),
        'vif_metadata_path': str(vif_meta_path),
    }


def train_mlr_pipeline(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    group_col: str = 'city_id',
    stratify_cols: list = None,
    split_strategy: str = 'stratified_group',
    size_col: str = '__sample_size__',
    random_state: int = 42,
    n_splits: int = 5,
    test_fold: int = 0,
    save_dir: str = '../../../data/results/mlr_pipeline',
    vif_threshold: float = 5.0,
    protected_features: list | None = None,
    vif_sample_size: int = 10000,
    vif_random_state: int = 42,
    verbose: int = 1,
) -> Dict[str, Any]:
    if stratify_cols is None:
        stratify_cols = ['CZ_median']

    save_dir = Path(save_dir)
    dirs = ensure_output_dirs(save_dir)

    required_cols = set(features) | {target_col, group_col, size_col} | set(stratify_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f'Missing required columns for pipeline: {missing_cols}')

    if verbose:
        print('[MLR] Starting pipeline')
        print(f"[MLR] Features selected: {len(features)}")
        print(f"[MLR] Group column: {group_col} | Stratify columns: {stratify_cols} | Size column: {size_col}")
        print(f"[MLR] Split strategy: {split_strategy}")
        print(f"[MLR] VIF threshold: {vif_threshold}")
        print(f"[MLR] Protected features: {protected_features or []}")

    if split_strategy == 'stratified_group':
        train_df, test_df = split_outer_train_test_by_stratified_group(
            df=df,
            target_col=target_col,
            group_col=group_col,
            stratify_cols=stratify_cols,
            size_col=size_col,
            n_splits=n_splits,
            random_state=random_state,
            test_fold=test_fold,
            verbose=verbose,
        )
    elif split_strategy == 'group':
        train_df, test_df = split_outer_train_test_by_group(
            df=df,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            test_fold=test_fold,
            verbose=verbose,
        )
    else:
        raise ValueError("split_strategy must be one of {'stratified_group', 'group'}")

    X_train_raw = train_df[features].copy()
    y_train = train_df[target_col].copy()
    X_test_raw = test_df[features].copy()
    y_test = test_df[target_col].copy()

    if verbose:
        print(f"[MLR] Raw matrices prepared: train={X_train_raw.shape}, test={X_test_raw.shape}")
        print('[MLR] Standardizing training features with train-set mean/std')

    X_train_std, X_test_std, standardization_info = _standardize_train_test_features(
        X_train=X_train_raw,
        X_test=X_test_raw,
        verbose=verbose,
    )

    y_train_aligned = y_train.loc[X_train_std.index]
    y_test_aligned = y_test.loc[X_test_std.index]

    if verbose:
        print(f"[MLR] Standardized matrices prepared: train={X_train_std.shape}, test={X_test_std.shape}")
        print('[MLR] Filtering features by VIF before OLS fit')

    X_vif_input = _maybe_sample_for_vif(
        X_train_std,
        max_rows=vif_sample_size,
        random_state=vif_random_state,
        verbose=verbose,
    )
    selected_features, vif_table = filter_features_by_vif_with_audit(
        X_vif_input,
        features=list(X_vif_input.columns),
        thresh=vif_threshold,
        protected_features=protected_features,
    )

    if not selected_features:
        raise ValueError('No features remain after VIF filtering.')

    selected_features = [feature for feature in X_train_std.columns if feature in selected_features]
    X_train_selected = X_train_std[selected_features].copy()
    X_test_selected = X_test_std[selected_features].copy()

    if verbose:
        print(f"[MLR] Features retained after VIF: {len(selected_features)}")
        print(f"[MLR] Selected features: {selected_features}")
        print('[MLR] Fitting OLS model on VIF-filtered standardized training data')

    model = _fit_ols_model(X_train_selected, y_train_aligned)
    holdout = _evaluate_model(model, X_test_selected, y_test_aligned)
    test_output_df = build_holdout_predictions_frame(test_df.loc[X_test_selected.index], y_test_aligned, holdout['y_pred'])

    metrics = {
        'rmse': float(holdout['rmse']),
        'mae': float(holdout['mae']),
        'r2': float(holdout['r2']),
        'n_train': int(len(X_train_std)),
        'n_test': int(len(X_test_std)),
        'n_features_used': int(X_train_selected.shape[1]),
        'vif_threshold': float(vif_threshold),
        'n_features_candidate': int(len(features)),
        'n_features_selected': int(len(selected_features)),
    }

    coefficient_table = _coef_table_from_model(model, feature_order=selected_features)
    coefficient_table = coefficient_table[['feature', 'coefficient', 'p_value', 'stars', 'display']]

    vif_table = vif_table.copy()
    vif_table['feature'] = pd.Categorical(vif_table['feature'], categories=features, ordered=True)
    vif_table = vif_table.sort_values('feature').reset_index(drop=True)
    vif_table['feature'] = vif_table['feature'].astype(str)

    model_table = pd.DataFrame({'feature': ['const'] + features})
    model_table = model_table.merge(vif_table, on='feature', how='left')
    model_table = model_table.merge(coefficient_table, on='feature', how='left')
    model_table['selected'] = model_table['selected'].fillna(model_table['feature'].eq('const'))
    model_table['protected'] = model_table['protected'].fillna(False)
    model_table['drop_iteration'] = model_table['drop_iteration'].where(pd.notna(model_table['drop_iteration']), None)
    model_table['model_rmse'] = metrics['rmse']
    model_table['model_mae'] = metrics['mae']
    model_table['model_r2'] = metrics['r2']
    model_table['model_adj_r2'] = float(model.rsquared_adj)
    model_table['n_train'] = metrics['n_train']
    model_table['n_test'] = metrics['n_test']
    model_table['n_features_selected'] = metrics['n_features_selected']
    model_table['n_features_candidate'] = metrics['n_features_candidate']

    metadata = {
        'random_state': int(random_state),
        'n_splits': int(n_splits),
        'test_fold': int(test_fold),
        'split_strategy': split_strategy,
        'group_col': group_col,
        'stratify_cols': stratify_cols,
        'size_col': size_col,
        'target_col': target_col,
        'features': features,
        'selected_features': selected_features,
        'protected_features': protected_features or [],
        'vif_threshold': float(vif_threshold),
        'standardization_info': standardization_info,
        'vif_sample_size': int(vif_sample_size),
        'vif_random_state': int(vif_random_state),
    }

    paths = _save_artifacts(
        model=model,
        metrics=metrics,
        metadata=metadata,
        X_test_raw=X_test_raw,
        test_output_df=test_output_df,
        y_test=y_test_aligned,
        coefficient_table=coefficient_table,
        vif_table=vif_table,
        model_table=model_table,
        dirs=dirs,
    )

    if verbose:
        print(f"[MLR] Holdout evaluation completed: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        print(f"[MLR] Artifacts saved to: {save_dir}")
        print(f"[MLR] Model: {paths['model_path']}")
        print(f"[MLR] Test raw data: {paths['X_test_raw_path']}")
        print(f"[MLR] Test predictions: {paths['test_output_path']}")
        print(f"[MLR] Coef table: {paths['coef_table_path']}")
        print(f"[MLR] VIF table: {paths['vif_table_path']}")
        print(f"[MLR] Model table: {paths['model_table_path']}")
        print(f"[MLR] VIF metadata: {paths['vif_metadata_path']}")

    return {
        **paths,
        'metrics': metrics,
        'best_params': None,
        'holdout_predictions': holdout['y_pred'],
    }


def get_significance_stars(p_value):
    """Returns classic academic significance stars based on p-value thresholds."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def compute_clean_ols(df_subset, features, target_col="target_suhi", standardize_x=True):
    """
    Fits an OLS model and returns a clean dictionary of results.
    Directly extracts coefficients and p-values from native model attributes 
    to bypass unstable summary table text parsing and prevent KeyErrors.
    """
    X = df_subset[features].dropna()
    y = df_subset[target_col].loc[X.index]

    if X.empty:
        return None

    if standardize_x:
        # Guard against divide-by-zero in z-score normalization.
        feature_std = X.std(ddof=0)
        zero_var_features = feature_std[feature_std == 0].index.tolist()
        if zero_var_features:
            print(f"    [OLS Notice] Dropping zero-variance features before z-score: {zero_var_features}")
            X = X.drop(columns=zero_var_features)

        if X.empty:
            print("    [OLS Bypassed] No valid features remain after zero-variance filtering.")
            return None

        X = (X - X.mean()) / X.std(ddof=0)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

    n_samples = X.shape[0]
    n_features = X.shape[1]

    if X.empty or len(X) <= X.shape[1] + 1:
        return None

    if n_samples <= (n_features + 5):
        print(
            f"    [OLS Bypassed] Insufficient samples ({n_samples}) for {n_features} features. Risk of Overfitting/Collapse."
        )
        return None

    # Fit standard OLS with Intercept
    X_const = sm.add_constant(X, has_constant='add')
    
    try:
        model = sm.OLS(y, X_const).fit()
    except Exception as e:
        print(f"    [OLS Error] Linear regression failed mathematically: {e}")
        return None

    # ------------------------------------------------------------------
    # SAFEST LAYER: Extract from native attributes, completely bypassing summaries
    # ------------------------------------------------------------------
    # model.params and model.pvalues are pandas Series indexed by feature names
    coefficients = model.params
    p_values = model.pvalues

    clean_results = {}
    for feature_name in coefficients.index:
        coef = coefficients[feature_name]
        p_val = p_values[feature_name]
        
        # Handle cases where p-value calculation returns NaN due to singular matrix inputs
        stars = get_significance_stars(p_val)

        # Format string based on feature type
        if feature_name != "const":
            display_value = f"{coef:.3f}{stars}"
        else:
            display_value = f"{coef:.3f}"
            
        clean_results[feature_name] = display_value

    # Append global fitness score safely
    clean_results["_Model_R2"] = f"{model.rsquared:.3f}"

    return clean_results