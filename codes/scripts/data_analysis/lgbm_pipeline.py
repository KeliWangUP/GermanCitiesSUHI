import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_HAS_OPTUNA = False

from lightgbm import LGBMRegressor

from pipeline_utils import (
    build_holdout_predictions_frame,
    ensure_output_dirs,
    split_outer_train_test_by_stratified_group,
)


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _default_param_space():
    # With ~20 features and sample sizes ranging from 10k to 600k, keep trees compact
    # and let early stopping find the effective boosting rounds.
    return {
        'num_leaves': [31, 63, 127, 255],
        'max_depth': [-1, 6, 8, 10],
        'learning_rate': [0.01, 0.02, 0.05],
        'n_estimators': [100, 300, 500, 1000],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'reg_alpha': [0.1, 0.5, 2.0],
        'reg_lambda': [0.1, 0.5, 2.0],
        'min_child_samples': [20, 50, 100],
    }


def _resolve_parallelism(
    search_jobs: int,
    cv_jobs: int,
    model_n_jobs: int,
    strategy: str = 'auto',
) -> Dict[str, int]:
    search_jobs = max(1, int(search_jobs))
    cv_jobs = max(1, int(cv_jobs))
    model_n_jobs = max(1, int(model_n_jobs))

    if strategy == 'cv':
        return {
            'search_jobs': 1,
            'cv_jobs': cv_jobs,
            'model_n_jobs': model_n_jobs,
        }
    if strategy == 'search':
        return {
            'search_jobs': search_jobs,
            'cv_jobs': 1,
            'model_n_jobs': model_n_jobs,
        }

    if search_jobs > 1:
        return {
            'search_jobs': search_jobs,
            'cv_jobs': 1,
            'model_n_jobs': model_n_jobs,
        }
    if cv_jobs > 1:
        return {
            'search_jobs': 1,
            'cv_jobs': cv_jobs,
            'model_n_jobs': 1,
        }
    return {
        'search_jobs': 1,
        'cv_jobs': 1,
        'model_n_jobs': model_n_jobs,
    }


def _build_lgbm_model(params: Dict[str, Any], random_state: int, model_n_jobs: int) -> LGBMRegressor:
    model_params = dict(params)
    # RandomizedSearchCV/Optuna best_params may already carry random_state.
    model_params.pop('random_state', None)
    model_params['random_state'] = random_state
    model_params['n_jobs'] = model_n_jobs
    return LGBMRegressor(**model_params)


def _fit_lgbm_with_early_stopping(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    random_state: int,
    model_n_jobs: int,
    categorical_features: list | None,
    early_stopping_rounds: int,
):
    model = _build_lgbm_model(params, random_state=random_state, model_n_jobs=model_n_jobs)
    fit_kwargs: Dict[str, Any] = {
        'categorical_feature': categorical_features,
        'eval_metric': 'rmse',
    }
    if X_valid is not None and y_valid is not None and early_stopping_rounds > 0:
        fit_kwargs['eval_set'] = [(X_valid, y_valid)]
        fit_kwargs['callbacks'] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _predict_with_best_iteration(model, X: pd.DataFrame):
    best_iteration = getattr(model, 'best_iteration_', None)
    if best_iteration is None or best_iteration <= 0:
        return model.predict(X)
    return model.predict(X, num_iteration=best_iteration)


def _get_cv(n_splits: int, random_state: int) -> GroupKFold:
    return GroupKFold(n_splits=n_splits)


def _evaluate_fold(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_idx,
    val_idx,
    random_state: int,
    model_n_jobs: int,
    categorical_features: list | None,
    early_stopping_rounds: int,
):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = _fit_lgbm_with_early_stopping(
        params=params,
        X_train=X_tr,
        y_train=y_tr,
        X_valid=X_val,
        y_valid=y_val,
        random_state=random_state,
        model_n_jobs=model_n_jobs,
        categorical_features=categorical_features,
        early_stopping_rounds=early_stopping_rounds,
    )
    preds = _predict_with_best_iteration(model, X_val)
    return {
        'rmse': _rmse(y_val, preds),
        'mae': mean_absolute_error(y_val, preds),
        'r2': r2_score(y_val, preds),
    }


def _cv_score_for_params(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv,
    random_state: int,
    cv_jobs: int = 8,
    model_n_jobs: int = -1,
    categorical_features: list | None = None,
    early_stopping_rounds: int = 50,
):
    fold_results = Parallel(n_jobs=cv_jobs, prefer='processes')(
        delayed(_evaluate_fold)(
            params,
            X_train,
            y_train,
            train_idx,
            val_idx,
            random_state,
            model_n_jobs,
            categorical_features,
            early_stopping_rounds,
        )
        for train_idx, val_idx in cv.split(X_train, y_train, groups=groups_train)
    )
    fold_rmse = [result['rmse'] for result in fold_results]
    return float(np.mean(fold_rmse)), fold_results


def _sample_optuna_params(trial, param_space: Dict[str, list], random_state: int):
    sampled = {
        key: trial.suggest_categorical(key, values)
        for key, values in param_space.items()
    }
    sampled['random_state'] = random_state
    return sampled


def _sample_random_params(param_space: Dict[str, list], rng: np.random.RandomState) -> Dict[str, Any]:
    sampled = {}
    for key, values in param_space.items():
        sampled[key] = values[rng.randint(0, len(values))]
    return sampled


def _tune_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv,
    random_state: int,
    param_space: Dict[str, list],
    n_trials: int,
    artifacts_dir: Path,
    verbose: int,
    search_jobs: int,
    cv_jobs: int,
    model_n_jobs: int,
    categorical_features: list | None,
    early_stopping_rounds: int,
):
    import importlib
    import importlib.util

    if importlib.util.find_spec('optuna') is None:
        raise ImportError('optuna not found')

    optuna = importlib.import_module('optuna')
    if verbose:
        print('Using Optuna for hyperparameter search')

    # Optuna's n_jobs relies on multi-threading; for this workload, fold-level process
    # parallelism is usually more stable and reaches higher CPU utilization.
    parallelism = _resolve_parallelism(
        search_jobs=search_jobs,
        cv_jobs=cv_jobs,
        model_n_jobs=model_n_jobs,
        strategy='cv',
    )
    if verbose:
        print(
            '[LGBM] Optuna parallelism: '
            f"search_jobs={parallelism['search_jobs']}, cv_jobs={parallelism['cv_jobs']}, "
            f"model_n_jobs={parallelism['model_n_jobs']}"
        )

    def _objective(trial):
        params = _sample_optuna_params(trial, param_space, random_state)
        mean_rmse, _ = _cv_score_for_params(
            params,
            X_train,
            y_train,
            groups_train,
            cv,
            random_state,
            cv_jobs=parallelism['cv_jobs'],
            model_n_jobs=parallelism['model_n_jobs'],
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
        )
        return mean_rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(_objective, n_trials=n_trials, n_jobs=parallelism['search_jobs'])

    try:
        joblib.dump(study, artifacts_dir / 'optuna_study.pkl')
    except Exception:
        pass

    return study.best_params, float(study.best_value)


def _tune_with_random_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv,
    random_state: int,
    param_space: Dict[str, list],
    n_trials: int,
    verbose: int,
    search_jobs: int,
    model_n_jobs: int,
    categorical_features: list | None,
    early_stopping_rounds: int,
    cv_jobs: int,
):
    rng = np.random.RandomState(random_state)
    sampled_params = [_sample_random_params(param_space, rng) for _ in range(n_trials)]
    parallelism = _resolve_parallelism(
        search_jobs=search_jobs,
        cv_jobs=cv_jobs,
        model_n_jobs=model_n_jobs,
        strategy='search',
    )

    if verbose:
        print(
            '[LGBM] Random search parallelism: '
            f"search_jobs={parallelism['search_jobs']}, cv_jobs={parallelism['cv_jobs']}, "
            f"model_n_jobs={parallelism['model_n_jobs']}"
        )

    def _score_candidate(trial_index: int, params: Dict[str, Any]):
        mean_rmse, _ = _cv_score_for_params(
            params,
            X_train,
            y_train,
            groups_train,
            cv,
            random_state,
            cv_jobs=parallelism['cv_jobs'],
            model_n_jobs=parallelism['model_n_jobs'],
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
        )
        return trial_index, params, mean_rmse

    trial_results = Parallel(n_jobs=parallelism['search_jobs'], prefer='processes')(
        delayed(_score_candidate)(trial_index, params)
        for trial_index, params in enumerate(sampled_params, start=1)
    )

    best_params = None
    best_score = float('inf')
    for trial_index, params, mean_rmse in trial_results:
        if verbose:
            print(f"[LGBM] Random search trial {trial_index}/{n_trials}: RMSE={mean_rmse:.4f}")
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    if best_params is None:
        raise ValueError('Random search failed to produce any valid parameter set.')

    return best_params, float(best_score)


def _train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    best_params: Dict[str, Any],
    random_state: int,
    model_n_jobs: int,
    categorical_features: list | None,
    early_stopping_rounds: int,
    validation_splits: int,
):
    if early_stopping_rounds > 0 and groups_train.nunique() >= 2:
        n_val_splits = min(validation_splits, groups_train.nunique())
        n_val_splits = max(2, n_val_splits)
        gkf = GroupKFold(n_splits=n_val_splits)
        train_idx, val_idx = next(gkf.split(X_train, y_train, groups=groups_train))
        tuned_model = _fit_lgbm_with_early_stopping(
            params=best_params,
            X_train=X_train.iloc[train_idx],
            y_train=y_train.iloc[train_idx],
            X_valid=X_train.iloc[val_idx],
            y_valid=y_train.iloc[val_idx],
            random_state=random_state,
            model_n_jobs=model_n_jobs,
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
        )

        best_iteration = getattr(tuned_model, 'best_iteration_', None)
        refit_params = dict(best_params)
        if best_iteration is not None and best_iteration > 0:
            refit_params['n_estimators'] = int(best_iteration)

        final_model = _build_lgbm_model(refit_params, random_state=random_state, model_n_jobs=model_n_jobs)
        final_model.fit(X_train, y_train, categorical_feature=categorical_features)
    else:
        final_model = _build_lgbm_model(best_params, random_state=random_state, model_n_jobs=model_n_jobs)
        final_model.fit(X_train, y_train, categorical_feature=categorical_features)
    return final_model


def _evaluate_holdout(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = _predict_with_best_iteration(model, X_test)
    return {
        'y_pred': y_pred,
        'rmse': _rmse(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }


def _save_artifacts(
    final_model,
    best_params: Dict[str, Any],
    metrics: Dict[str, float],
    metadata: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_output_df: pd.DataFrame,
    y_pred: np.ndarray,
    dirs: Dict[str, Path],
):
    final_model_path = dirs['models_dir'] / 'final_model.pkl'
    joblib.dump(final_model, final_model_path)

    X_test_path = dirs['folds_dir'] / 'X_test.parquet'
    y_test_path = dirs['folds_dir'] / 'y_test.parquet'
    y_pred_path = dirs['folds_dir'] / 'y_pred.parquet'
    test_output_path = dirs['folds_dir'] / 'test_predictions.parquet'
    pd.DataFrame(X_test.reset_index(drop=True)).to_parquet(X_test_path)
    pd.DataFrame(y_test.reset_index(drop=True), columns=[y_test.name or 'y_test']).to_parquet(y_test_path)
    pd.Series(y_pred).to_frame('y_pred').to_parquet(y_pred_path)
    test_output_df.reset_index(drop=True).to_parquet(test_output_path)

    metrics_path = dirs['metrics_dir'] / 'test_metrics.json'
    with open(metrics_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)

    params_path = dirs['artifacts_dir'] / 'best_params.json'
    with open(params_path, 'w') as fh:
        json.dump(best_params, fh, indent=2)

    metadata_path = dirs['artifacts_dir'] / 'metadata.json'
    with open(metadata_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    return {
        'final_model_path': str(final_model_path),
        'X_test_path': str(X_test_path),
        'y_test_path': str(y_test_path),
        'y_pred_path': str(y_pred_path),
        'params_path': str(params_path),
        'metrics_path': str(metrics_path),
        'metadata_path': str(metadata_path),
        'test_output_path': str(test_output_path),
    }


def train_lgbm_pipeline(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    group_col: str = 'city_id',
    stratify_cols: list = None,
    size_col: str = 'suhi_intensity',
    random_state: int = 42,
    n_splits: int = 5,
    test_fold: int = 0,
    use_optuna: bool = True,
    n_trials: int = 50,
    param_space: Dict[str, Any] = None,
    categorical_features: list | None = None,
    early_stopping_rounds: int = 50,
    validation_splits: int = 5,
    save_dir: str = "../../../data/results/lgbm_pipeline",
    model_n_jobs: int = 4,
    cv_jobs: int = 8,
    search_jobs: int = 8,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Simplified LightGBM pipeline:
    - global train/test split
    - 5-fold CV on training for hyperparameter tuning (Optuna preferred)
    - train final model on full training set with best params
    - test on held-out test set

    Returns dict with paths and summary metrics.
    """
    save_dir = Path(save_dir)
    dirs = ensure_output_dirs(save_dir)

    if verbose:
        print("[LGBM] Starting pipeline")

    if param_space is None:
        param_space = _default_param_space()

    if stratify_cols is None:
        stratify_cols = ['CZ_median']
    if categorical_features is None:
        categorical_features = []

    required_cols = set(features) | {target_col, group_col, size_col} | set(stratify_cols)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f'Missing required columns for pipeline: {missing_cols}')
    missing_cat_cols = [col for col in categorical_features if col not in features]
    if missing_cat_cols:
        raise KeyError(f'categorical_features must be included in features. Missing from features: {missing_cat_cols}')

    if verbose:
        print(f"[LGBM] Required columns verified: {len(required_cols)} columns")
        print(f"[LGBM] Outer split uses group_col='{group_col}', stratify_cols={stratify_cols}, size_col='{size_col}'")

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

    X_train = train_df[features]
    y_train = train_df[target_col]
    groups_train = train_df[group_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    if verbose:
        print(f"[LGBM] Feature matrix prepared: train={X_train.shape}, test={X_test.shape}")
        print(f"[LGBM] Inner CV uses GroupKFold(n_splits={n_splits}) by '{group_col}'")
    
    tuning_strategy = 'cv' if use_optuna else 'search'
    tuning_parallelism = _resolve_parallelism(
        search_jobs=search_jobs,
        cv_jobs=cv_jobs,
        model_n_jobs=model_n_jobs,
        strategy=tuning_strategy,
    )
    
    if verbose:
        print(
            '[LGBM] Effective tuning parallelism: '
            f"strategy={tuning_strategy}, "
            f"search_jobs={tuning_parallelism['search_jobs']}, "
            f"cv_jobs={tuning_parallelism['cv_jobs']}, "
            f"model_n_jobs={tuning_parallelism['model_n_jobs']}"
        )

    cv = _get_cv(n_splits=n_splits, random_state=random_state)

    best_params = None
    best_score = None

    if use_optuna:
        try:
            if verbose:
                print(f"[LGBM] Hyperparameter tuning with Optuna (n_trials={n_trials})")
            best_params, best_score = _tune_with_optuna(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                cv=cv,
                random_state=random_state,
                param_space=param_space,
                n_trials=n_trials,
                artifacts_dir=dirs['artifacts_dir'],
                verbose=verbose,
                search_jobs=search_jobs,
                cv_jobs=cv_jobs,
                model_n_jobs=model_n_jobs,
                categorical_features=categorical_features,
                early_stopping_rounds=early_stopping_rounds,
            )
        except Exception:
            if verbose:
                print("Optuna not available; falling back to manual random search")
                print(f"[LGBM] Random search tuning (n_trials={n_trials})")
            best_params, best_score = _tune_with_random_search(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                cv=cv,
                random_state=random_state,
                param_space=param_space,
                n_trials=n_trials,
                verbose=verbose,
                search_jobs=search_jobs,
                model_n_jobs=model_n_jobs,
                categorical_features=categorical_features,
                early_stopping_rounds=early_stopping_rounds,
                cv_jobs=cv_jobs,
            )
    else:
        if verbose:
            print(f"[LGBM] Random search tuning (n_trials={n_trials})")
        best_params, best_score = _tune_with_random_search(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            cv=cv,
            random_state=random_state,
            model_n_jobs=model_n_jobs,
            param_space=param_space,
            n_trials=n_trials,
            verbose=verbose,
            search_jobs=search_jobs,
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
            cv_jobs=cv_jobs,
        )

    # Train final model on full training set
    if verbose:
        print("[LGBM] Training final model with best parameters")
    final_model = _train_final_model(
        X_train,
        y_train,
        groups_train,
        best_params,
        random_state=random_state,
        model_n_jobs=model_n_jobs,
        categorical_features=categorical_features,
        early_stopping_rounds=early_stopping_rounds,
        validation_splits=validation_splits,
    )

    # Test set evaluation
    holdout = _evaluate_holdout(final_model, X_test, y_test)

    if verbose:
        print(
            "[LGBM] Holdout evaluation completed: "
            f"RMSE={holdout['rmse']:.4f}, MAE={holdout['mae']:.4f}, R2={holdout['r2']:.4f}"
        )
    test_output_df = build_holdout_predictions_frame(test_df, y_test, holdout['y_pred'])

    metrics = {
        'rmse': float(holdout['rmse']),
        'mae': float(holdout['mae']),
        'r2': float(holdout['r2']),
        'cv_best_score': float(best_score),
    }

    metadata = {
        'random_state': int(random_state),
        'n_splits': int(n_splits),
        'use_optuna': bool(use_optuna),
        'n_trials': int(n_trials),
        'test_fold': int(test_fold),
        'features': features,
        'model_n_jobs': int(model_n_jobs),
        'cv_jobs': int(cv_jobs),
        'search_jobs': int(search_jobs),
        'effective_tuning_strategy': tuning_strategy,
        'effective_tuning_parallelism': tuning_parallelism,
        'group_col': group_col,
        'stratify_cols': stratify_cols,
        'categorical_features': categorical_features,
        'size_col': size_col,
        'target_col': target_col,
    }
    paths = _save_artifacts(
        final_model=final_model,
        best_params=best_params,
        metrics=metrics,
        metadata=metadata,
        X_test=X_test,
        y_test=y_test,
        test_output_df=test_output_df,
        y_pred=holdout['y_pred'],
        dirs=dirs,
    )

    if verbose:
        print(f"[LGBM] Artifacts saved to: {save_dir}")
        print(f"[LGBM] Final model: {paths['final_model_path']}")
        print(f"[LGBM] Test features: {paths['X_test_path']}")
        print(f"[LGBM] Test target: {paths['y_test_path']}")
        print(f"[LGBM] Predictions: {paths['y_pred_path']}")
        print(f"[LGBM] Test predictions: {paths['test_output_path']}")

    return {
        **paths,
        'best_params': best_params,
        'metrics': metrics,
        'holdout_predictions': holdout['y_pred'],
    }


if __name__ == '__main__':
    print('This module provides `train_lgbm_pipeline` for programmatic use.')
