import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def filter_features_by_vif_with_protection(
    df_subset: pd.DataFrame, 
    features: list, 
    protected_features: list = None, 
    thresh: float = 5.0
) -> dict:
    """
    Iteratively eliminates features with high multicollinearity based on VIF scores,
    while guaranteeing that features specified in the 'protected_features' list are NEVER dropped.
    
    Prints a detailed final diagnostic summary of the surviving features and their 
    concluding VIF scores before exiting the pipeline.
    """
    current_features = features.copy()
    protected_set = set(protected_features) if protected_features else set()
    
    while True:
        X = df_subset[current_features].dropna()
        
        # Defensive check: If only 1 feature remains, stop immediately
        if X.shape[1] <= 1:
            if X.shape[1] == 0:
                return {}

            X_with_const = sm.add_constant(X, has_constant='add')
            vif_value = variance_inflation_factor(X_with_const.values, 1)
            feature_name = X.columns[0]

            print(f"\n    === Final VIF Filter Summary ===")
            print(f"      - {feature_name:<22} : VIF = {vif_value:>6.2f}{' [PROTECTED]' if feature_name in protected_set else ''}")
            print(f"    =================================\n")

            return {feature_name: vif_value}
            
        # Append constant for mathematically valid standard error estimation
        X_with_const = sm.add_constant(X, has_constant='add')
        
        vif_records = []
        for col in X.columns:
            idx_in_const = X_with_const.columns.get_loc(col)
            try:
                vif = variance_inflation_factor(X_with_const.values, idx_in_const)
            except Exception:
                vif = np.nan  # Catch exceptional LAPACK errors safely
                
            vif_records.append({'feature': col, 'VIF': vif})
            
        vif_df = pd.DataFrame(vif_records)
        vif_df = vif_df.sort_values(by='VIF', ascending=False).reset_index(drop=True)

        invalid_mask = vif_df['VIF'].isna() | np.isinf(vif_df['VIF'])
        invalid_features = vif_df.loc[invalid_mask, 'feature'].tolist()
        
        if invalid_features:
            # Drop the first invalid feature encountered to break the current state
            feat_to_purge = invalid_features[0]
            
            # Print a distinct diagnostic notice for transparency
            status = " [PROTECTED BYPASSED]" if feat_to_purge in protected_set else ""
            print(f"    [VIF Action] Purging Malformed Feature: '{feat_to_purge}' (VIF = NaN/Inf){status}")
            
            # Remove it immediately from the live pool and trigger a fresh calculation loop
            current_features.remove(feat_to_purge)
            continue
        
        # Scan down the sorted list to find the candidate to drop
        feature_to_drop = None
        for _, row in vif_df.iterrows():
            if row['VIF'] <= thresh:
                break
            if row['feature'] in protected_set:
                continue
            else:
                feature_to_drop = row['feature']
                max_vif_value = row['VIF']
                break
        
        # Decision Execution Layer
        if feature_to_drop:
            print(f"    [VIF Action] Dropping: '{feature_to_drop}' (VIF: {max_vif_value:.2f} > {thresh}). Protected features bypassed.")
            current_features.remove(feature_to_drop)
        else:
            # --------------------------------------------------------------
            # ADDITION: Loop finished. Print the final summary of surviving features.
            # --------------------------------------------------------------
            print(f"\n    === Final VIF Filter Summary ===")
            # Re-sort from lowest to highest for a clean reading progression
            vif_df_sorted = vif_df.sort_values(by='VIF', ascending=True).reset_index(drop=True)
            
            for _, row in vif_df_sorted.iterrows():
                feat = row['feature']
                vif_val = row['VIF']
                
                # Tag protected features in logs to verify retention correctness
                status_tag = " [PROTECTED]" if feat in protected_set else ""
                
                # Check if it technically violates the threshold but was saved by protection
                alert_tag = " ⚠️ (Exceeds threshold!)" if vif_val > thresh and feat in protected_set else ""
                
                print(f"      - {feat:<22} : VIF = {vif_val:>6.2f}{status_tag}{alert_tag}")
            print(f"    =================================\n")

            return dict(zip(vif_df_sorted['feature'], vif_df_sorted['VIF']))


def filter_features_by_vif_with_audit(
    df_subset: pd.DataFrame,
    features: list,
    protected_features: list = None,
    thresh: float = 5.0,
) -> tuple[list, pd.DataFrame]:
    """Iteratively filter features by VIF and return both the selected feature list and an audit table."""
    current_features = features.copy()
    protected_set = set(protected_features) if protected_features else set()
    feature_state = {
        feature: {
            'feature': feature,
            'vif': np.nan,
            'selected': True,
            'protected': feature in protected_set,
            'drop_iteration': None,
            'drop_reason': None,
        }
        for feature in current_features
    }

    iteration = 0
    while current_features:
        iteration += 1
        X = df_subset[current_features].dropna()
        if X.empty:
            raise ValueError('VIF filtering failed because the active feature set became empty after dropping NaNs.')

        if X.shape[1] == 1:
            feature_name = X.columns[0]
            vif_value = variance_inflation_factor(sm.add_constant(X, has_constant='add').values, 1)
            feature_state[feature_name]['vif'] = float(vif_value)
            return current_features, pd.DataFrame([feature_state[feature_name] for feature_name in features])

        X_with_const = sm.add_constant(X, has_constant='add')
        vif_records = []
        for column_name in X.columns:
            idx = X_with_const.columns.get_loc(column_name)
            try:
                vif_value = variance_inflation_factor(X_with_const.values, idx)
            except Exception:
                vif_value = np.nan
            vif_records.append({'feature': column_name, 'vif': float(vif_value) if pd.notna(vif_value) else np.nan})

        for record in vif_records:
            feature_state[record['feature']]['vif'] = record['vif']

        invalid_records = [record for record in vif_records if pd.isna(record['vif']) or np.isinf(record['vif'])]
        if invalid_records:
            drop_record = next((record for record in invalid_records if record['feature'] not in protected_set), None)
            if drop_record is None:
                protected_invalid = [record['feature'] for record in invalid_records]
                raise ValueError(
                    'VIF filtering cannot continue because the remaining invalid features are protected: '
                    f'{protected_invalid}'
                )
            current_features.remove(drop_record['feature'])
            feature_state[drop_record['feature']]['selected'] = False
            feature_state[drop_record['feature']]['drop_iteration'] = iteration
            feature_state[drop_record['feature']]['drop_reason'] = 'nan_or_inf'
            continue

        candidates = [record for record in vif_records if record['vif'] > thresh and record['feature'] not in protected_set]
        if not candidates:
            break

        max_vif = max(record['vif'] for record in candidates)
        top_candidates = [record for record in candidates if record['vif'] == max_vif]
        drop_record = min(top_candidates, key=lambda record: current_features.index(record['feature']))
        current_features.remove(drop_record['feature'])
        feature_state[drop_record['feature']]['selected'] = False
        feature_state[drop_record['feature']]['drop_iteration'] = iteration
        feature_state[drop_record['feature']]['drop_reason'] = f'vif_gt_{thresh}'

    vif_table = pd.DataFrame([feature_state[feature] for feature in features])
    vif_table['vif'] = pd.to_numeric(vif_table['vif'], errors='coerce')
    return current_features, vif_table