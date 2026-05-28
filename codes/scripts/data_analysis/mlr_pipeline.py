import numpy as np
import pandas as pd
import statsmodels.api as sm

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