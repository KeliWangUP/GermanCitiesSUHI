import pandas as pd
import joblib
import shap
from pathlib import Path

xtest = pd.read_parquet("data/results/lgbm_results_selected/global/scale_100m/folds/X_test.parquet")
model_path = Path("data/results/lgbm_results_selected/global/scale_100m/models/final_model.pkl")
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)
int_vals = explainer.shap_interaction_values(xtest.iloc[:10])
print(type(int_vals), int_vals.shape)
