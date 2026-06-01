import json
from pathlib import Path

mlr_dir = Path("/home/GermanCitiesSUHI/data/results/mlr_results")
lgbm_dir = Path("/home/GermanCitiesSUHI/data/results/lgbm_results_selected")

scales = [100, 250, 500, 750, 1000]
zones = ['15', '26']
changed = False

for zone in zones:
    for scale in scales:
        mlr_vif_path = mlr_dir / f"Zone_{zone}" / f"scale_{scale}m" / "artifacts" / "vif_metadata.json"
        lgbm_meta_path = lgbm_dir / "by_climate_zone" / f"CZ_{zone}" / f"scale_{scale}m" / "artifacts" / "metadata.json"
        
        if not mlr_vif_path.exists() or not lgbm_meta_path.exists():
            continue
            
        with open(mlr_vif_path, 'r') as f:
            mlr_data = json.load(f)
            mlr_feats = sorted(mlr_data.get('selected_features', []))
            
        with open(lgbm_meta_path, 'r') as f:
            lgbm_data = json.load(f)
            lgbm_feats = sorted(lgbm_data.get('features', []))
            # LGBM might append stratify_col if keep_stratify_as_feature is configured, let's remove CZ_median if present
            if 'CZ_median' in lgbm_feats:
                lgbm_feats.remove('CZ_median')
                
        if mlr_feats != lgbm_feats:
            print(f"Mismatch in Zone {zone} at {scale}m!")
            print(f"  MLR features ({len(mlr_feats)}): {mlr_feats}")
            print(f"  LGBM features ({len(lgbm_feats)}): {lgbm_feats}")
            # print diff
            added = set(mlr_feats) - set(lgbm_feats)
            removed = set(lgbm_feats) - set(mlr_feats)
            if added: print(f"  Added by MLR: {added}")
            if removed: print(f"  Removed by MLR: {removed}")
            changed = True
            print()

if not changed:
    print("All features match perfectly!")
