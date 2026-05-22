from pathlib import Path
import geopandas as gpd
import pandas as pd
from exactextract import exact_extract
import rasterio

def append_population_patch_pipeline(pop_raster_path: str):
    """
    Appends rigorous weighted population counts to the multi-scale unified 
    matrices by executing cross-layer projection alignment and fraction-weighted zonal sum.
    """
    GRID_SIZES = [100, 250, 500, 750, 1000]
    UNIFIED_DIR = Path("../../data/unified_scale_matrices").resolve()
    
    pop_path = Path(pop_raster_path).resolve()
    if not pop_path.exists():
        print(f"❌ Error: Population raster not found at {pop_path}")
        return

    # 1. Peek inside the population raster to capture its true spatial projection (CRS)
    with rasterio.open(pop_path) as src:
        raster_crs = src.crs
        print(f"🌍 Detected Population Raster Projection: {raster_crs}")

    # 2. Outer Resolution Loop: Process each unified matrix block
    for size in GRID_SIZES:
        file_path = UNIFIED_DIR / f"merged_metrics_{size}m.parquet"
        
        if not file_path.exists():
            print(f"⚠️ Target matrix missing: {file_path.name}. Skipping patch.")
            continue
            
        print(f"\n=======================================================")
        print(f"👥 INJECTING POPULATION PATCH FOR RESOLUTION: {size}m")
        print(f"=======================================================")
        
        # Load via GeoPandas (Now fully safe because we fixed the geo-metadata in the last turn!)
        print(f"Reading spatial database matrix...")
        grid_gdf = gpd.read_parquet(file_path)
        
        # 3. CRITICAL ACADEMIC DEFENSE: Structural CRS alignment
        # We transform vector shapes to match raster projection to ensure perfect overlay overlapping
        print(f"Aligning matrix coordinates to raster frame...")
        grid_gdf_reprojected = grid_gdf.to_crs(raster_crs)
        
        # 4. Heavy Zonal Engine Execution
        # We strictly implement 'sum' for population aggregations (Count stats)
        print(f"Deploying fraction-weighted exactextract engine (Computing spatial sum)...")
        pop_stats = exact_extract(
            pop_path,
            grid_gdf_reprojected,
            'sum',
            output='pandas'
        )
        
        # 5. Extract output array and assign back to the native database
        # exact_extract returns columns named like 'sum' or 'sum_band1', we standardize it here
        output_col_name = f'pop_sum'
        raw_pop_series = pop_stats.iloc[:, 0] # Fetch the first statistic column safely
        
        # 6. Physical Regularization: Replace any sub-pixel NaN with 0.0 (Unpopulated wilderness zones)
        grid_gdf[output_col_name] = raw_pop_series.fillna(0.0).values
        
        # 7. Safe In-place Re-Commit
        # Overwriting the existing Parquet file with the newly embedded population column
        grid_gdf.to_parquet(file_path, engine='pyarrow')
        
        # Summary diagnostic stats for paper reporting logging
        total_pop_captured = grid_gdf[output_col_name].sum()
        print(f"💾 SUCCESS: Population patch injected into -> {file_path.name}")
        print(f"   Aggregated Column Added : {output_col_name}")
        print(f"   Total Germany Population Captured at this scale: {total_pop_captured:,.2f} persons.")
        print(f"   Matrix Matrix Dimensions: {grid_gdf.shape[0]} rows x {grid_gdf.shape[1]} features.")