'''
general: city bounds -> bbox -> 3 scale grids -> mask by boudns -> save grids

details: city bounds -> bbox -> save bbox and construct folder level
basic folder structure: data/processed/{city}
city/
├── raw/
│     ├── lst_30m.tif
│     ├── bbox.json
│     ├── ...
│     └── lulc_10m.tif
│
├── grids/
│     ├── grid_100m.parquet
│     ├── grid_250m.parquet
│     └── grid_500m.parquet
│
├── metrics/
│     ├── metrics_100m.parquet
│     ├── metrics_250m.parquet
│     └── metrics_500m.parquet
'''
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box

def generate_bbox(city_bounds):
    """Generates a bounding box geometry from tuple bounds."""
    minx, miny, maxx, maxy = city_bounds
    return box(minx, miny, maxx, maxy)

def generate_grids(city_bounds, grid_sizes, crs):
    """
    Generates vectorized vector grids covering the city bounds.
    Ensures complete coverage by adjusting upper bounds and assigns CRS.
    """
    minx, miny, maxx, maxy = city_bounds
    grids = {}
    
    for size in grid_sizes:
        # Pad maxx and maxy by adding 'size' to guarantee complete coverage over the boundary edges
        x_coords = np.arange(minx, maxx + size, size)
        y_coords = np.arange(miny, maxy + size, size)
        
        # Vectorized grid creation using numpy meshgrid to avoid slow nested python loops
        xv, yv = np.meshgrid(x_coords[:-1], y_coords[:-1])
        xv = xv.flatten()
        yv = yv.flatten()
        
        # Create boxes directly from vectorized coordinates
        grid_cells = [box(x, y, x + size, y + size) for x, y in zip(xv, yv)]
        
        # Explicitly assign the CRS during initialization
        grids[size] = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
        
    return grids

def main():
    # Define constants and parameters
    CRS = 'EPSG:25832'
    full_polygon_path = '../../data/GHS_UCDB_GLOBE_R2024A.gpkg'
    layer_keep = "GHS_UCDB_THEME_GEOGRAPHY_GLOBE_R2024A"
    col_keep = ['ID_UC_G0', 'GC_UCN_MAI_2025', 'GC_CNT_GAD_2025', 
                'GC_UCA_KM2_2025', 'GC_POP_TOT_2025', 'geometry']
    grid_sizes = [100, 250, 500]

    # 1. Memory Optimization: Filter columns early during I/O if driver supports it, 
    # or drop unnecessary columns immediately to save RAM
    full_gdf = gpd.read_file(full_polygon_path, layer=layer_keep)
    
    # Standardize column names to fix potential BOM artifacts before matching
    full_gdf.columns = full_gdf.columns.str.replace("\ufeff", "", regex=False)
    full_gdf_subset = full_gdf[col_keep].copy()

    # 3. Data Cleaning: Clean text columns ONLY on the subset to boost processing speed
    text_columns = full_gdf_subset.select_dtypes(include=["object", "string"]).columns
    cleaned_text_df = full_gdf_subset[text_columns].astype(str)
    for col in text_columns:
        cleaned_text_df[col] = cleaned_text_df[col].str.replace("\ufeff", "", regex=False)
    full_gdf_subset.update(cleaned_text_df)

    # 2. Pipeline Optimization: Filter for target country FIRST to reduce row count drastically
    # Using .eq() to safely handle potential NA/NaN values
    germany_gdf = full_gdf_subset[full_gdf_subset['GC_CNT_GAD_2025'].eq('Germany')][col_keep].copy()
    print(f"Filtered Germany dataset: {len(germany_gdf)} rows retained out of {len(full_gdf_subset)} total rows.")
    # Explicitly clear full dataset from memory
    del full_gdf, full_gdf_subset, cleaned_text_df

    # 4. Project Coordinates
    germany_gdf = germany_gdf.to_crs(CRS)

    # 5. Iterative Grid Generation, Masking, and Export
    for idx, row in germany_gdf.iterrows():
        city_id = row['ID_UC_G0']
        city_name = row['GC_UCN_MAI_2025']
        
        # ------------------------------------------------------------------
        # FIX 1: Sanitize text and translate German umlauts to standard ASCII.
        # This completely eliminates PyArrow's encoding confusion.
        # ------------------------------------------------------------------
        safe_city_name = str(city_name).replace(" ", "_").replace("/", "-")
        safe_city_name = (
            safe_city_name.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "Ae")
            .replace("Ö", "Oe")
            .replace("Ü", "Ue")
            .replace("ß", "ss")
        )
        
        city_geometry = row['geometry']
        city_bounds = city_geometry.bounds
        
        city_gdf_single = gpd.GeoDataFrame(geometry=[city_geometry], crs=CRS)
        
        bbox = generate_bbox(city_bounds)
        grids = generate_grids(city_bounds, grid_sizes, crs=CRS)
        
        # ------------------------------------------------------------------
        # FIX 2: Convert the relative path into a strictly absolute local path.
        # PyArrow handles absolute Windows paths (e.g., C:\...) via a different 
        # internal logic that sidesteps the URI parser bug.
        # ------------------------------------------------------------------
        city_folder_path = Path("../../data") / "processed" / f"{safe_city_name}_{city_id}"
        city_folder_path = city_folder_path.resolve()  # Force to absolute path
        city_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Export bounding box (GeoJSON)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=CRS)
        bbox_gdf.to_file(city_folder_path / 'bbox.geojson', driver='GeoJSON')
        print(f"Generated bbox for {city_name} (ID: {city_id}) and saved to {city_folder_path / 'bbox.geojson'}")
        
        # Mask grids by city bounds and Export
        for size, grid_gdf in grids.items():
            masked_grid = gpd.sjoin(grid_gdf, city_gdf_single, how="inner", predicate="intersects")
            masked_grid = masked_grid.drop(columns=["index_right"]).reset_index(drop=True)
            
            if masked_grid.empty:
                continue
                
            # --------------------------------------------------------------
            # FIX 3: Explicitly convert pathlib.Path to an absolute string.
            # Some older PyArrow engines require the explicit string conversion.
            # --------------------------------------------------------------
            output_parquet_path = str(city_folder_path / f'grid_{size}m.parquet')
            masked_grid.to_parquet(output_parquet_path, index=False)
            print(f"Generated grid for {city_name} (ID: {city_id}) with size {size}m and saved to {output_parquet_path}")
if __name__ == "__main__":
    main()