import os
import re
import glob
import tempfile
import rasterio
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
from pathlib import Path
import pylandstats as pls
from functools import reduce
from exactextract import exact_extract

def read_grid(grid_path):
    grid = gpd.read_parquet(grid_path)
    return grid

def compute_zonal_mean_absolute(raster_path: str, grid_gdf: gpd.GeoDataFrame, output_col: str = 'raster_mean') -> gpd.GeoDataFrame:
    # 1. Inspect CRS alignment (They are confirmed aligned, but we keep this for multi-city safety)
    with rasterio.open(raster_path, 'r') as src:
        raster_crs = src.crs
        
        if grid_gdf.crs != raster_crs:
            processing_gdf = grid_gdf.to_crs(raster_crs)
        else:
            processing_gdf = grid_gdf.copy()
            
    # 2. Run exactextract explicitly wrapped as a single-element list to enforce standard output
    print(f"Extracting zonal mean for {len(processing_gdf)} geometries...")
    stats_df = exact_extract(raster_path, processing_gdf, 'mean', output='pandas')
        
    # 3. rename the output column to the user-specified name
    stats_df.rename(columns={'mean': output_col}, inplace=True)
    
    print("Pandas-accelerated zonal meanpipeline completed successfully.")
    return stats_df

def compute_mean_building_height(raster_path: str, grid_gdf: gpd.GeoDataFrame, output_col: str = 'building_height_mean') -> gpd.GeoDataFrame:
    # 1. Inspect CRS alignment (They are confirmed aligned, but we keep this for multi-city safety)
    with rasterio.open(raster_path, 'r') as src:
        raster_crs = src.crs
        
        if grid_gdf.crs != raster_crs:
            processing_gdf = grid_gdf.to_crs(raster_crs)
        else:
            processing_gdf = grid_gdf.copy()
    
        # 2. mask the raster with value less than 0.5 to exclude the non-building areas.
        raster_data = src.read(1)
        raster_data[raster_data < 0.5] = np.nan
        profile = src.profile.copy()
        # Create a temporary in-memory raster with the masked data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_raster_path = os.path.join(tmpdir, "masked_builtup_heights.tif")
        with rasterio.open(tmp_raster_path, 'w', **profile) as dst:
            dst.write(raster_data.astype(np.float32), 1)

        # 3. Run exactextract explicitly wrapped as a single-element list to enforce standard output
        print(f"Extracting mean building height for {len(processing_gdf)} geometries...")
        stats_df = exact_extract(tmp_raster_path, processing_gdf, 'mean', output='pandas')

        # 3. rename the output column to the user-specified name
        stats_df.rename(columns={'mean': output_col}, inplace=True)
        print("Pandas-accelerated height pipeline completed successfully.")    
    return stats_df

def compute_isa_percentage(raster_path: str, grid_gdf: gpd.GeoDataFrame, output_col: str = 'isa_fraction') -> gpd.GeoDataFrame:
    # 1. Inspect CRS alignment (They are confirmed aligned, but we keep this for multi-city safety)
    with rasterio.open(raster_path, 'r') as src:
        raster_crs = src.crs
        
        if grid_gdf.crs != raster_crs:
            processing_gdf = grid_gdf.to_crs(raster_crs)
        else:
            processing_gdf = grid_gdf.copy()
    
        # 2. binary the raster data by setting values larger than 0 to 1 and other to 0.
        raster_data = src.read(1)
        raster_data[raster_data <= 0] = 0
        raster_data[raster_data > 0] = 1
        profile = src.profile.copy()
        # Create a temporary in-memory raster with the masked data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_raster_path = os.path.join(tmpdir, "binary_isa.tif")
        with rasterio.open(tmp_raster_path, 'w', **profile) as dst:
            dst.write(raster_data.astype(np.float32), 1)

        # 3. Run exactextract explicitly wrapped as a single-element list to enforce standard output
        print(f"Extracting isa fraction for {len(processing_gdf)} geometries...")
        stats_df = exact_extract(tmp_raster_path, processing_gdf, 'mean', output='pandas')

        # 3. rename the output column to the user-specified name
        stats_df.rename(columns={'mean': output_col}, inplace=True)
        print("Pandas-accelerated ISA pipeline completed successfully.")    
    return stats_df

def cal_landstats(grid_gdf, landscape_raster_path):
    # calculate the landstats
    za = pls.ZonalAnalysis(landscape_raster_path, grid_gdf)

    # assign the name of class level calculator and landscape level calculator
    class_level_cal = ['proportion_of_landscape', 'patch_density', 'edge_density', 'largest_patch_index', 'landscape_shape_index']
    landscape_level_cal = ['shannon_diversity_index', 'contagion']

    # calculate the class level landstats and save them into different df
    class_level_cal_df = za.compute_class_metrics_df(metrics=class_level_cal)
    landscape_level_cal_df = za.compute_landscape_metrics_df(metrics=landscape_level_cal)

    # rename the columns for both df 
    class_metrics_rename = class_level_cal_df.rename(columns={'proportion_of_landscape': 'PLAND', 'patch_density': 'PD', 'edge_density': 'ED', 'largest_patch_index': 'LPI', 'landscape_shape_index': 'LSI'})
    landscape_metrics_rename = landscape_level_cal_df.rename(columns={'shannon_diversity_index': 'SHDI', 'contagion': 'CONTAG'})

    # reset the index of both df 
    class_metrics_rename.reset_index(inplace=True)
    landscape_metrics_rename.reset_index(inplace=True)

    # pivot and rename the both df
    class_attributes = ['PLAND', 'PD', 'ED', 'LPI', 'LSI']
    class_pivoted_dfs = []
    for attr in class_attributes:
        class_pivot = class_metrics_rename.pivot_table(
            index='zone', 
            columns='class_val', 
            values=attr, 
            aggfunc='first'
        )
        class_pivot.columns = [f'{attr}_cls_{col}' for col in class_pivot.columns]
        class_pivoted_dfs.append(class_pivot)

    # Combine all pivoted tables into one
    class_pivot_results = pd.concat(class_pivoted_dfs, axis=1).reset_index()

    # join the two dataframes with the zone column
    landstats_df = pd.merge(class_pivot_results, landscape_metrics_rename, on='zone')

    # drop the zone column
    landstats_df.drop(columns='zone', inplace=True)

    return landstats_df

def process_single_city_pipeline(cities_folder, grid_sizes, processed_dir):
    
    lst_suffix = '_lst.tif'
    buildt_up_suffix = '_built_up.tif'
    building_heights_suffix = '_building_heights_2018.tif'
    isa_suffix = '_gaia_2018.tif'
    lulc_suffix = '_worldcover2021.tif'
    
    # construct the paths for the rasters
    # city number is constructed by
    match = re.search(r'\d+$', cities_folder.name)
    city_id = match.group() if match else "".join(filter(str.isdigit, cities_folder.name))
    if not city_id:
        return f"Skipped: {cities_folder.name} (No City ID)"
    
    lst_path = cities_folder / f"{city_id}{lst_suffix}"
    built_up_path = cities_folder / f"{city_id}{buildt_up_suffix}"
    building_heights_path = cities_folder / f"{city_id}{building_heights_suffix}"
    isa_path = cities_folder / f"{city_id}{isa_suffix}"
    lulc_path = cities_folder / f"{city_id}{lulc_suffix}"

    # Defensive Input Validation: Verify the raster stack is physical before launching inner scale loop
    mandatory_raster_paths = [lst_path, built_up_path, building_heights_path, isa_path, lulc_path]
    if any(not p.exists() for p in mandatory_raster_paths):
        return f"Skipped: {cities_folder.name} (Missing Rasters)"

    for size in grid_sizes:
        print(f"\n--- Extracting Resolution: {size}m ---")
        master_grid_path = cities_folder / f"grid_{size}m.parquet"
        if not master_grid_path.exists():
            print(f"Grid file for size {size} does not exist. Please run the grid generation script first.")
            continue
        
        # Read grid and raster
        grid_gdf = read_grid(master_grid_path)
        # put city id into the grid_gdf for later merging
        grid_gdf['city_id'] = city_id
        
        # Compute zonal
        lst_mean_df = compute_zonal_mean_absolute(lst_path, grid_gdf, output_col='lst_mean')
        bcr_df = compute_zonal_mean_absolute(built_up_path, grid_gdf, output_col='BCR')
        building_height_mean_df = compute_mean_building_height(building_heights_path, grid_gdf, output_col='building_height_mean')
        isa_fraction_df = compute_isa_percentage(isa_path, grid_gdf, output_col='isa_fraction')
        landstats_df = cal_landstats(grid_gdf, lulc_path)

        # Merge all the results into a single DataFrame
        city_metric_features = [lst_mean_df, bcr_df, building_height_mean_df, isa_fraction_df, landstats_df]
        final_grid_df = grid_gdf[['geometry', 'city_id']].copy()
        city_scale_merged_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), city_metric_features)
        final_merged_df = pd.concat([final_grid_df, city_scale_merged_df], axis=1)

        # Save the final merged DataFrame to a new Parquet file
        output_path = cities_folder / f"metrics_{size}m.parquet"
        final_merged_df.to_parquet(output_path, index=False)
        print(f"Saved zonal statistics for {cities_folder.name} at {size}m resolution to {output_path}")
    
    return f"Completed: {cities_folder.name}"
    
if __name__ == '__main__':
    # 1. Define constants
    GRID_SIZES = [100, 250, 500, 750, 1000]
    PROCESSED_DIR = Path("../../data/processed/").resolve()
    
    # Gather target city directories
    target_city_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir()]
    
    # 2. Scientific Core Allocation Strategy
    # Do NOT overcommit (64 is too high due to heavy RAM footprints). 
    # Recommended: count physical CPU cores or split RAM capacity by 4GB per process.
    # We choose a defensively high performance standard (e.g., 8 to 16 depending on your server)
    num_workers = min(multiprocessing.cpu_count(), 32) 
    
    print(f"Initializing Multi-Core Spatial Pipeline with {num_workers} processes over {len(target_city_folders)} cities...")
    
    # 3. Launch isolated multi-processing worker block
    # Using starmap to elegantly feed variables down to the workers
    tasks = [(folder, GRID_SIZES, PROCESSED_DIR) for folder in target_city_folders]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_city_pipeline, tasks)
        
    # Print clean diagnostic reports compiled by workers
    print("\n--- Parallel Process Summary Report ---")
    for report in results:
        print(report)
        
    print('\n🎉 All decoupled multi-city metrics extractions completed clean.')