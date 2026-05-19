import os
import glob
import rasterio
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
from pathlib import Path
import pylandstats as pls
from exactextract import exact_extract

def read_grid(grid_path):
    grid = gpd.read_file(grid_path)
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
    
    print("Pandas-accelerated zonal pipeline completed successfully.")
    return stats_df

def compute_mean_building_height(raster_path: str, grid_gdf: gpd.GeoDataFrame, output_col: str = 'building_height_mean') -> gpd.GeoDataFrame:
    return

def compute_isa_percentage(raster_path: str, grid_gdf: gpd.GeoDataFrame, output_col: str = 'isa_percentage') -> gpd.GeoDataFrame:
    return

def cal_landstats(grid_gdf, landscape_raster_path):
    # calculate the landstats
    za = pls.ZonalAnalysis(landscape_raster_path, grid_gdf)

    # assgin the name of class level caculater and landscape level caculater
    class_level_cal = ['proportion_of_landscape', 'patch_density', 'edge_density', 'largest_patch_index', 'landscape_shape_index']
    landscape_level_cal = ['shannon_diversity_index', 'contagion']

    # calculate the class level landstats and save them into different df
    class_level_cal_df = za.compute_class_metrics_df(metrics=class_level_cal)
    landscape_level_cal_df = za.compute_landscape_metrics_df(metrics=landscape_level_cal)

    # reame the columns for both df 
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

def main():
    # Define paths
    grid_sizes = [100, 250, 500, 750, 1000]
    
    lst_name = '_lst.tif'
    buildt_up_name = '_built_up.tif'
    building_heights_name = '_building_heights_2018.tif'
    isa_name = '_gaia_2018.tif'
    lulc_name = '_worldcover2021.tif'
    
    processed_dir = Path("../../data/processed/").resolve()
    
    for size in grid_sizes:
        grid_path = list(processed_dir.glob(f"grid_{size}m.parquet"))
        if not grid_path:
            print(f"Grid file for size {size} does not exist. Please run the grid generation script first.")
            continue
        # Read grid and raster
        grid = read_grid(grid_path)

        # Construct raster paths based on naming convention
        city_name = grid_path[0].parent.name.split('_')[0]  # Extract city name from path
        lst_path = processed_dir / f"{city_name}{lst_name}"
        built_up_path = processed_dir / f"{city_name}{buildt_up_name}"
        building_heights_path = processed_dir / f"{city_name}{building_heights_name}"
        isa_path = processed_dir / f"{city_name}{isa_name}"
        lulc_path = processed_dir / f"{city_name}{lulc_name}"

        

if __name__ == '__main__':
    with multiprocessing.Pool(processes=64) as pool:
        pool.map(main, [None] * 3)  # Run main() for each grid size (100m, 250m, 500m)

    print('All cities are done')