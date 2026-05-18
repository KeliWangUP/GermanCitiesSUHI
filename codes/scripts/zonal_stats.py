import os
import glob
import rasterio
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
from pathlib import Path
import pylandstats as pls
from rasterstats import zonal_stats

def read_grid(grid_path):
    grid = gpd.read_file(grid_path)
    return grid

def calculate_mean(grid, raster_path, nodata_value=-9999, col_name='col_name'):
    with rasterio.open(raster_path, 'r') as raster:
        data = raster.read(1)

    mean = zonal_stats(grid, data, affine=raster.transform, stats=['mean'], nodata=nodata_value)

    grid[col_name] = [x['mean'] for x in mean]

    return grid

def calculate_non_zero_percentage(grid, raster_path, nodata_value=-9999, col_name='col_name'):
    with rasterio.open(raster_path, 'r') as raster:
        data = raster.read(1)

    data_binary = np.where(data > 0, 1, 0)
    non_zero_stats = zonal_stats(grid, data_binary, affine=raster.transform, stats=['sum'], nodata=0)
    count_stats = zonal_stats(grid, data_binary, affine=raster.transform, stats=['count'], nodata=nodata_value)

    grid[col_name] = [
        (x['sum'] / y['count']) if x['sum'] is not None and y['count'] is not None else None 
        for x, y in zip(non_zero_stats, count_stats)
    ]

    return grid

def calculate_percentage(grid, raster_path, col_name='col_name'):
    with rasterio.open(raster_path, 'r') as raster:
        data = raster.read(1)

    raster_sum = zonal_stats(grid, data, affine=raster.transform, stats=['sum'], nodata=0)
    raster_count = zonal_stats(grid, data, affine=raster.transform, stats=['count'], nodata=-9999)

    grid[col_name] = [
        (x['sum'] / y['count'] / 100) if x['sum'] is not None and y['count'] is not None else None 
        for x, y in zip(raster_sum, raster_count)
    ]

    return grid

def cal_landstats(grid_gdf, landscape_raster_path, output_csv_path):
    # calculate the landstats
    za = pls.ZonalAnalysis(landscape_raster_path, grid_gdf)

    # assgin the name of class level caculater and landscape level caculater
    class_level_cal = ['proportion_of_landscape', 'patch_density', 'edge_density']
    landscape_level_cal = ['shannon_diversity_index']

    # calculate the class level landstats and save them into different df
    class_level_cal_df = za.compute_class_metrics_df(metrics=class_level_cal)
    landscape_level_cal_df = za.compute_landscape_metrics_df(metrics=landscape_level_cal)

    # reame the columns for both df 
    class_metrics_rename = class_level_cal_df.rename(columns={'proportion_of_landscape': 'PLAND', 'patch_density': 'PD', 'edge_density': 'ED'})
    landscape_metrics_rename = landscape_level_cal_df.rename(columns={'shannon_diversity_index': 'SHDI'})

    # reset the index of both df 
    class_metrics_rename.reset_index(inplace=True)
    landscape_metrics_rename.reset_index(inplace=True)

    # pivot and rename the both df
    class_attributes = ['PLAND', 'PD', 'ED']
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

    # merge the landstats df with grid_csv_zonal_df by index
    grids_csv_with_land_cls_50m = pd.merge(grid_csv_zonal_df, landstats_df, left_index=True, right_index=True)
    grids_csv_with_land_cls_50m.drop(columns='Unnamed: 0', inplace=True)
    
    # save the landstats df to csv
    grids_csv_with_land_cls_50m.to_csv(output_csv_path, index=False)

    return print(f'New landstats data saved successfully: {output_csv_path}')

def main():
    # Define paths
    grid_sizes = [100, 250, 500]
    lst_name = '_lst.tif'
    buildt_up_name = '_built_up.tif'
    building_heights_name = '_building_heights_2018.tif'
    isa_name = '_gaia_2018.tif'
    processed_dir = Path("../../data/processed/").resolve()
    for size in grid_sizes:
        grid_path = list(processed_dir.glob(f"grid_{size}m.parquet"))
        if not grid_path:
            print(f"Grid file for size {size} does not exist. Please run the grid generation script first.")
            return
        

        # Read grid and raster
        grid = read_grid(grid_path)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=64) as pool:
        pool.map(main, [None] * 3)  # Run main() for each grid size (100m, 250m, 500m)

    print('All cities are done')