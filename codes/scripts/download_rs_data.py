import os
import ee
import re
import glob
import geemap 
import geopandas as gpd
from pathlib import Path

def gdf_to_ee_polygon(gdf):
    
    # Convert the local gdf to an Earth Engine Polygon
    fc = geemap.gdf_to_ee(gdf)
    ee_polygon = fc.first().geometry()
    
    return ee_polygon

def mask_landsat(image):
    # 1. get QA bands
    qa = image.select('QA_PIXEL')
    stQa = image.select('ST_QA')  # 专门针对温度产品的质量波段

    # 2. QA_PIXEL bitmask
    bitMask = {
        'fill':          1 << 0,  # 填充值
        'dilatedCloud':  1 << 1,  # 膨胀云
        'cirrus':        1 << 2,  # 卷云
        'cloud':         1 << 3,  # 云
        'cloudShadow':   1 << 4   # 云阴影
    }

    qaMask = (qa.bitwiseAnd(bitMask['fill']).eq(0)
              .And(qa.bitwiseAnd(bitMask['dilatedCloud']).eq(0))
              .And(qa.bitwiseAnd(bitMask['cirrus']).eq(0))
              .And(qa.bitwiseAnd(bitMask['cloud']).eq(0))
              .And(qa.bitwiseAnd(bitMask['cloudShadow']).eq(0)))

    # 3. ST_QA mask, excluede pixels with LST error > 2.5K
    stErrorMask = stQa.multiply(0.01).lte(5)

    # 4. radiometric saturation mask, exclude pixels with radiometric saturation (bit 9 of QA_RADSAT)
    radSatMask = image.select('QA_RADSAT').bitwiseAnd(1 << 9).eq(0)

    # 5. apply all masks to the image
    return (image.updateMask(qaMask)
            .updateMask(stErrorMask)
            .updateMask(radSatMask))

def add_LST(image):
    lst = (image.select('ST_B10')
           .multiply(0.00341802)
           .add(149.0)
           .subtract(273.15)
           .rename('LST_Celsius'))
    return image.addBands(lst)

def filter_landsat_image(geometry, start_date, end_date):
    
    date_filter = ee.Filter.date(start_date, end_date)
    summer_filter = ee.Filter.calendarRange(6, 9, 'month')

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(geometry) \
        .filter(date_filter) \
        .filter(summer_filter)
    
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(geometry) \
        .filter(date_filter) \
        .filter(summer_filter)
    
    l8_l9 = l8.merge(l9)

    median_composite = l8_l9.map(mask_landsat).map(add_LST).select('LST_Celsius').median()

    median_clipped = median_composite.clip(geometry)

    return median_clipped

def main():
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    
    CRS = 'EPSG:25832'
    processed_dir = Path("../../data/processed/").resolve()
    geojson_files = list(processed_dir.glob("*/bbox.geojson"))
    if not geojson_files:
        print("No 'bbox.geojson' files found in the specified directory tree.")
        return

    world_cover_2021 = ee.ImageCollection('ESA/WorldCover/v200').first()
    built_up_area = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S_10m/2018').select('built_surface')
    building_heights_2018 = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018").select('built_height')
    gaia_2018 = ee.Image('Tsinghua/FROM-GLC/GAIA/v10')

    # Load the shapefile of the city
    for geojson_file in geojson_files:
        dir_name = geojson_file.parent.name
        
        print(f"\n========================================")
        print(f"Processing target city folder: {dir_name}")
        print(f"Loading boundary from: {geojson_file}")
        
        match = re.search(r'\d+$', dir_name)
        if match:
            city_id = match.group()
        else:
            # Fallback: If no trailing digits, extract all digits available
            city_id = "".join(filter(str.isdigit, dir_name))

        gdf = gpd.read_file(geojson_file)
        
        if gdf.empty:
            print(f"Warning: Spatial file {geojson_file} is empty. Skipping.")
            continue
        
        ee_polygon = gdf_to_ee_polygon(gdf)
        
        lst_image = filter_landsat_image(ee_polygon, start_date='2018-06-01', end_date='2022-09-01')                
        
        geemap.ee_export_image_to_drive(lst_image, crs=CRS, description=f"{city_id}_landsat", 
                                        fileNamePrefix=f"{city_id}_lst", 
                                        folder=dir_name, 
                                        region=ee_polygon, 
                                        scale=30)
        geemap.ee_export_image_to_drive(world_cover_2021, crs=CRS, description=f"{city_id}_worldcover2021", 
                                        fileNamePrefix=f"{city_id}_worldcover2021", 
                                        folder= dir_name, 
                                        region=ee_polygon, 
                                        scale=10)
        geemap.ee_export_image_to_drive(built_up_area, crs=CRS, description=f"{city_id}_built_up", 
                                        fileNamePrefix=f"{city_id}_built_up", 
                                        folder= dir_name, 
                                        region=ee_polygon,
                                        scale=10)
        geemap.ee_export_image_to_drive(building_heights_2018, crs=CRS, description=f"{city_id}_building_heights_2018", 
                                        fileNamePrefix=f"{city_id}_building_heights_2018", 
                                        folder= dir_name, 
                                        region=ee_polygon,
                                        scale=10)
        geemap.ee_export_image_to_drive(gaia_2018, crs=CRS, description=f"{city_id}_gaia_2018", 
                                        fileNamePrefix=f"{city_id}_gaia_2018", 
                                        folder= dir_name, 
                                        region=ee_polygon,
                                        scale=30)

if __name__ == "__main__":
    main()