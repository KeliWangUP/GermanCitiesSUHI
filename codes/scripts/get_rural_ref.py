import ee
import geemap

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
    summer_filter = ee.Filter.calendarRange(6, 8, 'month')

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

def build_rural_halo(feature, buffer_inner_distance=2000, buffer_distance=5000):
    """
    Creates a ring-shaped geometry (halo) around the urban core.
    
    Args:
        feature (ee.Feature): The urban core polygon.
        buffer_inner_distance (int): Distance to exclude from city edge (default 2km).
        buffer_distance (int): Outer boundary for rural sampling (default 10km).
        
    Returns:
        ee.Feature: The ring geometry with original properties.
    """
    geom = feature.geometry()
    # Use a small maxError (e.g., 100m) to optimize complex global geometries
    inner_buffer = geom.buffer(buffer_inner_distance, 100)
    outer_buffer = geom.buffer(buffer_distance, 100)
    halo = outer_buffer.difference(inner_buffer)
    
    return ee.Feature(halo).copyProperties(feature)

def get_global_stable_rural_mask(urban_mask_source, dem_mask_source):
    # calculate the slope from DEM
    dem = dem_mask_source.select('DEM').mosaic()
    dem_fixed = dem.setDefaultProjection(crs='EPSG:4326', scale=30)
    slope = ee.Terrain.slope(dem_fixed)
    # define slope mask: slope < 5 degrees
    slope_mask = slope.lt(5)   

    # define water mask from DEM 'WBM' band
    water = dem_mask_source.select('WBM').mosaic()
    water_fixed = water.setDefaultProjection(crs='EPSG:4326', scale=30)
    water_mask = water_fixed.eq(0)

    # define non-urban mask from GAIA 'change_year_index' band
    turn_gaia_masked_to_zero = urban_mask_source.select('change_year_index').updateMask(urban_mask_source.select('change_year_index'))
    turn_gaia_none_to_zero = turn_gaia_masked_to_zero.unmask(0)
    non_urban_mask = turn_gaia_none_to_zero.eq(0)

    # combine all masks: rural area = not urban & low slope & not water
    rural_mask = non_urban_mask.multiply(slope_mask).multiply(water_mask)
    return rural_mask.byte()

def generate_final_sampling_mask(selected_polygons, base_rural_mask):
    """
    Creates a global mask that isolates pure rural pixels within the halo,
    while strictly excluding a 2km zone around ALL selected urban polygons.
    
    Args:
        selected_polygons (ee.FeatureCollection): All urban core geometries in your study.
        base_rural_mask (ee.Image): The pre-defined mask (Non-Urban & Low Slope & No Water).
        
    Returns:
        ee.Image: A binary mask (1 for valid sampling pixels, 0/masked for excluded).
    """
    
    # 1. Create a Global Exclusion Zone (All cities + 2km buffer)
    # We buffer everything by 2km in the vector domain - this is the "No-Go" zone
    def buffer_2km(feat):
        return feat.buffer(2000, 100)
    
    exclusion_vec = selected_polygons.map(buffer_2km)
    
    # Rasterize the exclusion zone (1 = strictly urban or near-urban influence)
    global_exclusion_mask = ee.Image(0).byte().paint(exclusion_vec, 1).setDefaultProjection(crs=target_crs, scale=target_scale)
    
    # 2. Rasterize the Rural Halos (2km - 5km search space)
    halos = selected_polygons.map(lambda f: build_rural_halo(f))
    roi_mask = ee.Image(0).byte().paint(halos, 1).setDefaultProjection(crs=target_crs, scale=target_scale)
    
    # 3. Final Logical Combination
    # Pixel must be: Inside the halo AND NOT in the 2km exclusion zone AND satisfy your terrain/LULC criteria
    final_mask = roi_mask.updateMask(global_exclusion_mask.Not()).updateMask(base_rural_mask)
        
    return final_mask.byte()

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