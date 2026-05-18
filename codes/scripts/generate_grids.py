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