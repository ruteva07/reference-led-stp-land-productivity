# -*- coding: utf-8 -*-
"""
STATE — annual MODIS NDVI benchmark-relative ecosystem condition for Africa
===========================================================================

Author
------
Evariste Rutebuka

Purpose
-------
Core State workflow used to construct benchmark-relative ecosystem-condition
classes from MODIS NDVI and annual ESA CCI/C3S land cover across terrestrial
Africa.

The workflow:
1) builds a common 250-m equal-area analytical grid (EPSG:6933);
2) harmonises annual CCI/C3S land cover to broad analysis classes;
3) builds annual mean NDVI surfaces for reporting years;
4) constructs fixed 2002–2006 reference NDVI distributions;
5) derives ecological-facing and policy-facing State thresholds;
6) classifies reporting-year pixels as Poor, Fair or Good;
7) writes benchmark tables, threshold-use audits, summaries and optional
   consolidated Africa-wide State raster products.

Core State definition
---------------------
State is the position of reporting-year productivity relative to a fixed
historical reference distribution for an ecologically comparable context.

The benchmark and reporting-year NDVI surfaces are intentionally constructed
differently:

REPORTING-YEAR NDVI
    For each reporting year, all valid MODIS NDVI observations from the
    selected months are averaged at each pixel to produce that year's NDVI
    surface. In the archived production configuration, all 12 months are used,
    so this is annual mean NDVI.

2002–2006 BENCHMARK NDVI
    The benchmark is NOT formed by first creating five annual-mean NDVI maps
    and then pooling those annual maps. Instead, all valid MODIS NDVI
    observations from the full 2002–2006 benchmark period are accumulated
    together at each pixel and averaged once to produce one multi-year
    benchmark NDVI value per eligible pixel.

    Reference distributions are then formed across those eligible benchmark
    pixels within the relevant contextual stratum:

        ecological-facing reference:
            broad LULC class × ecoregion

        policy-facing reference:
            broad LULC class × ecoregion × country

    The parametric thresholds used by this production script are:
        median(reference pixels) ± 1 population SD

    The alternative quantile thresholds are the 33rd and 66th percentiles.

Land-cover persistence
----------------------
The benchmark land-cover context is based on the modal broad LULC class across
2002–2006. Reference support first uses pixels retaining that class in at least
4 of 5 benchmark years. Where sample support is insufficient, the reference is
relaxed to pixels retaining the modal class in at least 3 of 5 years.

Reference-support categories
----------------------------
Reference populations are labelled:
    >= 5,000 pixels   : high
    2,000–4,999       : acceptable
    500–1,999         : weak
    < 500             : insufficient

The <500 category is intentionally retained as a diagnostic/design category.
It does not automatically prevent ecological State classification when finite
thresholds can be calculated. For policy-facing State, an insufficient
country-constrained reference may use the corresponding ecological reference
as the geographic fallback. The existing edge-case behaviour for an entirely
empty policy lookup is retained unchanged to preserve production outputs.

MODIS observation screening
---------------------------
This workflow does NOT apply the MOD13Q1 VI Quality or Pixel Reliability QA
bands. Observation control in this State implementation consists of:
    - masking raster/source NoData values;
    - masking known stored NoData codes; and
    - excluding NDVI values outside the configured valid range (-0.2 to 1.0).

This distinction should also be reflected in the manuscript Methods. No QA
filtering should be claimed unless it was performed in a separate upstream
step.

Condition codes
---------------
0 = NoData / unclassified
1 = Poor
2 = Fair
3 = Good

Data-product interpretation
---------------------------
The consolidated State rasters are reusable categorical products on the common
equal-area grid. They represent reporting-year condition relative to the fixed
benchmark, not Trend, magnitude of change, statistical confidence, or
uncertainty.

Production configuration
------------------------
Benchmark years: 2002–2006
Reporting/diagnostic years: 2001–2022
Focal data-product years: 2001, 2011, 2022
Temporal reporting composite: annual mean NDVI
Grid: EPSG:6933, 250 m
Excluded LULC groups: Sparse/bare; Water/snow/ice

"""
#%%
from __future__ import annotations

import math
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds as window_from_bounds
from shapely.ops import unary_union
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    scipy_stats = None
    SCIPY_AVAILABLE = False

# =============================================================================
# 0. USER SETTINGS — EDIT THESE PATHS
# =============================================================================

# ARCHIVE: edit only this project root for another machine.
MAIN_FOLDER = Path(r"C:\PATH\TO\Paper\Codes\Africa")
MODIS_NDVI_DIR = MAIN_FOLDER / "modis_download" / "NDVI"
MODIS_NDVI_GLOB = "*.tif"
CCI_LULC_DIR = MAIN_FOLDER / "cci_c3s_lulc_300m_africa" / "geotiff_lccs_class"
CCI_LULC_PATTERN = "*{year}*.tif"
ECOREGIONS_PATH = MAIN_FOLDER / "Polygons" / "Afr_Ecoregions2017_FG_True.gpkg"
ECOREGIONS_LAYER: Optional[str] = None
ECO_FIELD = "ECO_NAME"
ECO_CODE_FIELD = "ECO_NAME_CODE"  # numeric preferred folder code; created if missing

COUNTRIES_PATH = MAIN_FOLDER / "Polygons" / "Africa_49States_Undisputed.gpkg"
COUNTRIES_LAYER: Optional[str] = None
COUNTRY_FIELD = "ISO_A3"
COUNTRY_NAME_FIELD: Optional[str] = None
OUTPUT_ROOT = MAIN_FOLDER / "STATE_NDVI_Africa_Loop_Annual"

# =============================================================================
# 1. RUN SWITCHES
# =============================================================================

# -------------------------------------------------------------------------
# Main processing stages
# -------------------------------------------------------------------------
# Keep both False when the continental cache, ecoregion benchmarks and State
# summaries already exist. The data-product raster export below reuses them.
# ARCHIVE: enabled by default for a complete reproducible run. Existing cache
# products are reused when REUSE_EXISTING_CACHE=True.
RUN_STAGE_1_BUILD_CONTINENTAL_CACHE = True
RUN_STAGE_2_LOOP_ECOREGIONS = True
TEST_MODE = False
TEST_ECO_NAME_VALUES = ["Albertine Rift montane forests"]
TEST_ECO_CODE_VALUES: List[int] = []

# Original per-ecoregion raster writing from the main loop.
# Not needed for the consolidated Africa-wide data products.
WRITE_ECOREGION_RASTERS = False

# -------------------------------------------------------------------------
# Charts and histogram outputs
# -------------------------------------------------------------------------
# Turn all chart/histogram options off for raster-product-only runs.
WRITE_CHARTS = False
SAVE_HISTOGRAMS = False
SAVE_HISTOGRAMS_FOR_ECO_CODES: List[int] = []

RUN_CHARTS_ONLY = False
FORCE_REBUILD_CHARTS = False
MAKE_ALL_ECOREGIONS_CHARTS = False
MAKE_ECOREGION_CHARTS = False
TOP_N_CHART_ITEMS = 49

# -------------------------------------------------------------------------
# Consolidated Africa-wide State raster products
# -------------------------------------------------------------------------
# This is the only output stage intended for this raster-product run.
RUN_DATA_PRODUCT_STATE_RASTERS = True

# Requested State raster years.
DATA_PRODUCT_STATE_YEARS = [ 2008, 2022] ## you can request any year between 2001 and 2022, but only  2008 and 2022 are produced in the archived run.
# Four consolidated State products per year.
DATA_PRODUCT_STATE_SYSTEM_METHODS = [
    ("ecological", "parametric"),
    ("policy", "parametric"),
    ("ecological", "quantile"),
    ("policy", "quantile"),
]

# Reuse completed data products unless explicitly forced.
REUSE_EXISTING_DATA_PRODUCTS = True
FORCE_REBUILD_DATA_PRODUCTS = False

# Output folder is defined later, after TIME_TAG is derived.

MIN_COUNTRY_INTERSECTION_PIXELS = 500
SKIP_MISSING_ASSESSMENT_YEARS = True

# =============================================================================
# 2. TEMPORAL SETTINGS
# =============================================================================

BENCHMARK_YEARS = [2002, 2003, 2004, 2005, 2006]
FOCAL_REPORTING_YEARS = [2001, 2011, 2022]
# All years are produced once for interannual analysis.
DIAGNOSTIC_YEAR_WINDOWS: Dict[int, List[int]] = {
    2022: list(range(2001, 2023))
}
# Full year: [1,2,3,4,5,6,7,8,9,10,11,12]
# Example seasons:
#   Q1_Jan_Mar = [1,2,3]
#   Q2_Apr_Jun = [4,5,6]
#   Q3_Jul_Sep = [7,8,9]
#   Q4_Oct_Dec = [10,11,12]
COMPOSITE_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
COMPOSITE_LABEL: Optional[str] = "Annual"

NDVI_COMPOSITE_METHOD = "mean"  # production loop supports mean
MIN_VALID_OBS_ASSESSMENT: Optional[int] = None
MIN_VALID_OBS_BENCHMARK: Optional[int] = None

# =============================================================================
# 3. SPATIAL / PERFORMANCE SETTINGS
# =============================================================================

EA_CRS = "EPSG:6933"
EA_RES_M = 250.0
GRID_BUFFER_M = 1000.0
ALL_TOUCHED = True
BLOCK_SIZE = 512

COMPRESS = "ZSTD"
ZSTD_LEVEL = 12
TILE_SIZE = 512
REUSE_EXISTING_CACHE = True
FORCE_REBUILD_CACHE = False

# =============================================================================
# 4. NDVI, STABILITY, THRESHOLD SETTINGS
# =============================================================================

NDVI_SCALE = 10000.0
NDVI_NODATA_VALUES = [-32768, -9999, 32767]
VALID_NDVI_RANGE = (-0.2, 1.0)

STABILITY_PRIMARY = 4
STABILITY_FALLBACK = 3
MIN_PIXELS_HIGH = 5000
MIN_PIXELS_ACCEPTABLE = 2000
MIN_PIXELS_WEAK = 500

PARAM_CENTER = "median"  # median is more robust for non-normal benchmarks
STD_FACTOR = 1.0
Q_LOW = 33
Q_HIGH = 66

HIST_NBINS = 60
NORMALITY_SAMPLE_MAX = 5000
RANDOM_SEED = 42

# =============================================================================
# 5. LULC MAJOR GROUPS
# =============================================================================

LULC_GROUP_LABELS: Dict[int, str] = {
    1: "Cropland",
    2: "Cropland mosaic",
    3: "Natural/cropland mosaic",
    4: "Tree cover",
    5: "Shrubland",
    6: "Grassland",
    7: "Sparse/bare",
    8: "Wetland/flooded veg",
    9: "Urban",
    10: "Water/snow/ice",
    11: "Natural vegetation mosaic",
}

CCI_TO_MAJOR: Dict[int, int] = {
    10: 1, 11: 1, 12: 1, 20: 1,
    30: 2, 40: 3,
    50: 4, 60: 4, 61: 4, 62: 4, 70: 4, 71: 4, 72: 4,
    80: 4, 81: 4, 82: 4, 90: 4,
    120: 5, 121: 5, 122: 5,
    130: 6,
    150: 7, 151: 7, 152: 7, 153: 7, 200: 7, 201: 7, 202: 7,
    160: 8, 170: 8, 180: 8,
    190: 9,
    210: 10, 220: 10,
    100: 11, 110: 11, 140: 11,
}

EXCLUDE_LULC_GROUP_LABELS = ["Sparse/bare", "Water/snow/ice"]
EXCLUDE_LULC_GROUPS = [k for k, v in LULC_GROUP_LABELS.items() if v in EXCLUDE_LULC_GROUP_LABELS]
ANALYSIS_LULC_GROUPS = [k for k in LULC_GROUP_LABELS if k not in EXCLUDE_LULC_GROUPS]
# Consistent State colour palette: poor = red, fair = neutral, good = green.
COND_COLORS = {
    "Poor": "#b2182b",
    "Fair": "#d9d9d9",
    "Good": "#1b7837",
}
COND_ORDER = ["Poor", "Fair", "Good"]
COND_AREA_COLS = ["Poor_ha", "Fair_ha", "Good_ha"]
COND_PCT_COLS = ["Poor_pct", "Fair_pct", "Good_pct"]
COND_COLOR_LIST = [COND_COLORS[c] for c in COND_ORDER]

# =============================================================================
# 6. DERIVED SETTINGS
# =============================================================================

def make_month_tag(months: List[int], custom_label: Optional[str]) -> str:
    months = sorted(months)
    if custom_label:
        return custom_label
    if months == list(range(1, 13)):
        return "FullYear"
    return "M" + "_".join(f"{m:02d}" for m in months)


def infer_min_valid_assessment(months: List[int]) -> int:
    if MIN_VALID_OBS_ASSESSMENT is not None:
        return int(MIN_VALID_OBS_ASSESSMENT)
    if sorted(months) == list(range(1, 13)):
        return 5
    return max(2, int(math.ceil(len(months) * 0.75)))


def infer_min_valid_benchmark(months: List[int]) -> int:
    if MIN_VALID_OBS_BENCHMARK is not None:
        return int(MIN_VALID_OBS_BENCHMARK)
    return infer_min_valid_assessment(months) * len(BENCHMARK_YEARS)


COMPOSITE_MONTHS = sorted(COMPOSITE_MONTHS)
TIME_TAG = make_month_tag(COMPOSITE_MONTHS, COMPOSITE_LABEL)
BENCHMARK_TAG = f"{BENCHMARK_YEARS[0]}_{BENCHMARK_YEARS[-1]}"
BENCHMARK_TIME_TAG = f"{BENCHMARK_TAG}_{TIME_TAG}"
ASSESSMENT_YEARS = sorted(set(y for years in DIAGNOSTIC_YEAR_WINDOWS.values() for y in years))
ASSESSMENT_TAG = f"{ASSESSMENT_YEARS[0]}_{ASSESSMENT_YEARS[-1]}"
MIN_VALID_BENCHMARK = infer_min_valid_benchmark(COMPOSITE_MONTHS)
MIN_VALID_ASSESSMENT = infer_min_valid_assessment(COMPOSITE_MONTHS)

CACHE_DIR = OUTPUT_ROOT / "_continental_cache" / TIME_TAG
ECOREGION_OUT_DIR = OUTPUT_ROOT / "ecoregions" / TIME_TAG
LOOKUP_DIR = OUTPUT_ROOT / "lookups"

# Data-product output folder. Defined here because TIME_TAG is only available
# after the temporal/derived settings above have been evaluated.
DATA_PRODUCT_ROOT = OUTPUT_ROOT / "DATA_PRODUCTS" / "STATE" / TIME_TAG

# =============================================================================
# 7. UTILITIES
# =============================================================================

def _fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    mkdir(path.parent)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def safe_name(txt: str) -> str:
    txt = str(txt)
    txt = re.sub(r"[^A-Za-z0-9_\-]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt[:120]


def get_pixel_area_ha(transform) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d) / 10000.0


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]


def read_vector(path: Path, layer: Optional[str]) -> gpd.GeoDataFrame:
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def safe_union(geoms: Iterable):
    return unary_union(list(geoms))


def make_equal_area_grid(aoi_gdf: gpd.GeoDataFrame, out_crs: str, res_m: float, buffer_m: float):
    aoi_ea = aoi_gdf.to_crs(out_crs)
    minx, miny, maxx, maxy = aoi_ea.total_bounds
    minx -= buffer_m; miny -= buffer_m; maxx += buffer_m; maxy += buffer_m
    minx = math.floor(minx / res_m) * res_m
    miny = math.floor(miny / res_m) * res_m
    maxx = math.ceil(maxx / res_m) * res_m
    maxy = math.ceil(maxy / res_m) * res_m
    width = int(round((maxx - minx) / res_m))
    height = int(round((maxy - miny) / res_m))
    return from_origin(minx, maxy, res_m, res_m), width, height


def base_profile(transform, width: int, height: int, crs: str, dtype: str, nodata) -> dict:
    return {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": dtype, "crs": crs, "transform": transform, "nodata": nodata,
        "compress": COMPRESS, "zstd_level": ZSTD_LEVEL,
        "tiled": True, "blockxsize": TILE_SIZE, "blockysize": TILE_SIZE,
        "BIGTIFF": "IF_SAFER",
    }


def write_raster(path: Path, arr: np.ndarray, transform, crs: str, nodata, dtype: str):
    mkdir(path.parent)
    prof = base_profile(transform, arr.shape[1], arr.shape[0], crs, dtype, nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    print(f"[write] {path}")


def should_build(path: Path) -> bool:
    if FORCE_REBUILD_CACHE:
        return True
    if REUSE_EXISTING_CACHE and path.exists():
        print(f"[reuse] {path}")
        return False
    return True


def should_build_data_product(path: Path) -> bool:
    """Return True when a consolidated data-product raster should be created."""
    if FORCE_REBUILD_DATA_PRODUCTS:
        return True
    if REUSE_EXISTING_DATA_PRODUCTS and path.exists():
        print(f"[reuse data product] {path}")
        return False
    return True


def iter_windows(width: int, height: int, block_size: int = BLOCK_SIZE):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def clip_window(win: Window, width: int, height: int) -> Window:
    col_off = max(0, int(math.floor(win.col_off)))
    row_off = max(0, int(math.floor(win.row_off)))
    col_max = min(width, int(math.ceil(win.col_off + win.width)))
    row_max = min(height, int(math.ceil(win.row_off + win.height)))
    return Window(col_off, row_off, max(0, col_max - col_off), max(0, row_max - row_off))


def read_window(path: Path, win: Window, as_float=False, nodata_to_nan=False):
    with rasterio.open(path) as src:
        arr = src.read(1, window=win)
        if as_float:
            arr = arr.astype(np.float32)
        if nodata_to_nan:
            nd = src.nodata
            if nd is not None:
                arr = arr.astype(np.float32)
                arr[arr == nd] = np.nan
        return arr

# =============================================================================
# 8. INPUT DISCOVERY AND CONVERSION
# =============================================================================

def find_yearly_lulc(year: int) -> Path:
    hits = sorted(CCI_LULC_DIR.glob(CCI_LULC_PATTERN.format(year=year)))
    if not hits:
        hits = sorted(CCI_LULC_DIR.rglob(f"*{year}*.tif"))
    if not hits:
        raise FileNotFoundError(f"No CCI LULC raster found for {year} in {CCI_LULC_DIR}")
    if len(hits) > 1:
        print(f"[warn] multiple LULC rasters for {year}; using {hits[0]}")
    return hits[0]


def parse_date_from_name(path: Path) -> Optional[datetime]:
    name = path.name
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", name)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d)
    m = re.search(r"A(20\d{2})(\d{3})", name)
    if m:
        y, doy = int(m.group(1)), int(m.group(2))
        return datetime(y, 1, 1) + timedelta(days=doy - 1)
    return None


def find_ndvi_files_for_years(years: List[int], months: List[int], required=True) -> List[Path]:
    selected = []
    for p in sorted(MODIS_NDVI_DIR.glob(MODIS_NDVI_GLOB)):
        dt = parse_date_from_name(p)
        if dt and dt.year in years and dt.month in months:
            selected.append(p)
    if required and not selected:
        raise FileNotFoundError(f"No MODIS NDVI files for years={years}, months={months}")
    return selected


def map_cci_to_major(cci_arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cci_arr, dtype=np.uint8)
    for cci_code, group_id in CCI_TO_MAJOR.items():
        out[cci_arr == cci_code] = group_id
    return out


def derive_lulc_mode_and_stability(lulc_stack: np.ndarray, valid_groups: List[int]):
    counts = [np.sum(lulc_stack == gid, axis=0).astype(np.uint8) for gid in valid_groups]
    count_cube = np.stack(counts, axis=0)
    max_idx = np.argmax(count_cube, axis=0)
    max_count = np.max(count_cube, axis=0).astype(np.uint8)
    mode = np.zeros(lulc_stack.shape[1:], dtype=np.uint8)
    groups = np.array(valid_groups, dtype=np.uint8)
    mode[max_count > 0] = groups[max_idx[max_count > 0]]
    return mode, max_count

# =============================================================================
# 9. VECTOR AND GRID PREP
# =============================================================================

def prepare_vectors():
    print("\n" + "=" * 90)
    print("[1] Preparing vectors")
    print("=" * 90)
    ecos = clean_geometries(read_vector(ECOREGIONS_PATH, ECOREGIONS_LAYER))
    countries = clean_geometries(read_vector(COUNTRIES_PATH, COUNTRIES_LAYER))
    if ECO_FIELD not in ecos.columns:
        raise KeyError(f"Missing {ECO_FIELD} in ecoregions")
    if COUNTRY_FIELD not in countries.columns:
        raise KeyError(f"Missing {COUNTRY_FIELD} in countries")
    if ecos.crs is None or countries.crs is None:
        raise ValueError("Vectors must have CRS defined")
    if ECO_CODE_FIELD not in ecos.columns:
        ecos[ECO_CODE_FIELD] = np.arange(1, len(ecos) + 1, dtype=np.int32)
    ecos[ECO_CODE_FIELD] = ecos[ECO_CODE_FIELD].astype(int)
    if TEST_MODE:
        if TEST_ECO_CODE_VALUES:
            ecos = ecos[ecos[ECO_CODE_FIELD].isin(TEST_ECO_CODE_VALUES)].copy()
        else:
            ecos = ecos[ecos[ECO_FIELD].isin(TEST_ECO_NAME_VALUES)].copy()
        if ecos.empty:
            raise ValueError("TEST_MODE selected no ecoregions")
        print(f"[test] selected: {ecos[[ECO_CODE_FIELD, ECO_FIELD]].to_dict('records')}")
    countries = countries.to_crs(ecos.crs)
    ecos = ecos.reset_index(drop=True).copy()
    countries = countries.reset_index(drop=True).copy()
    ecos["ECO_ID"] = np.arange(1, len(ecos) + 1, dtype=np.int32)
    countries["COUNTRY_ID"] = np.arange(1, len(countries) + 1, dtype=np.int32)
    eco_lookup = ecos[["ECO_ID", ECO_CODE_FIELD, ECO_FIELD]].rename(columns={ECO_CODE_FIELD: "ECO_NAME_CODE", ECO_FIELD: "ECO_NAME"})
    country_cols = ["COUNTRY_ID", COUNTRY_FIELD]
    if COUNTRY_NAME_FIELD and COUNTRY_NAME_FIELD in countries.columns:
        country_cols.append(COUNTRY_NAME_FIELD)
    country_lookup = countries[country_cols].rename(columns={COUNTRY_FIELD: "ISO_A3"})
    mkdir(LOOKUP_DIR)
    save_csv(eco_lookup, LOOKUP_DIR / "ecoregion_lookup.csv")
    save_csv(country_lookup, LOOKUP_DIR / "country_lookup.csv")
    extent_gdf = gpd.GeoDataFrame(geometry=[safe_union(ecos.geometry)], crs=ecos.crs)
    return ecos, countries, extent_gdf, eco_lookup, country_lookup


def prepare_grid(extent_gdf):
    print("\n" + "=" * 90)
    print("[2] Preparing equal-area grid")
    print("=" * 90)
    transform, width, height = make_equal_area_grid(extent_gdf, EA_CRS, EA_RES_M, GRID_BUFFER_M)
    pix_ha = get_pixel_area_ha(transform)
    print(f"[grid] {width:,} cols × {height:,} rows = {width*height:,} pixels; pixel={pix_ha:.4f} ha")
    return {"transform": transform, "width": width, "height": height, "shape": (height, width), "pix_ha": pix_ha}

# =============================================================================
# 10. CONTINENTAL CACHE
# =============================================================================

def create_id_rasters(ecos, countries, grid):
    mkdir(CACHE_DIR)
    transform, shape = grid["transform"], grid["shape"]
    eco_path = CACHE_DIR / "Ecoregion_ID_EA250m.tif"
    country_path = CACHE_DIR / "Country_ID_EA250m.tif"
    if should_build(eco_path):
        e = ecos.to_crs(EA_CRS)
        arr = rasterize([(g, int(v)) for g, v in zip(e.geometry, e["ECO_ID"])], out_shape=shape, transform=transform, fill=0, all_touched=ALL_TOUCHED, dtype="int32")
        write_raster(eco_path, arr, transform, EA_CRS, 0, "int32")
        del arr
    if should_build(country_path):
        c = countries.to_crs(EA_CRS)
        arr = rasterize([(g, int(v)) for g, v in zip(c.geometry, c["COUNTRY_ID"])], out_shape=shape, transform=transform, fill=0, all_touched=ALL_TOUCHED, dtype="int32")
        write_raster(country_path, arr, transform, EA_CRS, 0, "int32")
        del arr
    return eco_path, country_path


def build_lulc_cache(grid):
    print("\n" + "=" * 90)
    print("[3] Building/rereading LULC cache")
    print("=" * 90)
    transform, width, height = grid["transform"], grid["width"], grid["height"]
    mode_path = CACHE_DIR / f"LULC_major_mode_{BENCHMARK_TAG}_EA250m.tif"
    stability_path = CACHE_DIR / f"LULC_major_mode_count_{BENCHMARK_TAG}_EA250m.tif"
    if should_build(mode_path) or should_build(stability_path):
        srcs, vrts = [], []
        try:
            for year in BENCHMARK_YEARS:
                src = rasterio.open(find_yearly_lulc(year))
                vrt = WarpedVRT(src, crs=EA_CRS, transform=transform, width=width, height=height, resampling=Resampling.nearest, src_nodata=src.nodata, dst_nodata=0)
                srcs.append(src); vrts.append(vrt)
            prof_m = base_profile(transform, width, height, EA_CRS, "uint8", 0)
            prof_s = base_profile(transform, width, height, EA_CRS, "uint8", 0)
            windows = list(iter_windows(width, height, BLOCK_SIZE))
            t0 = time.perf_counter()
            with rasterio.open(mode_path, "w", **prof_m) as dst_m, rasterio.open(stability_path, "w", **prof_s) as dst_s:
                for i, win in enumerate(windows, 1):
                    stack = []
                    for vrt in vrts:
                        data = vrt.read(1, window=win, masked=True).filled(0).astype(np.int32)
                        stack.append(map_cci_to_major(data))
                    mode, cnt = derive_lulc_mode_and_stability(np.stack(stack), list(LULC_GROUP_LABELS.keys()))
                    dst_m.write(mode, 1, window=win); dst_s.write(cnt, 1, window=win)
                    if i % 100 == 0 or i == len(windows):
                        print(f"[LULC mode] {i}/{len(windows)} | {_fmt(time.perf_counter()-t0)}")
        finally:
            for v in vrts: v.close()
            for s in srcs: s.close()
    reporting = {}
    for year in ASSESSMENT_YEARS:
        out = CACHE_DIR / f"LULC_major_reporting_{year}_EA250m.tif"
        reporting[year] = out
        if not should_build(out):
            continue
        with rasterio.open(find_yearly_lulc(year)) as src:
            with WarpedVRT(src, crs=EA_CRS, transform=transform, width=width, height=height, resampling=Resampling.nearest, src_nodata=src.nodata, dst_nodata=0) as vrt:
                prof = base_profile(transform, width, height, EA_CRS, "uint8", 0)
                windows = list(iter_windows(width, height, BLOCK_SIZE))
                t0 = time.perf_counter()
                with rasterio.open(out, "w", **prof) as dst:
                    for i, win in enumerate(windows, 1):
                        data = vrt.read(1, window=win, masked=True).filled(0).astype(np.int32)
                        dst.write(map_cci_to_major(data), 1, window=win)
                        if i % 100 == 0 or i == len(windows):
                            print(f"[LULC report {year}] {i}/{len(windows)} | {_fmt(time.perf_counter()-t0)}")
    return mode_path, stability_path, reporting


def build_ndvi_composite(
    years: List[int],
    tag: str,
    grid,
    min_valid: int,
    required=True
) -> Optional[Tuple[Path, Path]]:
    """
    Build an NDVI mean composite across all selected source observations on the common EA-250 m grid.

    Important fix:
    MODIS NDVI rasters are usually int16 scaled by 10000. Therefore, masked
    integer arrays cannot be filled directly with np.nan. We first convert the
    masked array to float32, then fill masked values with np.nan.
    """

    transform, width, height = grid["transform"], grid["width"], grid["height"]

    out = CACHE_DIR / f"NDVI_{NDVI_COMPOSITE_METHOD}_{tag}_EA250m.tif"
    cnt_out = CACHE_DIR / f"NDVI_valid_count_{tag}_EA250m.tif"

    # Reuse only if both outputs already exist.
    if (not FORCE_REBUILD_CACHE) and REUSE_EXISTING_CACHE and out.exists() and cnt_out.exists():
        print(f"[reuse] {out}")
        print(f"[reuse] {cnt_out}")
        return out, cnt_out

    # If only one of the pair exists, rebuild both to avoid mismatched cache outputs.
    if out.exists() and not cnt_out.exists():
        print(f"[cache-warning] NDVI composite exists but count raster is missing. Rebuilding both: {tag}")
        try:
            out.unlink()
        except Exception:
            pass

    if cnt_out.exists() and not out.exists():
        print(f"[cache-warning] count raster exists but NDVI composite is missing. Rebuilding both: {tag}")
        try:
            cnt_out.unlink()
        except Exception:
            pass

    files = find_ndvi_files_for_years(
        years,
        COMPOSITE_MONTHS,
        required=required
    )

    if not files:
        print(f"[skip] no NDVI files for {tag}")
        return None

    if NDVI_COMPOSITE_METHOD.lower() != "mean":
        raise NotImplementedError(
            "Production loop currently supports NDVI_COMPOSITE_METHOD='mean' only."
        )

    print(
        f"[NDVI] {tag}: {len(files)} files | "
        f"years={years} | months={COMPOSITE_MONTHS} | min_valid={min_valid}"
    )

    srcs, vrts = [], []

    try:
        for fp in files:
            src = rasterio.open(fp)

            vrt = WarpedVRT(
                src,
                crs=EA_CRS,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )

            srcs.append(src)
            vrts.append(vrt)

        prof_n = base_profile(
            transform,
            width,
            height,
            EA_CRS,
            "float32",
            -9999
        )

        prof_c = base_profile(
            transform,
            width,
            height,
            EA_CRS,
            "uint16",
            0
        )

        windows = list(iter_windows(width, height, BLOCK_SIZE))
        t0 = time.perf_counter()

        with rasterio.open(out, "w", **prof_n) as dst_n, rasterio.open(cnt_out, "w", **prof_c) as dst_c:
            for i, win in enumerate(windows, 1):
                h, w = int(win.height), int(win.width)

                s = np.zeros((h, w), dtype=np.float64)
                c = np.zeros((h, w), dtype=np.uint16)

                for vrt in vrts:
                    # Read as masked array.
                    marr = vrt.read(1, window=win, masked=True)

                    # Critical fix:
                    # Convert int16 masked array to float32 BEFORE filling with np.nan.
                    arr = marr.astype(np.float32).filled(np.nan)

                    # Treat known NDVI nodata values as missing.
                    for nd in NDVI_NODATA_VALUES:
                        arr[arr == nd] = np.nan

                    # Also treat the source nodata value as missing if available.
                    if vrt.src_nodata is not None:
                        arr[arr == vrt.src_nodata] = np.nan

                    # Apply MODIS scale factor.
                    arr = arr / NDVI_SCALE

                    # No MOD13Q1 QA band is applied in this workflow.
                    # Observation screening is limited to NoData handling plus
                    # the configured valid NDVI range, preserving the production
                    # State implementation.
                    # Remove physically implausible or unwanted NDVI values.
                    lo, hi = VALID_NDVI_RANGE
                    arr[(arr < lo) | (arr > hi)] = np.nan

                    valid = np.isfinite(arr)

                    s[valid] += arr[valid]
                    c[valid] += 1

                mean = np.full((h, w), -9999, dtype=np.float32)

                ok = c >= min_valid
                mean[ok] = (s[ok] / c[ok]).astype(np.float32)

                dst_n.write(mean, 1, window=win)
                dst_c.write(c, 1, window=win)

                if i % 100 == 0 or i == len(windows):
                    print(
                        f"[NDVI {tag}] {i}/{len(windows)} | "
                        f"{_fmt(time.perf_counter() - t0)}"
                    )

    finally:
        for v in vrts:
            v.close()
        for s in srcs:
            s.close()

    print(f"[write] {out}")
    print(f"[write] {cnt_out}")

    return out, cnt_out


def build_ndvi_cache(grid):
    print("\n" + "=" * 90)
    print("[4] Building/rereading NDVI cache")
    print("=" * 90)
    # Benchmark behaviour intentionally preserved from the production run:
    # all selected MODIS observations across 2002–2006 are averaged once per
    # pixel; reference-distribution median/SD or quantiles are calculated later
    # across eligible pixels within each contextual stratum.
    benchmark = build_ndvi_composite(BENCHMARK_YEARS, f"benchmark_{BENCHMARK_TIME_TAG}", grid, MIN_VALID_BENCHMARK, required=True)
    if benchmark is None:
        raise RuntimeError("Could not create benchmark NDVI composite")
    assessment = {}
    for y in ASSESSMENT_YEARS:
        res = build_ndvi_composite([y], f"assessment_{TIME_TAG}_{y}", grid, MIN_VALID_ASSESSMENT, required=not SKIP_MISSING_ASSESSMENT_YEARS)
        if res is not None:
            assessment[y] = res
        else:
            print(f"[warn] skipped assessment year {y}")
    return benchmark, assessment


def build_continental_cache(ecos, countries, grid):
    mkdir(CACHE_DIR)
    eco_id, country_id = create_id_rasters(ecos, countries, grid)
    mode, stability, reporting = build_lulc_cache(grid)
    benchmark, assessment = build_ndvi_cache(grid)
    cache = {
        "eco_id": eco_id,
        "country_id": country_id,
        "lulc_mode": mode,
        "lulc_stability": stability,
        "lulc_reporting": reporting,
        "ndvi_benchmark": benchmark[0],
        "ndvi_benchmark_count": benchmark[1],
        "ndvi_assessment": {y: p[0] for y, p in assessment.items()},
        "ndvi_assessment_count": {y: p[1] for y, p in assessment.items()},
    }
    rows = []
    for k, v in cache.items():
        if isinstance(v, dict):
            for y, p in v.items(): rows.append({"key": f"{k}_{y}", "path": str(p)})
        else:
            rows.append({"key": k, "path": str(v)})
    save_csv(pd.DataFrame(rows), CACHE_DIR / "cache_manifest.csv")
    return cache

# =============================================================================
# 11. BENCHMARK, CLASSIFICATION, SUMMARY
# =============================================================================

def compute_stats(vals: np.ndarray) -> Dict[str, float | int | str]:
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "q25": np.nan, "q33": np.nan, "q66": np.nan, "q75": np.nan, "skewness": np.nan, "kurtosis": np.nan, "normality_test": "not_run", "normality_p": np.nan}
    mean, median, std = float(np.mean(v)), float(np.median(v)), float(np.std(v, ddof=0))
    q25, q33, q66, q75 = np.percentile(v, [25, Q_LOW, Q_HIGH, 75])
    if std > 0:
        z = (v - mean) / std
        skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    else:
        skew, kurt = 0.0, 0.0
    test, pval = "not_run", np.nan
    if SCIPY_AVAILABLE and n >= 20:
        rng = np.random.default_rng(RANDOM_SEED)
        sample = v if n <= NORMALITY_SAMPLE_MAX else rng.choice(v, size=NORMALITY_SAMPLE_MAX, replace=False)
        try:
            _, pval = scipy_stats.normaltest(sample)
            test = "dagostino_k2_sampled" if n > NORMALITY_SAMPLE_MAX else "dagostino_k2"
            pval = float(pval)
        except Exception:
            test, pval = "failed", np.nan
    return {"n": n, "mean": mean, "median": median, "std": std, "min": float(np.min(v)), "max": float(np.max(v)), "q25": float(q25), "q33": float(q33), "q66": float(q66), "q75": float(q75), "skewness": skew, "kurtosis": kurt, "normality_test": test, "normality_p": pval}


def threshold_from_stats(stats: Dict[str, float | int | str]) -> Dict[str, float]:
    """
    Derive State break points from a contextual reference distribution.

    Production parametric rule: median ± 1 population SD
    (because PARAM_CENTER="median", STD_FACTOR=1.0, and compute_stats uses
    np.std(..., ddof=0)). Quantile rule: configured Q_LOW/Q_HIGH percentiles.
    """
    center = float(stats["median"] if PARAM_CENTER.lower() == "median" else stats["mean"])
    sd = float(stats["std"])
    return {"param_low": center - STD_FACTOR * sd, "param_high": center + STD_FACTOR * sd, "q_low": float(stats["q33"]), "q_high": float(stats["q66"])}


def choose_benchmark(vals4: np.ndarray, vals3: np.ndarray):
    """
    Select the benchmark population and attach a sample-support category.

    Important:
    `<500` is intentionally retained as "insufficient" rather than discarded.
    This label is a reference-support diagnostic; downstream ecological
    classification may still use finite thresholds from that reference.
    Policy-facing classification retains the original ecological-fallback
    behaviour.
    """
    if vals4.size >= MIN_PIXELS_HIGH:
        return vals4, "high", STABILITY_PRIMARY, 1
    if vals3.size >= MIN_PIXELS_ACCEPTABLE:
        return vals3, "acceptable", STABILITY_FALLBACK, 2
    if vals3.size >= MIN_PIXELS_WEAK:
        return vals3, "weak", STABILITY_FALLBACK, 3
    return vals3, "insufficient", STABILITY_FALLBACK, 99


def benchmark_record(kind, bid, gid, eco_id, eco_code, eco_name, ctry, iso, vals, stab, conf, level, source, counts):
    stats = compute_stats(vals)
    thr = threshold_from_stats(stats) if stats["n"] > 0 else {"param_low": np.nan, "param_high": np.nan, "q_low": np.nan, "q_high": np.nan}
    return {"benchmark_type": kind, "benchmark_id": bid, "benchmark_years": BENCHMARK_TAG, "time_tag": TIME_TAG, "months": ",".join(map(str, COMPOSITE_MONTHS)), "LULC_group": gid, "LULC_label": LULC_GROUP_LABELS[gid], "ECO_ID": eco_id, "ECO_NAME_CODE": eco_code, "ECO_NAME": eco_name, "COUNTRY_ID": ctry, "ISO_A3": iso, "selected_stability_rule": stab, "benchmark_confidence": conf, "fallback_level": level, "fallback_source": source, **counts, "selected_n_pixels": stats["n"], **{k: v for k, v in stats.items() if k != "n"}, **thr}


def build_benchmarks(arrays, eco_id_val, eco_code, eco_name, country_lookup, countries_in_eco, eco_out):
    nb, lm, st, ei, ci = arrays["ndvi_benchmark"], arrays["lulc_mode"], arrays["stability"], arrays["eco_id"], arrays["country_id"]
    eco_mask = ei == eco_id_val
    valid = np.isfinite(nb)
    rows_e, rows_p = [], []
    iso_map = dict(zip(country_lookup["COUNTRY_ID"].astype(int), country_lookup["ISO_A3"].astype(str)))
    for gid in ANALYSIS_LULC_GROUPS:
        base = eco_mask & (lm == gid) & valid
        vals, conf, stab, level = choose_benchmark(nb[base & (st >= STABILITY_PRIMARY)], nb[base & (st >= STABILITY_FALLBACK)])
        counts = {"n_total_pixels": int((eco_mask & (lm == gid)).sum()), "n_valid_ndvi_pixels": int(base.sum()), "n_mode5_valid": int((base & (st == 5)).sum()), "n_mode4plus_valid": int((base & (st >= 4)).sum()), "n_mode3plus_valid": int((base & (st >= 3)).sum())}
        rows_e.append(benchmark_record("ecological", f"LULC{gid}_ECO{eco_id_val}", gid, eco_id_val, eco_code, eco_name, 0, "", vals, stab, conf, level, "local_lulc_ecoregion" if conf != "insufficient" else "insufficient_ecoregion_reference", counts))
    for ctry in countries_in_eco:
        iso = iso_map.get(ctry, "")
        for gid in ANALYSIS_LULC_GROUPS:
            base = eco_mask & (ci == ctry) & (lm == gid) & valid
            vals, conf, stab, level = choose_benchmark(nb[base & (st >= STABILITY_PRIMARY)], nb[base & (st >= STABILITY_FALLBACK)])
            counts = {"n_total_pixels": int((eco_mask & (ci == ctry) & (lm == gid)).sum()), "n_valid_ndvi_pixels": int(base.sum()), "n_mode5_valid": int((base & (st == 5)).sum()), "n_mode4plus_valid": int((base & (st >= 4)).sum()), "n_mode3plus_valid": int((base & (st >= 3)).sum())}
            rows_p.append(benchmark_record("policy", f"LULC{gid}_ECO{eco_id_val}_CTRY{ctry}", gid, eco_id_val, eco_code, eco_name, ctry, iso, vals, stab, conf, level, "local_lulc_ecoregion_country" if conf != "insufficient" else "fallback_to_ecological_lulc_ecoregion", counts))
    eco_df, pol_df = pd.DataFrame(rows_e), pd.DataFrame(rows_p)
    save_csv(eco_df, eco_out / "benchmarks" / f"ecological_benchmarks_{BENCHMARK_TIME_TAG}.csv")
    save_csv(pol_df, eco_out / "benchmarks" / f"policy_benchmarks_{BENCHMARK_TIME_TAG}.csv")
    return eco_df, pol_df


def build_lookup(df: pd.DataFrame, method: str):
    lookup = {}
    for _, r in df.iterrows():
        gid, eco, ctry = int(r["LULC_group"]), int(r["ECO_ID"]), int(r["COUNTRY_ID"])
        low, high = (float(r["param_low"]), float(r["param_high"])) if method == "parametric" else (float(r["q_low"]), float(r["q_high"]))
        lookup[(gid, eco, ctry)] = {"low": low, "high": high, "confidence": r.get("benchmark_confidence", "unknown"), "benchmark_id": r.get("benchmark_id", "")}
    return lookup


def classify(ndvi, lulc, eco_arr, ctry_arr, eco_id, system, lookup, eco_lookup=None):
    out = np.zeros(ndvi.shape, dtype=np.uint8)
    audit = []
    valid = (eco_arr == eco_id) & np.isfinite(ndvi) & np.isin(lulc, ANALYSIS_LULC_GROUPS)
    if system == "policy":
        valid &= ctry_arr > 0
    for gid in ANALYSIS_LULC_GROUPS:
        if system == "ecological":
            rec = lookup.get((gid, eco_id, 0))
            if not rec or not np.isfinite(rec["low"]) or not np.isfinite(rec["high"]):
                continue
            m = valid & (lulc == gid)
            items = [(m, 0, rec)]
        else:
            items = []
            for ctry in np.unique(ctry_arr[valid & (lulc == gid)]):
                if ctry <= 0: continue
                rec = lookup.get((gid, eco_id, int(ctry)))
                if (not rec or rec.get("confidence") == "insufficient" or not np.isfinite(rec["low"]) or not np.isfinite(rec["high"])) and eco_lookup:
                    rec = eco_lookup.get((gid, eco_id, 0))
                if rec and np.isfinite(rec["low"]) and np.isfinite(rec["high"]):
                    items.append((valid & (lulc == gid) & (ctry_arr == ctry), int(ctry), rec))
        for m, ctry, rec in items:
            low, high = float(rec["low"]), float(rec["high"])
            out[m & (ndvi < low)] = 1
            out[m & (ndvi >= low) & (ndvi <= high)] = 2
            out[m & (ndvi > high)] = 3
            audit.append({"benchmark_system": system, "LULC_group": gid, "COUNTRY_ID": ctry, "n_classified": int(m.sum()), "benchmark_id_used": rec.get("benchmark_id", ""), "benchmark_confidence_used": rec.get("confidence", ""), "low": low, "high": high})
    return out, pd.DataFrame(audit)


def roles_for_year(year: int):
    roles = []
    for focal, years in DIAGNOSTIC_YEAR_WINDOWS.items():
        if year in years:
            roles.append({"focal_reporting_year": focal, "diagnostic_role": "focal" if year == focal else ("context_before" if year < focal else "context_after")})
    return roles or [{"focal_reporting_year": year, "diagnostic_role": "standalone"}]


def summarize(cond, lulc, eco_arr, ctry_arr, eco_id, eco_code, eco_name, year, system, method, country_lookup, pix_ha):
    iso_map = dict(zip(country_lookup["COUNTRY_ID"].astype(int), country_lookup["ISO_A3"].astype(str)))
    rows = []
    valid = (eco_arr == eco_id) & np.isin(lulc, ANALYSIS_LULC_GROUPS)
    if system == "policy": valid &= ctry_arr > 0
    for gid in ANALYSIS_LULC_GROUPS:
        countries = [0] if system == "ecological" else [int(x) for x in np.unique(ctry_arr[valid & (lulc == gid)]) if x > 0]
        for ctry in countries:
            m = valid & (lulc == gid)
            if system == "policy": m &= ctry_arr == ctry
            total = float(m.sum()) * pix_ha
            if total <= 0: continue
            poor = float((m & (cond == 1)).sum()) * pix_ha
            fair = float((m & (cond == 2)).sum()) * pix_ha
            good = float((m & (cond == 3)).sum()) * pix_ha
            classified = poor + fair + good
            pp, fp, gp = (poor/classified*100, fair/classified*100, good/classified*100) if classified > 0 else (0,0,0)
            for role in roles_for_year(year):
                rows.append({"period": f"Assessment_{TIME_TAG}_{year}__Benchmark_{BENCHMARK_TIME_TAG}__ReportLULC_{year}", "benchmark_years": BENCHMARK_TAG, "time_tag": TIME_TAG, "months": ",".join(map(str, COMPOSITE_MONTHS)), "assessment_year": year, "reporting_lulc_year": year, **role, "benchmark_system": system, "method": method, "LULC_group": gid, "LULC_label": LULC_GROUP_LABELS[gid], "ECO_ID": eco_id, "ECO_NAME_CODE": eco_code, "ECO_NAME": eco_name, "COUNTRY_ID": ctry, "ISO_A3": iso_map.get(ctry, "") if ctry else "", "Total_reporting_LULC_ha": total, "Classified_ha": classified, "Poor_ha": poor, "Fair_ha": fair, "Good_ha": good, "Poor_pct": pp, "Fair_pct": fp, "Good_pct": gp})
    return pd.DataFrame(rows)


# =============================================================================
# 11B. CHART-ONLY UTILITIES
# =============================================================================

def save_chart_if_needed(fig, out_png: Path, dpi: int = 300) -> None:
    """
    Save a chart only if it does not already exist, unless FORCE_REBUILD_CHARTS=True.
    The figure is always closed to avoid memory accumulation during full loops.
    """
    mkdir(out_png.parent)

    if out_png.exists() and not FORCE_REBUILD_CHARTS:
        print(f"[skip chart] {out_png}")
        plt.close(fig)
        return

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[write chart] {out_png}")


def expected_all_summary_path() -> Path:
    return OUTPUT_ROOT / f"ALL_ECOREGIONS_State_summary_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv"


def expected_all_context_summary_path() -> Path:
    return OUTPUT_ROOT / f"ALL_ECOREGIONS_State_diagnostic_context_summary_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv"


def find_ecoregion_summary_files() -> List[Path]:
    """
    Find existing ecoregion State summary CSVs for the selected time tag and
    assessment period.
    """
    pattern = f"State_summary_ECO_*_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv"
    files = sorted(ECOREGION_OUT_DIR.glob(f"ECO_*/summaries/{pattern}"))

    if not files:
        # Fallback for slightly different naming or previous exploratory runs.
        files = sorted(ECOREGION_OUT_DIR.glob("ECO_*/summaries/State_summary_ECO_*.csv"))

    return files


def read_state_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in [
        "assessment_year", "focal_reporting_year", "reporting_lulc_year",
        "LULC_group", "ECO_ID", "ECO_NAME_CODE", "COUNTRY_ID",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["Poor_ha", "Fair_ha", "Good_ha", "Classified_ha",
                "Poor_pct", "Fair_pct", "Good_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def aggregate_state_area(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Aggregate Poor/Fair/Good areas and derive percentages.
    """
    if df.empty:
        return pd.DataFrame()

    missing = [c for c in COND_AREA_COLS if c not in df.columns]
    if missing:
        print(f"[charts] missing area columns: {missing}")
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, dropna=False)[COND_AREA_COLS]
        .sum()
        .reset_index()
    )

    out["Classified_ha"] = out[COND_AREA_COLS].sum(axis=1)

    out["Poor_pct"] = np.where(
        out["Classified_ha"] > 0,
        out["Poor_ha"] / out["Classified_ha"] * 100.0,
        0.0,
    )
    out["Fair_pct"] = np.where(
        out["Classified_ha"] > 0,
        out["Fair_ha"] / out["Classified_ha"] * 100.0,
        0.0,
    )
    out["Good_pct"] = np.where(
        out["Classified_ha"] > 0,
        out["Good_ha"] / out["Classified_ha"] * 100.0,
        0.0,
    )

    return out


def plot_state_stacked_percent(
    df_pct: pd.DataFrame,
    x_col: str,
    out_png: Path,
    title: str,
    xlabel: str,
    figsize: Tuple[float, float] = (12, 7),
    rotation: int = 45,
) -> None:
    """
    Generic stacked percentage chart for Poor/Fair/Good state classes.
    """
    if df_pct.empty:
        print(f"[charts] empty data, skipping: {out_png.name}")
        return

    plot_df = df_pct.set_index(x_col)[COND_PCT_COLS].copy()
    plot_df.columns = COND_ORDER

    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=COND_COLOR_LIST,
        edgecolor="black",
        linewidth=0.2,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Area share (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(title="State class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")

    save_chart_if_needed(fig, out_png)


def plot_grouped_stacked_state_percent(
    df_pct: pd.DataFrame,
    group_col: str,
    bar_col: str,
    out_png: Path,
    title: str,
    xlabel: str,
    group_order: Optional[List] = None,
    bar_order: Optional[List] = None,
    figsize: Tuple[float, float] = (14, 7),
    rotation: int = 45,
) -> None:
    """
    Grouped stacked Poor/Fair/Good chart.

    Example uses:
      - group_col='assessment_year', bar_col='method'
        gives Parametric and Quantile bars side-by-side for each year.
      - group_col='ISO_A3', bar_col='assessment_year'
        gives 2001, 2011, 2022 bars side-by-side for each country.
    """
    if df_pct.empty:
        print(f"[charts] empty data, skipping: {out_png.name}")
        return

    required = [group_col, bar_col] + COND_PCT_COLS
    missing = [c for c in required if c not in df_pct.columns]
    if missing:
        print(f"[charts] missing columns {missing}, skipping: {out_png.name}")
        return

    tmp = df_pct.copy()
    tmp[group_col] = tmp[group_col].astype(str)
    tmp[bar_col] = tmp[bar_col].astype(str)

    if group_order is None:
        group_order = list(tmp[group_col].drop_duplicates())
    else:
        group_order = [str(x) for x in group_order]

    if bar_order is None:
        bar_order = list(tmp[bar_col].drop_duplicates())
    else:
        bar_order = [str(x) for x in bar_order]

    # Reindex to a complete group x bar grid so missing bars appear as zeros.
    idx = pd.MultiIndex.from_product(
        [group_order, bar_order],
        names=[group_col, bar_col]
    )

    tmp_idx = tmp.set_index([group_col, bar_col])
    plot_df = tmp_idx.reindex(idx)[COND_PCT_COLS].fillna(0.0).reset_index()

    n_groups = len(group_order)
    n_bars = max(1, len(bar_order))

    if n_groups == 0 or n_bars == 0:
        print(f"[charts] no groups/bars available, skipping: {out_png.name}")
        return

    x = np.arange(n_groups)
    total_group_width = 0.78
    bar_width = min(0.30, total_group_width / n_bars)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    # Hatches distinguish bar categories such as method or year without
    # changing the State-class colour meaning.
    hatch_options = ["", "///", "xx", "\\\\", "..", "--"]
    hatches = [hatch_options[i % len(hatch_options)] for i in range(n_bars)]

    fig, ax = plt.subplots(figsize=figsize)

    for j, b in enumerate(bar_order):
        sub = plot_df[plot_df[bar_col] == b].set_index(group_col).reindex(group_order)
        bottom = np.zeros(n_groups)

        for cls_name, area_col, color in zip(COND_ORDER, COND_PCT_COLS, COND_COLOR_LIST):
            vals = sub[area_col].fillna(0.0).to_numpy(dtype=float)

            ax.bar(
                x + offsets[j],
                vals,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.2,
                hatch=hatches[j],
            )
            bottom += vals

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel("Area share (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    import matplotlib.patches as mpatches

    class_handles = [
        mpatches.Patch(facecolor=COND_COLORS[c], edgecolor="black", label=c)
        for c in COND_ORDER
    ]
    bar_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatches[i], label=str(b))
        for i, b in enumerate(bar_order)
    ]

    leg1 = ax.legend(
        handles=class_handles,
        title="State class",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=bar_handles,
        title=bar_col.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 0.55),
        loc="upper left",
    )

    save_chart_if_needed(fig, out_png)


def plot_all_ecoregions_state_charts(all_df: pd.DataFrame) -> None:
    """
    Create continent-level State charts from the merged ALL_ECOREGIONS summary.

    New outputs added:
      1. State_ALL_ecological_parametric_vs_quantile_by_year
      2. State_ALL_policy_parametric_vs_quantile_by_year
      3. State_ALL_policy_country_grouped_years
      4. State_ALL_ecological_country_grouped_years, if country-resolved
         ecological rows exist in the summary.
      5. State_ALL_ecological_ecoregion_grouped_years as a useful fallback
         for ecological benchmark summaries.
    """
    if all_df.empty:
        print("[charts] all-ecoregions dataframe is empty.")
        return

    chart_dir = OUTPUT_ROOT / "charts" / TIME_TAG
    mkdir(chart_dir)

    # -------------------------------------------------------------------------
    # Existing headline: Africa-wide ecological-parametric State through time.
    # -------------------------------------------------------------------------
    sub = all_df[
        (all_df["benchmark_system"].astype(str) == "ecological")
        & (all_df["method"].astype(str) == "parametric")
    ].copy()

    if not sub.empty:
        yr = aggregate_state_area(sub, ["assessment_year"])
        yr = yr.sort_values("assessment_year")
        yr["assessment_year"] = yr["assessment_year"].astype(str)

        plot_state_stacked_percent(
            yr,
            x_col="assessment_year",
            out_png=chart_dir / f"State_ALL_ecological_parametric_by_year_{TIME_TAG}.png",
            title=f"Africa-wide State composition by year — ecological parametric ({TIME_TAG})",
            xlabel="Assessment year",
            figsize=(13, 7),
            rotation=90,
        )

    # -------------------------------------------------------------------------
    # Existing headline: Africa-wide ecological-parametric State by LULC for
    # focal years.
    # -------------------------------------------------------------------------
    focal = sub[sub["assessment_year"].isin(FOCAL_REPORTING_YEARS)].copy()
    if not focal.empty:
        ag = aggregate_state_area(focal, ["assessment_year", "LULC_label"])
        ag["label"] = ag["assessment_year"].astype(str) + " | " + ag["LULC_label"].astype(str)

        plot_state_stacked_percent(
            ag,
            x_col="label",
            out_png=chart_dir / f"State_ALL_ecological_parametric_LULC_focal_years_{TIME_TAG}.png",
            title=f"Africa-wide State by LULC and focal year — ecological parametric ({TIME_TAG})",
            xlabel="Assessment year | LULC group",
            figsize=(16, 8),
            rotation=90,
        )

    # -------------------------------------------------------------------------
    # Existing headline: top ecoregions by Poor share in latest focal year.
    # -------------------------------------------------------------------------
    possible_years = set(pd.to_numeric(sub["assessment_year"], errors="coerce").dropna().astype(int))
    latest = max([y for y in FOCAL_REPORTING_YEARS if y in possible_years], default=None)
    if latest is not None:
        latest_df = sub[sub["assessment_year"] == latest].copy()
        eco = aggregate_state_area(latest_df, ["ECO_NAME_CODE", "ECO_NAME"])
        eco = eco.sort_values("Poor_pct", ascending=False).head(TOP_N_CHART_ITEMS)

        if not eco.empty:
            labels = eco["ECO_NAME_CODE"].astype(str) + " | " + eco["ECO_NAME"].astype(str)
            fig, ax = plt.subplots(figsize=(11, max(7, len(eco) * 0.32)))
            ax.barh(labels, eco["Poor_pct"], color=COND_COLORS["Poor"])
            ax.set_xlabel("Poor condition share (%)")
            ax.set_ylabel("Ecoregion")
            ax.set_title(f"Top ecoregions by Poor State share — {latest}, ecological parametric")
            ax.invert_yaxis()
            save_chart_if_needed(
                fig,
                chart_dir / f"State_ALL_top_ecoregions_poor_pct_{latest}_{TIME_TAG}.png",
            )

    # -------------------------------------------------------------------------
    # NEW 1. Continental ecological parametric vs quantile by year.
    # Same assessment year is grouped; Parametric and Quantile bars are adjacent.
    # -------------------------------------------------------------------------
    eco_sens = all_df[
        (all_df["benchmark_system"].astype(str) == "ecological")
        & (all_df["method"].astype(str).isin(["parametric", "quantile"]))
    ].copy()

    if not eco_sens.empty:
        ag = aggregate_state_area(eco_sens, ["assessment_year", "method"])
        ag = ag.sort_values(["assessment_year", "method"])

        year_order = [str(y) for y in sorted(pd.to_numeric(ag["assessment_year"], errors="coerce").dropna().astype(int).unique())]
        method_order = ["parametric", "quantile"]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="assessment_year",
            bar_col="method",
            group_order=year_order,
            bar_order=method_order,
            out_png=chart_dir / f"State_ALL_ecological_parametric_vs_quantile_by_year_{TIME_TAG}.png",
            title=f"Africa-wide ecological State: parametric vs quantile by year ({TIME_TAG})",
            xlabel="Assessment year",
            figsize=(14, 7),
            rotation=90,
        )

    # -------------------------------------------------------------------------
    # NEW 2. Continental policy-facing parametric vs quantile by year.
    # -------------------------------------------------------------------------
    pol_sens = all_df[
        (all_df["benchmark_system"].astype(str) == "policy")
        & (all_df["method"].astype(str).isin(["parametric", "quantile"]))
    ].copy()

    if not pol_sens.empty:
        ag = aggregate_state_area(pol_sens, ["assessment_year", "method"])
        ag = ag.sort_values(["assessment_year", "method"])

        year_order = [str(y) for y in sorted(pd.to_numeric(ag["assessment_year"], errors="coerce").dropna().astype(int).unique())]
        method_order = ["parametric", "quantile"]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="assessment_year",
            bar_col="method",
            group_order=year_order,
            bar_order=method_order,
            out_png=chart_dir / f"State_ALL_policy_parametric_vs_quantile_by_year_{TIME_TAG}.png",
            title=f"Africa-wide policy-facing State: parametric vs quantile by year ({TIME_TAG})",
            xlabel="Assessment year",
            figsize=(14, 7),
            rotation=90,
        )

    # -------------------------------------------------------------------------
    # NEW 3. Policy country grouped years.
    # Each country has 2001/2011/2022 bars, stacked Poor/Fair/Good.
    # Uses parametric as the primary reporting method.
    # -------------------------------------------------------------------------
    pol_country = all_df[
        (all_df["benchmark_system"].astype(str) == "policy")
        & (all_df["method"].astype(str) == "parametric")
        & (all_df["assessment_year"].isin(FOCAL_REPORTING_YEARS))
    ].copy()

    if not pol_country.empty and "ISO_A3" in pol_country.columns:
        ag = aggregate_state_area(pol_country, ["ISO_A3", "assessment_year"])
        ag = ag[ag["ISO_A3"].astype(str).str.len() > 0]

        # Sort countries by latest focal-year Poor share.
        latest = max([y for y in FOCAL_REPORTING_YEARS if y in set(pd.to_numeric(ag["assessment_year"], errors="coerce").dropna().astype(int))], default=None)
        if latest is not None:
            latest_rank = (
                ag[ag["assessment_year"] == latest]
                .sort_values("Poor_pct", ascending=False)
            )
            country_order = latest_rank["ISO_A3"].astype(str).tolist()
        else:
            country_order = sorted(ag["ISO_A3"].astype(str).unique())

        if TOP_N_CHART_ITEMS is not None:
            country_order = country_order[:TOP_N_CHART_ITEMS]

        year_order = [str(y) for y in FOCAL_REPORTING_YEARS]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="ISO_A3",
            bar_col="assessment_year",
            group_order=country_order,
            bar_order=year_order,
            out_png=chart_dir / f"State_ALL_policy_country_grouped_years_{TIME_TAG}.png",
            title=f"Policy-facing State by country and reporting year — parametric ({TIME_TAG})",
            xlabel="Country",
            figsize=(max(14, len(country_order) * 0.45), 8),
            rotation=90,
        )

    # -------------------------------------------------------------------------
    # NEW 4. Ecological country grouped years.
    # This requires country-resolved rows under benchmark_system='ecological'.
    # The current State summary may not contain these because ecological summaries
    # were originally aggregated at ecoregion level. If absent, the chart is
    # skipped with a clear note.
    # -------------------------------------------------------------------------
    eco_country = all_df[
        (all_df["benchmark_system"].astype(str) == "ecological")
        & (all_df["method"].astype(str) == "parametric")
        & (all_df["assessment_year"].isin(FOCAL_REPORTING_YEARS))
    ].copy()

    if "ISO_A3" in eco_country.columns:
        eco_country = eco_country[eco_country["ISO_A3"].astype(str).str.len() > 0].copy()

    if not eco_country.empty:
        ag = aggregate_state_area(eco_country, ["ISO_A3", "assessment_year"])

        latest = max([y for y in FOCAL_REPORTING_YEARS if y in set(pd.to_numeric(ag["assessment_year"], errors="coerce").dropna().astype(int))], default=None)
        if latest is not None:
            latest_rank = ag[ag["assessment_year"] == latest].sort_values("Poor_pct", ascending=False)
            country_order = latest_rank["ISO_A3"].astype(str).tolist()
        else:
            country_order = sorted(ag["ISO_A3"].astype(str).unique())

        if TOP_N_CHART_ITEMS is not None:
            country_order = country_order[:TOP_N_CHART_ITEMS]

        year_order = [str(y) for y in FOCAL_REPORTING_YEARS]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="ISO_A3",
            bar_col="assessment_year",
            group_order=country_order,
            bar_order=year_order,
            out_png=chart_dir / f"State_ALL_ecological_country_grouped_years_{TIME_TAG}.png",
            title=f"Ecological-benchmark State by country and reporting year — parametric ({TIME_TAG})",
            xlabel="Country",
            figsize=(max(14, len(country_order) * 0.45), 8),
            rotation=90,
        )
    else:
        print(
            "[charts] skipped State_ALL_ecological_country_grouped_years: "
            "existing summaries do not contain country-resolved ecological rows. "
            "To create this chart, the State summary stage must also summarise "
            "ecological-benchmark classifications by country."
        )

    # -------------------------------------------------------------------------
    # NEW 5. Useful ecological fallback: ecoregion grouped years.
    # This is more consistent with ecological benchmark outputs.
    # -------------------------------------------------------------------------
    eco_ecoregion = all_df[
        (all_df["benchmark_system"].astype(str) == "ecological")
        & (all_df["method"].astype(str) == "parametric")
        & (all_df["assessment_year"].isin(FOCAL_REPORTING_YEARS))
    ].copy()

    if not eco_ecoregion.empty:
        ag = aggregate_state_area(eco_ecoregion, ["ECO_NAME_CODE", "ECO_NAME", "assessment_year"])

        latest = max([y for y in FOCAL_REPORTING_YEARS if y in set(pd.to_numeric(ag["assessment_year"], errors="coerce").dropna().astype(int))], default=None)
        if latest is not None:
            latest_rank = ag[ag["assessment_year"] == latest].sort_values("Poor_pct", ascending=False)
            eco_order_codes = latest_rank["ECO_NAME_CODE"].astype(str).tolist()
        else:
            eco_order_codes = sorted(ag["ECO_NAME_CODE"].astype(str).unique())

        if TOP_N_CHART_ITEMS is not None:
            eco_order_codes = eco_order_codes[:TOP_N_CHART_ITEMS]

        ag["eco_label"] = ag["ECO_NAME_CODE"].astype(str) + " | " + ag["ECO_NAME"].astype(str)
        label_lookup = (
            ag.drop_duplicates("ECO_NAME_CODE")
            .assign(ECO_NAME_CODE_STR=lambda d: d["ECO_NAME_CODE"].astype(str))
            .set_index("ECO_NAME_CODE_STR")["eco_label"]
            .to_dict()
        )
        group_order = [label_lookup.get(code, code) for code in eco_order_codes]
        ag["eco_label_for_plot"] = ag["ECO_NAME_CODE"].astype(str).map(label_lookup)

        year_order = [str(y) for y in FOCAL_REPORTING_YEARS]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="eco_label_for_plot",
            bar_col="assessment_year",
            group_order=group_order,
            bar_order=year_order,
            out_png=chart_dir / f"State_ALL_ecological_ecoregion_grouped_years_{TIME_TAG}.png",
            title=f"Ecological State by ecoregion and reporting year — parametric ({TIME_TAG})",
            xlabel="Ecoregion",
            figsize=(max(14, len(group_order) * 0.55), 8),
            rotation=90,
        )


def plot_ecoregion_state_charts(summary_path: Path) -> None:
    """
    Create per-ecoregion charts from one existing ecoregion State summary CSV.
    """
    df = read_state_summary(summary_path)

    if df.empty:
        print(f"[charts] empty summary: {summary_path}")
        return

    eco_code = str(df["ECO_NAME_CODE"].dropna().iloc[0]) if "ECO_NAME_CODE" in df.columns and df["ECO_NAME_CODE"].notna().any() else "unknown"
    eco_name = str(df["ECO_NAME"].dropna().iloc[0]) if "ECO_NAME" in df.columns and df["ECO_NAME"].notna().any() else "unknown"

    eco_dir = summary_path.parents[1]
    chart_dir = eco_dir / "charts"
    mkdir(chart_dir)

    sub = df[
        (df["benchmark_system"].astype(str) == "ecological")
        & (df["method"].astype(str) == "parametric")
    ].copy()

    if not sub.empty:
        yr = aggregate_state_area(sub, ["assessment_year"])
        yr = yr.sort_values("assessment_year")
        yr["assessment_year"] = yr["assessment_year"].astype(str)

        plot_state_stacked_percent(
            yr,
            x_col="assessment_year",
            out_png=chart_dir / f"State_ECO_{eco_code}_ecological_parametric_by_year_{TIME_TAG}.png",
            title=f"State composition by year — ECO {eco_code}: {eco_name}",
            xlabel="Assessment year",
            figsize=(12, 7),
            rotation=90,
        )

    focal = sub[sub["assessment_year"].isin(FOCAL_REPORTING_YEARS)].copy()
    if not focal.empty:
        ag = aggregate_state_area(focal, ["assessment_year", "LULC_label"])
        ag["label"] = ag["assessment_year"].astype(str) + " | " + ag["LULC_label"].astype(str)

        plot_state_stacked_percent(
            ag,
            x_col="label",
            out_png=chart_dir / f"State_ECO_{eco_code}_ecological_parametric_LULC_focal_years_{TIME_TAG}.png",
            title=f"State by LULC and focal year — ECO {eco_code}: {eco_name}",
            xlabel="Assessment year | LULC group",
            figsize=(15, 8),
            rotation=90,
        )

    policy = df[
        (df["benchmark_system"].astype(str) == "policy")
        & (df["method"].astype(str) == "parametric")
    ].copy()

    possible_years = set(pd.to_numeric(policy["assessment_year"], errors="coerce").dropna().astype(int))
    latest = max([y for y in FOCAL_REPORTING_YEARS if y in possible_years], default=None)

    if latest is not None:
        p_latest = policy[policy["assessment_year"] == latest].copy()
        if not p_latest.empty:
            ag = aggregate_state_area(p_latest, ["ISO_A3"])
            ag = ag[ag["ISO_A3"].astype(str).str.len() > 0]
            ag = ag.sort_values("Poor_pct", ascending=False).head(TOP_N_CHART_ITEMS)

            if not ag.empty:
                plot_state_stacked_percent(
                    ag,
                    x_col="ISO_A3",
                    out_png=chart_dir / f"State_ECO_{eco_code}_policy_parametric_country_{latest}_{TIME_TAG}.png",
                    title=f"Policy-facing State by country — ECO {eco_code}, {latest}",
                    xlabel="Country",
                    figsize=(12, 7),
                    rotation=90,
                )

    sens = df[
        (df["benchmark_system"].astype(str) == "ecological")
        & (df["assessment_year"].isin(FOCAL_REPORTING_YEARS))
        & (df["method"].astype(str).isin(["parametric", "quantile"]))
    ].copy()

    if not sens.empty:
        ag = aggregate_state_area(sens, ["assessment_year", "method"])
        ag = ag.sort_values(["assessment_year", "method"])

        year_order = [str(y) for y in FOCAL_REPORTING_YEARS]
        method_order = ["parametric", "quantile"]

        plot_grouped_stacked_state_percent(
            ag,
            group_col="assessment_year",
            bar_col="method",
            group_order=year_order,
            bar_order=method_order,
            out_png=chart_dir / f"State_ECO_{eco_code}_parametric_vs_quantile_focal_years_{TIME_TAG}.png",
            title=f"Parametric vs quantile State sensitivity — ECO {eco_code}: {eco_name}",
            xlabel="Assessment year",
            figsize=(12, 7),
            rotation=0,
        )


def build_all_summary_from_ecoregion_files(summary_files: List[Path]) -> pd.DataFrame:
    frames = []

    for p in summary_files:
        try:
            frames.append(read_state_summary(p))
        except Exception as e:
            print(f"[warn] cannot read {p}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_chart_only_mode() -> None:
    """
    Chart-only runner. It reads existing CSV outputs and creates only missing
    PNG files. It does not run raster processing, benchmark construction,
    State classification, or summary creation.
    """
    print("\n" + "=" * 90)
    print("[charts only] creating missing State charts from existing summary CSVs")
    print("=" * 90)

    summary_files = find_ecoregion_summary_files()
    print(f"[charts only] ecoregion summary files found: {len(summary_files):,}")

    if not summary_files:
        print(f"[charts only] no ecoregion summaries found under: {ECOREGION_OUT_DIR}")
        return

    if MAKE_ALL_ECOREGIONS_CHARTS:
        all_path = expected_all_summary_path()

        if all_path.exists():
            print(f"[charts only] reading merged summary: {all_path}")
            all_df = read_state_summary(all_path)
        else:
            print("[charts only] merged ALL_ECOREGIONS summary not found; building in memory from ecoregion files.")
            all_df = build_all_summary_from_ecoregion_files(summary_files)

        if not all_df.empty:
            plot_all_ecoregions_state_charts(all_df)
        else:
            print("[charts only] no merged dataframe available for ALL_ECOREGIONS charts.")

    if MAKE_ECOREGION_CHARTS:
        t0 = time.perf_counter()

        for i, p in enumerate(summary_files, start=1):
            try:
                plot_ecoregion_state_charts(p)
            except Exception as e:
                print(f"[warning] chart creation failed for {p}: {e}")

            if i % 25 == 0 or i == len(summary_files):
                print(f"[charts only] {i:,}/{len(summary_files):,} ecoregions | {_fmt(time.perf_counter() - t0)}")


# =============================================================================
# 11C. CONSOLIDATED STATE DATA-PRODUCT RASTERS
# =============================================================================

def state_product_dir() -> Path:
    return DATA_PRODUCT_ROOT


def state_product_path(year: int, system: str, method: str) -> Path:
    """
    Africa-wide reusable State raster path.

    Codes:
        0 = NoData / unclassified
        1 = Poor
        2 = Fair
        3 = Good
    """
    return (
        state_product_dir()
        / f"State_{year}_{system}_{method}_{TIME_TAG}_EA250m.tif"
    )


def state_product_lookup_path() -> Path:
    return state_product_dir() / f"State_class_lookup_{TIME_TAG}.csv"


def state_product_manifest_path() -> Path:
    return state_product_dir() / f"State_data_product_manifest_{TIME_TAG}.csv"


def benchmark_csv_path(eco_code: int, system: str) -> Path:
    return (
        ECOREGION_OUT_DIR
        / f"ECO_{eco_code}"
        / "benchmarks"
        / f"{system}_benchmarks_{BENCHMARK_TIME_TAG}.csv"
    )


def read_benchmark_table_for_product(eco_code: int, system: str) -> pd.DataFrame:
    """Read an existing benchmark CSV generated by the main State workflow."""
    p = benchmark_csv_path(eco_code, system)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {system} benchmark CSV for ECO {eco_code}: {p}\n"
            "Run RUN_STAGE_2_LOOP_ECOREGIONS=True once, or make sure the "
            "existing benchmark CSVs are available before exporting data-product rasters."
        )
    return pd.read_csv(p)


def initialise_state_product_rasters(grid, product_specs: List[Tuple[int, str, str]]) -> Dict[Tuple[int, str, str], rasterio.io.DatasetWriter]:
    """
    Open output rasters for writing. Existing rasters are skipped unless forced.

    Returns only datasets that still need to be built.
    """
    mkdir(state_product_dir())
    transform, width, height = grid["transform"], grid["width"], grid["height"]
    prof = base_profile(transform, width, height, EA_CRS, "uint8", 0)

    open_outputs: Dict[Tuple[int, str, str], rasterio.io.DatasetWriter] = {}

    for year, system, method in product_specs:
        out = state_product_path(year, system, method)
        if not should_build_data_product(out):
            continue
        mkdir(out.parent)
        # Create and initialise to NoData/unclassified. This also makes partial
        # failures visible. Reopen in r+ mode so windows can be read/updated
        # without overwriting neighbouring ecoregions in overlapping bounds.
        zeros = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        with rasterio.open(out, "w", **prof) as init_dst:
            for win in iter_windows(width, height, TILE_SIZE):
                block = zeros[: int(win.height), : int(win.width)]
                init_dst.write(block, 1, window=win)
        dst = rasterio.open(out, "r+")
        open_outputs[(year, system, method)] = dst
        print(f"[open data product] {out}")

    return open_outputs


def close_state_product_rasters(open_outputs: Dict[Tuple[int, str, str], rasterio.io.DatasetWriter]) -> None:
    for dst in open_outputs.values():
        try:
            dst.close()
        except Exception:
            pass


def write_state_product_lookup_and_manifest(product_specs: List[Tuple[int, str, str]]) -> None:
    """Write small lookup tables that make the raster products self-explanatory."""
    lookup = pd.DataFrame([
        {"value": 0, "label": "NoData / unclassified"},
        {"value": 1, "label": "Poor"},
        {"value": 2, "label": "Fair"},
        {"value": 3, "label": "Good"},
    ])
    save_csv(lookup, state_product_lookup_path())

    rows = []
    for year, system, method in product_specs:
        rows.append({
            "dimension": "State",
            "assessment_year": year,
            "benchmark_system": system,
            "threshold_method": method,
            "time_tag": TIME_TAG,
            "benchmark_years": BENCHMARK_TAG,
            "benchmark_time_tag": BENCHMARK_TIME_TAG,
            "crs": EA_CRS,
            "resolution_m": EA_RES_M,
            "nodata": 0,
            "dtype": "uint8",
            "class_codes": "0=NoData/unclassified; 1=Poor; 2=Fair; 3=Good",
            "path": str(state_product_path(year, system, method)),
        })
    save_csv(pd.DataFrame(rows), state_product_manifest_path())


def build_state_data_product_rasters(ecos, country_lookup, grid, cache) -> None:
    """
    Build reusable Africa-wide consolidated State rasters.

    This function reuses:
      - cached annual NDVI rasters,
      - cached reporting-year LULC rasters,
      - cached ecoregion/country ID rasters,
      - benchmark CSVs already created by the State ecoregion loop.

    It does not rebuild the 5-year benchmark raster. That benchmark composite is
    already part of the State cache and is treated as an input product rather than
    rewritten here.
    """
    if not RUN_DATA_PRODUCT_STATE_RASTERS:
        return

    print("\n" + "=" * 90)
    print("[data products] Building consolidated Africa-wide State rasters")
    print("=" * 90)
    print(f"[data products] years: {DATA_PRODUCT_STATE_YEARS}")
    print(f"[data products] system/methods: {DATA_PRODUCT_STATE_SYSTEM_METHODS}")
    print(f"[data products] output: {state_product_dir()}")

    product_specs = [
        (int(year), str(system), str(method))
        for year in DATA_PRODUCT_STATE_YEARS
        for system, method in DATA_PRODUCT_STATE_SYSTEM_METHODS
    ]

    missing_years = [
        y for y in DATA_PRODUCT_STATE_YEARS
        if y not in cache.get("ndvi_assessment", {}) or y not in cache.get("lulc_reporting", {})
    ]
    if missing_years:
        raise FileNotFoundError(
            f"Missing cached NDVI or LULC reporting rasters for requested State product years: {missing_years}. "
            "Run RUN_STAGE_1_BUILD_CONTINENTAL_CACHE=True first, or check CACHE_DIR."
        )

    open_outputs = initialise_state_product_rasters(grid, product_specs)
    write_state_product_lookup_and_manifest(product_specs)

    if not open_outputs:
        print("[data products] all requested State products already exist; nothing to build.")
        return

    ecos_ea = ecos.to_crs(EA_CRS)
    width, height, transform = grid["width"], grid["height"], grid["transform"]

    try:
        for eco_i, (_, row) in enumerate(ecos_ea.iterrows(), 1):
            eco_id = int(row["ECO_ID"])
            eco_code = int(row[ECO_CODE_FIELD])
            eco_name = str(row[ECO_FIELD])
            print("\n" + "-" * 90)
            print(f"[data products] ECO {eco_i}/{len(ecos_ea)} | ECO_ID={eco_id} | ECO_NAME_CODE={eco_code} | {eco_name}")

            # Determine the window covering this ecoregion.
            minx, miny, maxx, maxy = row.geometry.bounds
            win = clip_window(
                window_from_bounds(minx, miny, maxx, maxy, transform=transform),
                width,
                height,
            )
            if win.width <= 0 or win.height <= 0:
                print("[data products skip] empty ecoregion window")
                continue

            eco_arr = read_window(cache["eco_id"], win)
            eco_mask = eco_arr == eco_id
            if not np.any(eco_mask):
                print("[data products skip] no pixels in rasterized ecoregion")
                continue

            ctry_arr = read_window(cache["country_id"], win)

            # Read benchmark CSVs once per ecoregion and make method-specific lookups.
            eco_df = read_benchmark_table_for_product(eco_code, "ecological")
            pol_df = read_benchmark_table_for_product(eco_code, "policy")
            lookups = {}
            for method in sorted(set(m for _, m in DATA_PRODUCT_STATE_SYSTEM_METHODS)):
                e_lookup = build_lookup(eco_df, method)
                p_lookup = build_lookup(pol_df, method) if not pol_df.empty else {}
                lookups[("ecological", method)] = (e_lookup, None)
                lookups[("policy", method)] = (p_lookup, e_lookup)

            for year in DATA_PRODUCT_STATE_YEARS:
                ndvi = read_window(cache["ndvi_assessment"][year], win, as_float=True, nodata_to_nan=True)
                lulc = read_window(cache["lulc_reporting"][year], win)

                for system, method in DATA_PRODUCT_STATE_SYSTEM_METHODS:
                    key = (int(year), str(system), str(method))
                    if key not in open_outputs:
                        continue

                    lookup, fallback = lookups[(system, method)]
                    if system == "policy" and not lookup:
                        print(f"[data products warning] empty policy lookup for ECO {eco_code}; skipping {year} {system} {method}")
                        continue

                    cond, _ = classify(
                        ndvi,
                        lulc,
                        eco_arr,
                        ctry_arr,
                        eco_id,
                        system,
                        lookup,
                        eco_lookup=fallback,
                    )

                    # Update only this ecoregion's pixels so neighbouring ecoregions in
                    # overlapping bounding windows are not overwritten by zeros.
                    dst = open_outputs[key]
                    current = dst.read(1, window=win)
                    current[eco_mask] = cond[eco_mask]
                    dst.write(current, 1, window=win)

    finally:
        close_state_product_rasters(open_outputs)

    print("\n[data products] consolidated State rasters finished.")


# =============================================================================
# 12. ECOREGION LOOP
# =============================================================================

def run_ecoregion_loop(ecos, country_lookup, grid, cache):
    print("\n" + "=" * 90)
    print("[5] Looping through ecoregions")
    print("=" * 90)
    ecos_ea = ecos.to_crs(EA_CRS)
    width, height, transform, pix_ha = grid["width"], grid["height"], grid["transform"], grid["pix_ha"]
    summary_paths = []
    for _, row in ecos_ea.iterrows():
        eco_id, eco_code, eco_name = int(row["ECO_ID"]), int(row[ECO_CODE_FIELD]), str(row[ECO_FIELD])
        eco_out = ECOREGION_OUT_DIR / f"ECO_{eco_code}"
        mkdir(eco_out)
        print("\n" + "-" * 90)
        print(f"[eco] ECO_ID={eco_id} | ECO_NAME_CODE={eco_code} | {eco_name}")
        minx, miny, maxx, maxy = row.geometry.bounds
        win = clip_window(window_from_bounds(minx, miny, maxx, maxy, transform=transform), width, height)
        if win.width <= 0 or win.height <= 0:
            print("[skip] empty window"); continue
        eco_arr = read_window(cache["eco_id"], win)
        ctry_arr = read_window(cache["country_id"], win)
        lm = read_window(cache["lulc_mode"], win)
        st = read_window(cache["lulc_stability"], win)
        nb = read_window(cache["ndvi_benchmark"], win, as_float=True, nodata_to_nan=True)
        eco_mask = eco_arr == eco_id
        if not np.any(eco_mask):
            print("[skip] no pixels in rasterized ecoregion"); continue
        u, counts = np.unique(ctry_arr[eco_mask & (ctry_arr > 0)], return_counts=True)
        countries = [int(cid) for cid, cnt in zip(u, counts) if cnt >= MIN_COUNTRY_INTERSECTION_PIXELS]
        print(f"[eco] countries: {countries}")
        arrays = {"eco_id": eco_arr, "country_id": ctry_arr, "lulc_mode": lm, "stability": st, "ndvi_benchmark": nb}
        eco_df, pol_df = build_benchmarks(arrays, eco_id, eco_code, eco_name, country_lookup, countries, eco_out)
        summaries, audits = [], []
        for method in ["parametric", "quantile"]:
            e_lookup = build_lookup(eco_df, method)
            p_lookup = build_lookup(pol_df, method) if not pol_df.empty else {}
            for year in ASSESSMENT_YEARS:
                if year not in cache["ndvi_assessment"] or year not in cache["lulc_reporting"]:
                    print(f"[skip] year {year} missing cache"); continue
                ndvi = read_window(cache["ndvi_assessment"][year], win, as_float=True, nodata_to_nan=True)
                lulc = read_window(cache["lulc_reporting"][year], win)
                for system, lookup, fallback in [("ecological", e_lookup, None), ("policy", p_lookup, e_lookup)]:
                    if system == "policy" and not lookup: continue
                    cond, audit = classify(ndvi, lulc, eco_arr, ctry_arr, eco_id, system, lookup, eco_lookup=fallback)
                    audit["assessment_year"] = year; audit["method"] = method; audit["time_tag"] = TIME_TAG
                    audits.append(audit)
                    summaries.append(summarize(cond, lulc, eco_arr, ctry_arr, eco_id, eco_code, eco_name, year, system, method, country_lookup, pix_ha))
                    if WRITE_ECOREGION_RASTERS:
                        out_transform = rasterio.windows.transform(win, transform)
                        write_raster(eco_out / "rasters" / f"State_{system}_{method}_{TIME_TAG}_{year}_ECO_{eco_code}.tif", cond, out_transform, EA_CRS, 0, "uint8")
        if summaries:
            df = pd.concat(summaries, ignore_index=True)
            path = eco_out / "summaries" / f"State_summary_ECO_{eco_code}_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv"
            save_csv(df, path)
            summary_paths.append(path)
        if audits:
            save_csv(pd.concat(audits, ignore_index=True), eco_out / "diagnostics" / f"threshold_usage_audit_ECO_{eco_code}_{BENCHMARK_TIME_TAG}.csv")
    merged = []
    for p in summary_paths:
        try: merged.append(pd.read_csv(p))
        except Exception as e: print(f"[warn] cannot read {p}: {e}")
    if merged:
        all_df = pd.concat(merged, ignore_index=True)
        save_csv(all_df, OUTPUT_ROOT / f"ALL_ECOREGIONS_State_summary_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv")
        diag = all_df.groupby(["focal_reporting_year", "benchmark_system", "method", "LULC_group", "LULC_label", "ECO_NAME_CODE", "ECO_NAME", "COUNTRY_ID", "ISO_A3"], dropna=False).agg(Poor_pct_mean=("Poor_pct", "mean"), Poor_pct_min=("Poor_pct", "min"), Poor_pct_max=("Poor_pct", "max"), Good_pct_mean=("Good_pct", "mean"), Good_pct_min=("Good_pct", "min"), Good_pct_max=("Good_pct", "max"), n_diagnostic_years=("assessment_year", "nunique")).reset_index()
        save_csv(diag, OUTPUT_ROOT / f"ALL_ECOREGIONS_State_diagnostic_context_summary_{BENCHMARK_TIME_TAG}_{ASSESSMENT_TAG}.csv")

# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    t0 = time.perf_counter()
    print("\n" + "#" * 90)
    print("STATE ASSESSMENT — CONSOLIDATED RASTER DATA-PRODUCT EXPORT")
    print("#" * 90)
    print(f"Start time: {_now()}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"TEST_MODE: {TEST_MODE}")
    print(f"TIME_TAG: {TIME_TAG}; months={COMPOSITE_MONTHS}")
    print(f"BENCHMARK_YEARS: {BENCHMARK_YEARS}")
    print(f"DIAGNOSTIC_YEAR_WINDOWS: {DIAGNOSTIC_YEAR_WINDOWS}")
    print(f"ASSESSMENT_YEARS: {ASSESSMENT_YEARS}")
    print(f"WRITE_ECOREGION_RASTERS: {WRITE_ECOREGION_RASTERS}")
    print(f"RUN_DATA_PRODUCT_STATE_RASTERS: {RUN_DATA_PRODUCT_STATE_RASTERS}")
    print(f"DATA_PRODUCT_STATE_YEARS: {DATA_PRODUCT_STATE_YEARS}")
    print("#" * 90)
    print(f"RUN_CHARTS_ONLY: {RUN_CHARTS_ONLY}")
    print(f"FORCE_REBUILD_CHARTS: {FORCE_REBUILD_CHARTS}")
    for p in [OUTPUT_ROOT, CACHE_DIR, ECOREGION_OUT_DIR, LOOKUP_DIR]: mkdir(p)
    config = pd.DataFrame([
        ("TEST_MODE", TEST_MODE), ("TEST_ECO_NAME_VALUES", TEST_ECO_NAME_VALUES),
        ("BENCHMARK_YEARS", BENCHMARK_YEARS), ("FOCAL_REPORTING_YEARS", FOCAL_REPORTING_YEARS),
        ("DIAGNOSTIC_YEAR_WINDOWS", DIAGNOSTIC_YEAR_WINDOWS), ("ASSESSMENT_YEARS", ASSESSMENT_YEARS),
        ("COMPOSITE_MONTHS", COMPOSITE_MONTHS), ("TIME_TAG", TIME_TAG),
        ("BENCHMARK_TIME_TAG", BENCHMARK_TIME_TAG), ("MIN_VALID_BENCHMARK", MIN_VALID_BENCHMARK),
        ("MIN_VALID_ASSESSMENT", MIN_VALID_ASSESSMENT), ("WRITE_ECOREGION_RASTERS", WRITE_ECOREGION_RASTERS),
        ("RUN_CHARTS_ONLY", RUN_CHARTS_ONLY), ("FORCE_REBUILD_CHARTS", FORCE_REBUILD_CHARTS),
        ("WRITE_CHARTS", WRITE_CHARTS), ("SAVE_HISTOGRAMS", SAVE_HISTOGRAMS),
        ("RUN_DATA_PRODUCT_STATE_RASTERS", RUN_DATA_PRODUCT_STATE_RASTERS),
        ("DATA_PRODUCT_STATE_YEARS", DATA_PRODUCT_STATE_YEARS),
        ("DATA_PRODUCT_STATE_SYSTEM_METHODS", DATA_PRODUCT_STATE_SYSTEM_METHODS),
        ("REUSE_EXISTING_DATA_PRODUCTS", REUSE_EXISTING_DATA_PRODUCTS),
        ("FORCE_REBUILD_DATA_PRODUCTS", FORCE_REBUILD_DATA_PRODUCTS),
        ("DATA_PRODUCT_ROOT", DATA_PRODUCT_ROOT),
        ("EA_CRS", EA_CRS), ("EA_RES_M", EA_RES_M)
    ], columns=["setting", "value"])
    save_csv(config, OUTPUT_ROOT / f"run_configuration_{TIME_TAG}.csv")

    if RUN_CHARTS_ONLY and not RUN_DATA_PRODUCT_STATE_RASTERS:
        run_chart_only_mode()
        print("\n" + "#" * 90)
        print("STATE CHART-ONLY RUN FINISHED")
        print(f"End time: {_now()} | elapsed {_fmt(time.perf_counter() - t0)}")
        print(f"Outputs: {OUTPUT_ROOT}")
        print("#" * 90)
        return

    ecos, countries, extent_gdf, eco_lookup, country_lookup = prepare_vectors()
    grid = prepare_grid(extent_gdf)
    if RUN_STAGE_1_BUILD_CONTINENTAL_CACHE:
        cache = build_continental_cache(ecos, countries, grid)
    else:
        reporting = {y: CACHE_DIR / f"LULC_major_reporting_{y}_EA250m.tif" for y in ASSESSMENT_YEARS if (CACHE_DIR / f"LULC_major_reporting_{y}_EA250m.tif").exists()}
        ndvi_ass = {y: CACHE_DIR / f"NDVI_{NDVI_COMPOSITE_METHOD}_assessment_{TIME_TAG}_{y}_EA250m.tif" for y in ASSESSMENT_YEARS if (CACHE_DIR / f"NDVI_{NDVI_COMPOSITE_METHOD}_assessment_{TIME_TAG}_{y}_EA250m.tif").exists()}
        cache = {"eco_id": CACHE_DIR / "Ecoregion_ID_EA250m.tif", "country_id": CACHE_DIR / "Country_ID_EA250m.tif", "lulc_mode": CACHE_DIR / f"LULC_major_mode_{BENCHMARK_TAG}_EA250m.tif", "lulc_stability": CACHE_DIR / f"LULC_major_mode_count_{BENCHMARK_TAG}_EA250m.tif", "lulc_reporting": reporting, "ndvi_benchmark": CACHE_DIR / f"NDVI_{NDVI_COMPOSITE_METHOD}_benchmark_{BENCHMARK_TIME_TAG}_EA250m.tif", "ndvi_assessment": ndvi_ass}
    if RUN_STAGE_2_LOOP_ECOREGIONS:
        run_ecoregion_loop(ecos, country_lookup, grid, cache)

    if RUN_DATA_PRODUCT_STATE_RASTERS:
        build_state_data_product_rasters(ecos, country_lookup, grid, cache)

    print("\n" + "#" * 90)
    print("STATE ASSESSMENT ECOREGION LOOP FINISHED")
    print(f"End time: {_now()} | elapsed {_fmt(time.perf_counter() - t0)}")
    print(f"Outputs: {OUTPUT_ROOT}")
    print("#" * 90)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
