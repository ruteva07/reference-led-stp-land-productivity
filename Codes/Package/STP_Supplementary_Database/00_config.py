from pathlib import Path

PROJECT_ROOT = Path(r"C:\PATH\TO\Paper\Codes\Africa\STP_Supplementary_Database")
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

RUN_PREPARE_ZONES = False        

RUN_STATE = False
RUN_TREND = False
RUN_PERFORMANCE = False
RUN_JOINT_STP = True

REBUILD_ZONE_RASTERS = True

CREATE_FIGURES = True
SHOW_FIGURES = False
SAVE_FIGURES = True

SAVE_PARQUET = True
SAVE_EXCEL_SUMMARY = True

RUN_COUNTRY = True
RUN_ECOREGION = False
RUN_LULC = True
RUN_COUNTRY_LULC = True
RUN_ECOREGION_LULC = False

COUNTRY_VECTOR = Path(r"C:\PATH\TO\Paper\Codes\Africa\Polygons\Africa_49States_Undisputed.gpkg")
COUNTRY_LAYER = None 
COUNTRY_ID_FIELD = "ISO_A3"
COUNTRY_NAME_FIELD = "NAM_0"
ECOREGION_VECTOR = Path(r"C:\PATH\TO\Paper\Codes\Africa\Polygons\Afr_Ecoregions2017_FG_True.gpkg")
ECOREGION_LAYER = None
ECOREGION_ID_FIELD = "ECO_NAME_CODE"
ECOREGION_NAME_FIELD = "ECO_NAME"
BIOME_ID_FIELD = "BIOME_NUM"
BIOME_NAME_FIELD = "BIOME_NAME"
REALM_FIELD = "REALM"

STATE_RASTERS = {
"policy_quantile": Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\DATA_PRODUCTS\STATE\Annual\State_2022_policy_quantile_Annual_EA250m.tif"),
"policy_parametric": Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\DATA_PRODUCTS\STATE\Annual\State_2022_policy_parametric_Annual_EA250m.tif"),
"ecological_quantile": Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\DATA_PRODUCTS\STATE\Annual\State_2022_ecological_quantile_Annual_EA250m.tif"),
"ecological_parametric": Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\DATA_PRODUCTS\STATE\Annual\State_2022_ecological_parametric_Annual_EA250m.tif"),}
PRIMARY_STATE_FORMULATION = "ecological_parametric"

TREND_DIRECTION_RASTERS = {
"earlier_2001_2011": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\DATA_PRODUCTS\TREND_DIRECTION\Annual\TrendNDVI_Trend_2001_2011_TrendDirection3Class_PMKNW_Annual_EA250m.tif"),
"recent_2012_2022": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\DATA_PRODUCTS\TREND_DIRECTION\Annual\TrendNDVI_Trend_2012_2022_TrendDirection3Class_PMKNW_Annual_EA250m.tif"),
"full_2001_2022": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\DATA_PRODUCTS\TREND_DIRECTION\Annual\TrendNDVI_Trend_2001_2022_TrendDirection3Class_PMKNW_Annual_EA250m.tif"),}
TREND_FUNNEL_RASTERS = {
"potential": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\Trend_2001_2022\rasters\TrendNDVI_Trend_2001_2022_DegrPotential_Annual_EA250m.tif"),
"ols": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\Trend_2001_2022\rasters\TrendNDVI_Trend_2001_2022_DegrP_neg_Annual_EA250m.tif"),
"newey_west": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\Trend_2001_2022\rasters\TrendNDVI_Trend_2001_2022_DegrNW_neg_Annual_EA250m.tif"),
"mann_kendall": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\Trend_2001_2022\rasters\TrendNDVI_Trend_2001_2022_DegrMK_neg_Annual_EA250m.tif"),
"strict_joint": Path(r"C:\PATH\TO\Paper\Codes\Africa\TREND_NDVI_Africa_From_StateCache\Annual\Trend_2001_2022\rasters\TrendNDVI_Trend_2001_2022_DegrPMKNW_neg_Annual_EA250m.tif"),}
LULC_TRANSITION_COUNT_RASTER = None

PERFORMANCE_RASTERS = {
"performance_2011": Path(r"C:\PATH\TO\Paper\Codes\Africa\PERFORMANCE_NDVI_Africa_Annual\Annual\rasters\PerfNDVI_Performance_2011_vs_Baseline_Performance_class_5bin_Annual_EA250m.tif"),
"performance_2022": Path(r"C:\PATH\TO\Paper\Codes\Africa\PERFORMANCE_NDVI_Africa_Annual\Annual\rasters\PerfNDVI_Performance_2022_vs_Baseline_Performance_class_5bin_Annual_EA250m.tif"),}

LULC_RASTERS = {
2001: Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\_continental_cache\Annual\LULC_major_reporting_2001_EA250m.tif"),
2011: Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\_continental_cache\Annual\LULC_major_reporting_2011_EA250m.tif"),
2022: Path(r"C:\PATH\TO\Paper\Codes\Africa\STATE_NDVI_Africa_Loop_Annual\_continental_cache\Annual\LULC_major_reporting_2022_EA250m.tif"),}

LULC_GROUP_LABELS = {1:"Cropland",2:"Cropland mosaic",3:"Natural/cropland mosaic",4:"Tree cover",5:"Shrubland",6:"Grassland",7:"Sparse/bare",8:"Wetland/flooded vegetation",9:"Urban",10:"Water/snow/ice",11:"Natural vegetation mosaic"}
INCLUDED_LULC_CODES = [1,2,3,4,5,6,8,9,11]
EXCLUDED_LULC_CODES = [0,7,10]
STATE_CLASS_LOOKUP = {1:"Poor",2:"Fair",3:"Good"}
TREND_CLASS_LOOKUP = {1:"Confirmed decline",2:"Neutral/unconfirmed",3:"Confirmed improvement"}
PERFORMANCE_CLASS_LOOKUP = {1:"Strong loss",2:"Moderate loss",3:"Stable",4:"Moderate gain",5:"Strong gain"}

MIN_VALID_PIXELS = 400
MIN_VALID_AREA_HA = 2500.0
PREFERRED_VALID_PIXELS = 1600
PREFERRED_VALID_AREA_HA = 10000.0
MIN_JACCARD_UNION_PIXELS = 100
MIN_CLASS_PIXELS_EACH_MAP = 25
MIN_KAPPA_VALID_PIXELS = 400
MIN_TREND_FUNNEL_DENOMINATOR_PIXELS = 100
MIN_DIAGNOSTIC_GROUP_PIXELS = 25
FIGURE_DPI = 300
FIGURE_FORMATS = ["png","pdf"]
TOP_N_UNITS_IN_FIGURES = 30

REUSABLE_ZONE_DIR = OUTPUT_ROOT / "reusable_zones"
STATE_OUTPUT_DIR = OUTPUT_ROOT / "state"
TREND_OUTPUT_DIR = OUTPUT_ROOT / "trend"
PERFORMANCE_OUTPUT_DIR = OUTPUT_ROOT / "performance"
JOINT_OUTPUT_DIR = OUTPUT_ROOT / "joint_stp"
FIGURE_OUTPUT_DIR = OUTPUT_ROOT / "figures"
TABLE_OUTPUT_DIR = OUTPUT_ROOT / "tables"
LOG_OUTPUT_DIR = OUTPUT_ROOT / "logs"
QC_OUTPUT_DIR = OUTPUT_ROOT / "quality_control"
for d in [OUTPUT_ROOT,REUSABLE_ZONE_DIR,STATE_OUTPUT_DIR,TREND_OUTPUT_DIR,PERFORMANCE_OUTPUT_DIR,JOINT_OUTPUT_DIR,FIGURE_OUTPUT_DIR,TABLE_OUTPUT_DIR,LOG_OUTPUT_DIR,QC_OUTPUT_DIR]: d.mkdir(parents=True,exist_ok=True)
