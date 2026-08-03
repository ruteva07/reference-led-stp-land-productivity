from pathlib import Path
import importlib.util,sys
from datetime import datetime
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from modules.core import logger,validate_alignment
from modules.workflows import prepare_zones,run_state,run_trend,run_performance,run_joint
spec=importlib.util.spec_from_file_location('cfg',ROOT/'00_config.py');cfg=importlib.util.module_from_spec(spec);spec.loader.exec_module(cfg)
log=logger(cfg.LOG_OUTPUT_DIR/f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
paths=list(cfg.LULC_RASTERS.values())
if cfg.RUN_STATE or cfg.RUN_JOINT_STP:paths+=list(cfg.STATE_RASTERS.values())
if cfg.RUN_TREND or cfg.RUN_JOINT_STP:paths+=list(cfg.TREND_DIRECTION_RASTERS.values())
if cfg.RUN_TREND:paths+=list(cfg.TREND_FUNNEL_RASTERS.values())
if cfg.RUN_PERFORMANCE or cfg.RUN_JOINT_STP:paths+=list(cfg.PERFORMANCE_RASTERS.values())
validate_alignment(dict.fromkeys(paths),cfg.STATE_RASTERS[cfg.PRIMARY_STATE_FORMULATION],cfg.QC_OUTPUT_DIR/'raster_alignment.csv');log.info('Alignment passed')
if cfg.RUN_PREPARE_ZONES:prepare_zones(cfg,log)
needed=[cfg.REUSABLE_ZONE_DIR/'country_id_EA250m.tif',cfg.REUSABLE_ZONE_DIR/'ecoregion_id_EA250m.tif']
if any(not p.exists() for p in needed):raise FileNotFoundError('Run RUN_PREPARE_ZONES=True first')
if cfg.RUN_STATE:run_state(cfg,log)
if cfg.RUN_TREND:run_trend(cfg,log)
if cfg.RUN_PERFORMANCE:run_performance(cfg,log)
if cfg.RUN_JOINT_STP:run_joint(cfg,log)
log.info('Workflow complete')
