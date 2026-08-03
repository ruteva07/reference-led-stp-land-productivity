# -*- coding: utf-8 -*-
"""
TREND — annual MODIS NDVI trend assessment for Africa
======================================================

Author
------
Evariste Rutebuka

Purpose
-------
Core Trend workflow used to estimate pixel-level temporal change in annual
MODIS NDVI across terrestrial Africa.

This script reuses the harmonised annual NDVI and land-cover rasters generated
by the State workflow. It does not rebuild annual MODIS composites from the
original 16-day observations and it does not use State benchmark thresholds.

Analytical windows
------------------
Three fixed Trend periods are evaluated independently:

    Earlier period : 2001–2011  (11 annual observations when complete)
    Recent period  : 2012–2022  (11 annual observations when complete)
    Full period    : 2001–2022  (22 annual observations when complete)

For each pixel, only available annual NDVI observations are used. A Trend is
estimated only when the pixel meets the configured minimum-observation rule:

    max(8 years, ceil(0.75 × window length))

which gives:
    9 of 11 years for 2001–2011,
    9 of 11 years for 2012–2022, and
    17 of 22 years for 2001–2022.

Trend statistics
----------------
For every eligible pixel and analytical window the script calculates:

    - ordinary least-squares (OLS) slope in physical NDVI units per year;
    - OLS intercept;
    - Pearson correlation coefficient (r);
    - coefficient of determination (R²);
    - finite-sample OLS significance using Student's t distribution with
      df = n - 2;
    - Newey–West heteroskedasticity/autocorrelation-consistent slope statistic
      using the configured maximum lag (3) and an asymptotic two-sided normal
      p-value;
    - Mann–Kendall S-derived z statistic, two-sided normal-approximation
      p-value and Kendall tau; and
    - number of valid annual NDVI observations.


Decline / degradation-support logic
-----------------------------------
The directional NDVI decline threshold is:

    slope <= -0.0002 NDVI units per year

and statistical support is evaluated at:

    p < 0.10.

Several diagnostic masks are produced. The strict negative Trend mask used by
the current workflow is:

    DegrPMKNW_neg

which requires all of the following:

    1. OLS p < 0.10 and slope <= -0.0002;
    2. Newey–West p < 0.10 and slope <= -0.0002; and
    3. Mann–Kendall p < 0.10 with negative tau.

OLS inference is based only on the finite-sample Student t distribution.
No separate normal-approximation score is calculated or used for OLS.
The Mann–Kendall z statistic remains part of the Mann–Kendall test itself.

Annual NDVI screening
---------------------
The annual NDVI rasters are read from the State cache. This Trend script masks:

    - the raster's declared NoData value;
    - the configured NDVI NoData value (-9999); and
    - annual NDVI values outside the configured physical range (-0.2 to 1.0).


LULC transition diagnostic
--------------------------
Annual broad land-cover classes from the State cache are used only as
contextual/post-Trend diagnostics and for the reporting-domain summaries.
They do not determine the NDVI slope.

For each Trend window the script derives the modal LULC class and a
transition-history diagnostic. For a complete 2001–2022 history, the
interpretive classes approximate:

    0 transitions   -> strongest evidence that decline occurred within a
                       persistent broad LULC class
    1–2 transitions -> likely genuine within-class decline
    3–5 transitions -> moderate transition; interpret with caution
    >5 transitions  -> likely transition-affected

Thresholds are scaled to the number of valid consecutive LULC year-pairs for
shorter or incomplete histories.

Summary denominator
-------------------
To preserve the production results, degradation-area summaries use the
selected analysis LULC domain as the denominator:

    valid ecoregion pixels whose window-modal LULC belongs to
    ANALYSIS_LULC_GROUPS.

The denominator is not additionally restricted by the NDVI_valid_n raster.
Consequently, a binary degradation mask value of 0 means "not classified as
degraded by that mask" and can include pixels without sufficient Trend support.
This behaviour is retained deliberately because changing it would alter the
reported area percentages.

Outputs
-------
For each Trend window the workflow writes:

    - continuous Trend-statistic rasters;
    - valid-observation-count raster;
    - binary degradation/support masks;
    - LULC transition/context diagnostics;
    - an output manifest;
    - optional grouped summary tables;
    - optional charts; and
    - strict improvement masks plus a three-class Trend-direction product:
          1 = strict confirmed degradation
          2 = neutral / no strict confirmed directional Trend
          3 = strict confirmed improvement.

All outputs retain the common State-cache equal-area grid.

"""
#%%
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import ExitStack
from datetime import datetime, timedelta
import math
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window

# =============================================================================
# 0. USER SETTINGS
# =============================================================================

# ARCHIVE: edit only this project root for another machine.
MAIN_FOLDER = Path(r"C:\PATH\TO\Paper\Codes\Africa")
TIME_TAG = "Annual"
STATE_CACHE_DIR = (
    MAIN_FOLDER
    / "STATE_NDVI_Africa_Loop_Annual" 
    / "_continental_cache"
    / TIME_TAG
)
LOOKUP_DIR = MAIN_FOLDER / "STATE_NDVI_Africa_Loop_Annual" / "lookups"
TREND_OUT_ROOT = MAIN_FOLDER / "TREND_NDVI_Africa_From_StateCache" / TIME_TAG
TREND_OUT_ROOT.mkdir(parents=True, exist_ok=True)

PREFIX = "TrendNDVI"

RUN_BUILD_TREND_RASTERS = True   # archive default: build/reuse continental Trend rasters
RUN_SUMMARY_TABLES = True
FORCE_REBUILD_TREND_RASTERS = False # set True to rebuild trend rasters even if they already exist
REUSE_EXISTING_TREND_RASTERS = True # set True to skip trend raster rebuild if they already exist

# Chart-only mode reads existing summary tables and creates missing PNG charts.
# It does not validate cache inputs, rebuild rasters, or rerun summaries.
RUN_CHARTS_ONLY = False
RUN_CHARTS = True
FORCE_REBUILD_CHARTS = True

# Main mask used for ranking charts.
# This is the strictest degradation mask in the current workflow.
CHART_MAIN_MASK = "DegrPMKNW_neg"

# Sort country/ecoregion ranking charts by this trend window.
CHART_SORT_WINDOW = "Trend_2001_2022"

# Keep figures readable. Use None to show all countries/ecoregions.
TOP_N_COUNTRIES = 49
TOP_N_ECOREGIONS = 30

# Optional: create country-specific detail charts.
# Leave COUNTRY_DETAIL_CODES empty to build charts for every country found.
RUN_COUNTRY_DETAIL_CHARTS = True
COUNTRY_DETAIL_CODES: List[str] = []
TOP_N_ECOREGIONS_PER_COUNTRY = 50

# -------------------------------------------------------------------------
# Integrated Trend-direction data products
# -------------------------------------------------------------------------
# These products mirror the strict negative Trend rule with a positive rule
# and combine both directions into one reusable three-class raster.
RUN_TREND_DIRECTION_PRODUCTS = True
WRITE_COMPONENT_IMPROVEMENT_MASKS = True
REUSE_EXISTING_DIRECTION_PRODUCTS = True
FORCE_REBUILD_DIRECTION_PRODUCTS = False

# =============================================================================
# 1. TREND WINDOWS
# =============================================================================

TREND_WINDOWS: Dict[str, List[int]] = {
    "Trend_2001_2011": list(range(2001, 2012)),
    "Trend_2012_2022": list(range(2012, 2023)),
    "Trend_2001_2022": list(range(2001, 2023)),
}

# =============================================================================
# 2. STATISTICAL SETTINGS
# =============================================================================

NDVI_NODATA = -9999.0
VALID_NDVI_RANGE = (-0.2, 1.0)

MIN_VALID_OBS_FRACTION = 0.75
MIN_VALID_OBS_ABSOLUTE = 8

SLOPE_TH_PHYSICAL = -0.0002
SLOPE_TH_PHYSICAL_POS = abs(SLOPE_TH_PHYSICAL)
R2_THRESHOLD = 0.20
P_90 = 0.10
NW_MAX_LAG = 3

# Legacy stability-proportion settings retained for provenance. The active
# LULC validation class uses transition-count/rate thresholds instead.
STRICT_STABILITY_PROP = 0.90
HIGH_STABILITY_PROP = 0.80
MODERATE_STABILITY_PROP = 0.60

# =============================================================================
# 3. PROCESSING SETTINGS
# =============================================================================

BLOCK_SIZE = 512
COMPRESS = "ZSTD"
ZSTD_LEVEL = 12
TILE_SIZE = 512
FLOAT_NODATA = -9999.0
UINT8_NODATA = 0
UINT16_NODATA = 0

# =============================================================================
# 4. LULC SETTINGS
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

EXCLUDE_LULC_GROUP_LABELS = ["Sparse/bare", "Water/snow/ice"]
EXCLUDE_LULC_GROUPS = [k for k, v in LULC_GROUP_LABELS.items() if v in EXCLUDE_LULC_GROUP_LABELS]
ANALYSIS_LULC_GROUPS = [k for k in LULC_GROUP_LABELS if k not in EXCLUDE_LULC_GROUPS]


GENUINITY_LABELS = {
    1: "1 | 0 transitions — strongest evidence of genuine degradation",
    2: "2 | 1–2 transitions — likely genuine degradation",
    3: "3 | 3–5 transitions — moderate transition, interpret with caution",
    4: "4 | >5 transitions — likely transition-affected degradation",
}

SHORT_GENUINITY_LABELS = {
    1: "0 transitions\n(strongest evidence)",
    2: "1–2 transitions\n(likely genuine)",
    3: "3–5 transitions\n(moderate / check)",
    4: ">5 transitions\n(transition-affected)",
}
LULC_STABILITY_LABELS = SHORT_GENUINITY_LABELS.copy()
LOW_TRANSITION_RATE = 2 / 21
MODERATE_TRANSITION_RATE = 5 / 21
# =============================================================================
# 5. OUTPUT MASK DEFINITIONS
# =============================================================================

MASK_LABELS = {
    "DegrPotential": "Potential degradation by negative slope threshold",
    "ConfirmP_all": "Confirmed trend by OLS p-value, any direction",
    "DegrP_neg": "Negative degradation trend confirmed by OLS p-value",
    "ConfirmNW_all": "Confirmed trend by Newey-West p-value, any direction",
    "DegrNW_neg": "Negative degradation trend confirmed by Newey-West p-value",
    "ConfirmMK_all": "Confirmed monotonic trend by Mann-Kendall, any direction",
    "DegrMK_neg": "Negative degradation trend confirmed by Mann-Kendall",
    "ConfirmPMK_all": "Confirmed trend by OLS p-value and Mann-Kendall, any direction",
    "DegrPMK_neg": "Negative degradation trend confirmed by OLS p-value and Mann-Kendall",
    "ConfirmPNW_all": "Confirmed trend by OLS p-value and Newey-West, any direction",
    "DegrPNW_neg": "Negative degradation trend confirmed by OLS p-value and Newey-West",
    "ConfirmMKNW_all": "Confirmed trend by Mann-Kendall and Newey-West, any direction",
    "DegrMKNW_neg": "Negative degradation trend confirmed by Mann-Kendall and Newey-West",
    "ConfirmPMKNW_all": "Strict confirmed trend by OLS p, MK and Newey-West, any direction",
    "DegrPMKNW_neg": "Strict negative degradation trend confirmed by OLS p, MK and Newey-West",
}
SUMMARY_MASKS = ["DegrPotential", "DegrP_neg", "DegrNW_neg", "DegrMK_neg", "DegrPMKNW_neg"]

# =============================================================================
# 6. PATH HELPERS
# =============================================================================

def _fmt(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ndvi_path(year: int) -> Path:
    return STATE_CACHE_DIR / f"NDVI_mean_assessment_{TIME_TAG}_{year}_EA250m.tif"


def lulc_path(year: int) -> Path:
    return STATE_CACHE_DIR / f"LULC_major_reporting_{year}_EA250m.tif"


def ecoregion_id_path() -> Path:
    return STATE_CACHE_DIR / "Ecoregion_ID_EA250m.tif"


def country_id_path() -> Path:
    return STATE_CACHE_DIR / "Country_ID_EA250m.tif"


def get_window_dir(window_name: str) -> Path:
    return TREND_OUT_ROOT / window_name


def get_raster_dir(window_name: str) -> Path:
    return get_window_dir(window_name) / "rasters"


def get_summary_dir(window_name: str) -> Path:
    return get_window_dir(window_name) / "summaries"


def get_diagnostic_dir(window_name: str) -> Path:
    return get_window_dir(window_name) / "diagnostics"


def get_chart_dir() -> Path:
    return TREND_OUT_ROOT / "charts"


def trend_raster_path(window_name: str, metric: str) -> Path:
    return get_raster_dir(window_name) / f"{PREFIX}_{window_name}_{metric}_{TIME_TAG}_EA250m.tif"


def mask_raster_path(window_name: str, mask_name: str) -> Path:
    return get_raster_dir(window_name) / f"{PREFIX}_{window_name}_{mask_name}_{TIME_TAG}_EA250m.tif"


def lulc_stability_path(window_name: str, metric: str) -> Path:
    return get_raster_dir(window_name) / f"{PREFIX}_{window_name}_{metric}_{TIME_TAG}_EA250m.tif"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")

# =============================================================================
# 7. RASTER HELPERS
# =============================================================================

def iter_windows(width: int, height: int, block_size: int = BLOCK_SIZE):
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def output_profile_like(src_profile: dict, dtype: str, nodata) -> dict:
    prof = src_profile.copy()
    prof.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress=COMPRESS,
        zstd_level=ZSTD_LEVEL,
        tiled=True,
        blockxsize=TILE_SIZE,
        blockysize=TILE_SIZE,
        BIGTIFF="IF_SAFER",
    )
    return prof


def write_float_window(dst, arr: np.ndarray, win: Window) -> None:
    out = np.where(np.isfinite(arr), arr, FLOAT_NODATA).astype(np.float32)
    dst.write(out, 1, window=win)


def write_uint8_window(dst, arr: np.ndarray, win: Window) -> None:
    dst.write(arr.astype(np.uint8), 1, window=win)


def write_uint16_window(dst, arr: np.ndarray, win: Window) -> None:
    dst.write(arr.astype(np.uint16), 1, window=win)


def validate_cache_inputs() -> None:
    required = [ecoregion_id_path(), country_id_path()]
    for years in TREND_WINDOWS.values():
        for y in years:
            required.append(ndvi_path(y))
            required.append(lulc_path(y))
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[missing inputs]")
        for p in missing[:50]:
            print(f"  - {p}")
        if len(missing) > 50:
            print(f"  ... and {len(missing)-50} more")
        raise FileNotFoundError("Some required State-cache rasters are missing.")
    print("[cache] all required input rasters found.")


def validate_alignment(paths: List[Path]) -> dict:
    with rasterio.open(paths[0]) as ref:
        ref_profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)
        ref_crs = ref.crs
        ref_transform = ref.transform
    for p in paths[1:]:
        with rasterio.open(p) as src:
            if (src.height, src.width) != ref_shape:
                raise ValueError(f"Shape mismatch: {p}")
            if src.crs != ref_crs:
                raise ValueError(f"CRS mismatch: {p}")
            if src.transform != ref_transform:
                raise ValueError(f"Transform mismatch: {p}")
    return ref_profile

# =============================================================================
# 8. STATISTICAL HELPERS
# =============================================================================

try:
    from scipy.special import erfc as scipy_erfc
    SCIPY_ERFC_AVAILABLE = True
except Exception:
    scipy_erfc = None
    SCIPY_ERFC_AVAILABLE = False

from scipy.stats import t as scipy_student_t


def two_sided_normal_p_array(statistic: np.ndarray) -> np.ndarray:
    """
    Convert a standard-normal test statistic to a two-sided p-value.

    Used for the asymptotic Newey–West statistic and the Mann–Kendall
    normal approximation. It is not used for OLS, whose p-value is obtained
    from the finite-sample Student t distribution.
    """
    stat_abs = np.abs(statistic)
    out = np.full_like(stat_abs, np.nan, dtype=np.float64)
    finite = np.isfinite(stat_abs)

    if SCIPY_ERFC_AVAILABLE:
        out[finite] = scipy_erfc(stat_abs[finite] / np.sqrt(2.0))
        return out

    from math import erf, sqrt

    def _p(v):
        return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(float(v)) / sqrt(2.0))))

    out[finite] = np.vectorize(_p)(stat_abs[finite])
    return out


def min_valid_obs_for_window(n_years: int) -> int:
    return max(MIN_VALID_OBS_ABSOLUTE, int(math.ceil(n_years * MIN_VALID_OBS_FRACTION)))


def linear_stats_nan(y_stack: np.ndarray, years: List[int], min_obs: int) -> Dict[str, np.ndarray]:
    N, H, W = y_stack.shape
    P = H * W
    Y = y_stack.reshape(N, P).astype(np.float64)
    valid = np.isfinite(Y)
    n = valid.sum(axis=0).astype(np.float64)
    ok = n >= min_obs

    t = np.array(years, dtype=np.float64)
    t = t - t[0]

    Y0 = np.where(valid, Y, 0.0)
    sums_y = Y0.sum(axis=0)
    ybar = np.full(P, np.nan, dtype=np.float64)
    ybar[ok] = sums_y[ok] / n[ok]

    Tv = valid * t[:, None]
    sum_t = Tv.sum(axis=0)
    tbar = np.full(P, np.nan, dtype=np.float64)
    tbar[ok] = sum_t[ok] / n[ok]

    Yc = np.zeros_like(Y, dtype=np.float64)
    Tc = np.zeros_like(Y, dtype=np.float64)
    Yc[:, ok] = np.where(valid[:, ok], Y[:, ok] - ybar[ok], 0.0)
    Tc[:, ok] = np.where(valid[:, ok], t[:, None] - tbar[ok], 0.0)

    cov_num = (Tc * Yc).sum(axis=0)
    var_t = (Tc ** 2).sum(axis=0)
    var_y = (Yc ** 2).sum(axis=0)

    slope = np.full(P, np.nan, dtype=np.float64)
    intercept = np.full(P, np.nan, dtype=np.float64)
    slope_ok = ok & (var_t > 0)
    slope[slope_ok] = cov_num[slope_ok] / var_t[slope_ok]
    intercept[slope_ok] = ybar[slope_ok] - slope[slope_ok] * tbar[slope_ok]

    r = np.full(P, np.nan, dtype=np.float64)
    denom = np.sqrt(var_t * var_y)
    r_ok = slope_ok & (denom > 0)
    r[r_ok] = cov_num[r_ok] / denom[r_ok]
    r = np.clip(r, -1.0, 1.0)
    r2 = r ** 2

    # Finite-sample OLS inference: t_OLS follows Student's t with df = n - 2.
    # This is the authoritative OLS p-value used by the production Trend masks.
    ols_t = np.full(P, np.nan, dtype=np.float64)
    ols_t_ok = ok & np.isfinite(r2) & (n > 2) & ((1.0 - r2) > 0)
    ols_t[ols_t_ok] = r[ols_t_ok] * np.sqrt(n[ols_t_ok] - 2.0) / np.sqrt(1.0 - r2[ols_t_ok])

    p = np.full(P, np.nan, dtype=np.float64)
    p[ols_t_ok] = 2.0 * scipy_student_t.sf(np.abs(ols_t[ols_t_ok]), df=n[ols_t_ok] - 2.0)

    # Newey-West HAC-corrected t statistic for slope.
    beta0_2d = intercept[None, :]
    beta1_2d = slope[None, :]
    t_col = t[:, None]
    y_hat = beta0_2d + beta1_2d * t_col
    residuals = np.where(valid, Y - y_hat, 0.0)

    X = Tc
    sum_x2 = (X ** 2).sum(axis=0)
    Lmax = min(NW_MAX_LAG, N - 1) if N > 1 else 0
    S_sum = ((X ** 2) * (residuals ** 2)).sum(axis=0)

    for lag in range(1, Lmax + 1):
        weight = 1.0 - lag / (Lmax + 1.0)
        X_t = X[lag:, :]
        X_lag = X[:-lag, :]
        e_t = residuals[lag:, :]
        e_lag = residuals[:-lag, :]
        valid_pair = valid[lag:, :] & valid[:-lag, :]
        S_lag = (X_t * e_t * X_lag * e_lag * valid_pair).sum(axis=0)
        S_sum += 2.0 * weight * S_lag

    var_nw = np.full(P, np.nan, dtype=np.float64)
    nw_ok = slope_ok & (sum_x2 > 0)
    var_nw[nw_ok] = S_sum[nw_ok] / (sum_x2[nw_ok] ** 2)

    nw_t = np.full(P, np.nan, dtype=np.float64)
    nw_se_ok = nw_ok & np.isfinite(var_nw) & (var_nw > 0)
    nw_t[nw_se_ok] = slope[nw_se_ok] / np.sqrt(var_nw[nw_se_ok])
    # Production behaviour: use the asymptotic standard-normal two-sided
    # probability for the Newey-West HAC statistic.
    nw_p = two_sided_normal_p_array(nw_t)

    def img(v):
        return v.reshape(H, W).astype(np.float32)

    return {
        "slope": img(slope),
        "intercept": img(intercept),
        "r": img(r),
        "r2": img(r2),
        "p": img(p),
        "nw_t": img(nw_t),
        "nw_p": img(nw_p),
        "valid_n": n.reshape(H, W).astype(np.uint16),
    }


def mann_kendall_stats_nan(y_stack: np.ndarray, min_obs: int) -> Dict[str, np.ndarray]:
    N, H, W = y_stack.shape
    P = H * W
    Y = y_stack.reshape(N, P).astype(np.float64)
    valid = np.isfinite(Y)
    n = valid.sum(axis=0).astype(np.float64)
    ok = n >= min_obs

    S = np.zeros(P, dtype=np.float64)
    for i in range(N - 1):
        yi = Y[i]
        vi = valid[i]
        for j in range(i + 1, N):
            yj = Y[j]
            vj = valid[j]
            m = vi & vj
            diff = yj - yi
            S[(diff > 0) & m] += 1
            S[(diff < 0) & m] -= 1

    mk_z = np.full(P, np.nan, dtype=np.float64)
    tau = np.full(P, np.nan, dtype=np.float64)

    n_ok = n[ok]
    varS = n_ok * (n_ok - 1.0) * (2.0 * n_ok + 5.0) / 18.0
    valid_var = varS > 0
    if valid_var.any():
        ok_idx = np.where(ok)[0][valid_var]
        S_sub = S[ok_idx]
        var_sub = varS[valid_var]
        z_vals = np.zeros_like(S_sub, dtype=np.float64)
        pos = S_sub > 0
        neg = S_sub < 0
        z_vals[pos] = (S_sub[pos] - 1.0) / np.sqrt(var_sub[pos])
        z_vals[neg] = (S_sub[neg] + 1.0) / np.sqrt(var_sub[neg])
        mk_z[ok_idx] = z_vals

    mk_p = two_sided_normal_p_array(mk_z)
    denom = 0.5 * n * (n - 1.0)
    tau_ok = ok & (denom > 0)
    tau[tau_ok] = S[tau_ok] / denom[tau_ok]

    return {
        "mk_z": mk_z.reshape(H, W).astype(np.float32),
        "mk_p": mk_p.reshape(H, W).astype(np.float32),
        "mk_tau": tau.reshape(H, W).astype(np.float32),
    }

# =============================================================================
# 9. LULC TRANSITION VALIDATION DIAGNOSTIC
# =============================================================================

def compute_lulc_stability(lulc_stack: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute LULC transition diagnostics for validating NDVI degradation results.

    Philosophy
    ----------
    This diagnostic is not intended to classify LULC stability as an ecological
    condition in itself. Its purpose is to support post-trend validation:

        Among pixels reported as degraded by NDVI trend tests, did the pixel
        remain sufficiently stable in LULC through the trend window?

    This helps distinguish:

        1. likely genuine vegetation-condition degradation within a persistent
           broad ecosystem type

        from

        2. degradation signals that may be affected by land-cover transition.

    Inputs
    ------
    lulc_stack : np.ndarray
        Array with shape (N, H, W), where N is the number of years in the
        trend window. Values are major LULC group IDs. Zero is treated as NoData.

    Outputs
    -------
    Dictionary of arrays:

        lulc_mode
            Dominant LULC group across the trend window.

        lulc_mode_count
            Number of valid years assigned to the dominant LULC group.

        lulc_mode_proportion
            Dominant LULC count divided by number of valid LULC years.

        lulc_unique_count
            Number of unique valid LULC groups observed during the trend window.

        lulc_transition_count
            Number of year-to-year transitions between valid consecutive years.

        lulc_possible_transition_count
            Number of valid consecutive year-pairs where a transition could be
            evaluated.

        lulc_transition_rate
            transition_count / possible_transition_count.

        lulc_first_last_same
            1 where the first valid and last valid LULC classes are the same.

        lulc_degradation_validation_class
            0 = NoData / insufficient LULC history
                0: "NoData",
                1: "1 | 0 transitions — strongest evidence of genuine degradation",
                2: "2 | 1–2 transitions — likely genuine degradation",
                3: "3 | 3–5 transitions — moderate transition, interpret with caution",
                4: "4 | >5 transitions — likely transition-affected degradation"

        lulc_stability_class
            Backward-compatible alias of lulc_degradation_validation_class.
            This prevents older writing code from breaking, but the preferred
            interpretation is now degradation-validation confidence.

    Threshold logic
    ---------------
    The thresholds are scaled to the number of possible transitions in each
    trend window.

    They mimic this rule for a 22-year window, where 21 transitions are possible:

        0 transitions     -> class 1
        1–2 transitions   -> class 2
        3–5 transitions   -> class 3
        >5 transitions    -> class 4

    For shorter windows, the thresholds scale proportionally. For example,
    for an 11-year window with 10 possible transitions:

        0 transitions     -> class 1
        1 transition      -> class 2
        2 transitions     -> class 3
        >=3 transitions   -> class 4

    Notes
    -----
    This function classifies all pixels, but the interpretation should be made
    mainly after intersecting these classes with degradation masks.
    """

    N, H, W = lulc_stack.shape

    # -------------------------------------------------------------------------
    # 1. Basic valid LULC history
    # -------------------------------------------------------------------------
    valid = lulc_stack > 0
    n_valid = valid.sum(axis=0).astype(np.uint16)

    # -------------------------------------------------------------------------
    # 2. Dominant LULC mode diagnostics
    # -------------------------------------------------------------------------
    counts = []

    for gid in LULC_GROUP_LABELS.keys():
        counts.append((lulc_stack == gid).sum(axis=0).astype(np.uint16))

    count_cube = np.stack(counts, axis=0)
    group_ids = np.array(list(LULC_GROUP_LABELS.keys()), dtype=np.uint8)

    max_idx = np.argmax(count_cube, axis=0)
    mode_count = np.max(count_cube, axis=0).astype(np.uint16)

    lulc_mode = np.zeros((H, W), dtype=np.uint8)
    has_lulc = mode_count > 0
    lulc_mode[has_lulc] = group_ids[max_idx[has_lulc]]

    lulc_unique_count = (count_cube > 0).sum(axis=0).astype(np.uint8)

    lulc_mode_proportion = np.full((H, W), np.nan, dtype=np.float32)
    ok_valid = n_valid > 0
    lulc_mode_proportion[ok_valid] = (
        mode_count[ok_valid].astype(np.float32)
        / n_valid[ok_valid].astype(np.float32)
    )

    # -------------------------------------------------------------------------
    # 3. First-valid and last-valid LULC classes
    # -------------------------------------------------------------------------
    # This is safer than comparing only the first and last array slices, because
    # a pixel may have NoData in the first or final year.
    valid_flat = valid.reshape(N, -1)
    lulc_flat = lulc_stack.reshape(N, -1)

    P = lulc_flat.shape[1]
    has_any_valid = valid_flat.any(axis=0)

    first_idx = np.argmax(valid_flat, axis=0)
    last_idx = N - 1 - np.argmax(valid_flat[::-1, :], axis=0)

    first_lulc_flat = np.zeros(P, dtype=np.uint8)
    last_lulc_flat = np.zeros(P, dtype=np.uint8)

    pix_idx = np.arange(P)
    first_lulc_flat[has_any_valid] = lulc_flat[
        first_idx[has_any_valid],
        pix_idx[has_any_valid]
    ]

    last_lulc_flat[has_any_valid] = lulc_flat[
        last_idx[has_any_valid],
        pix_idx[has_any_valid]
    ]

    first_lulc = first_lulc_flat.reshape(H, W)
    last_lulc = last_lulc_flat.reshape(H, W)

    lulc_first_last_same = (
        (first_lulc == last_lulc)
        & (first_lulc > 0)
        & (last_lulc > 0)
    ).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 4. Year-to-year transition diagnostics
    # -------------------------------------------------------------------------
    # A transition is counted only where both consecutive years are valid and
    # the LULC major group changes.
    consecutive_valid = valid[:-1, :, :] & valid[1:, :, :]
    consecutive_changed = lulc_stack[:-1, :, :] != lulc_stack[1:, :, :]

    lulc_transition_count = (
        consecutive_valid & consecutive_changed
    ).sum(axis=0).astype(np.uint16)

    lulc_possible_transition_count = (
        consecutive_valid
    ).sum(axis=0).astype(np.uint16)

    lulc_transition_rate = np.full((H, W), np.nan, dtype=np.float32)
    ok_transition = lulc_possible_transition_count > 0
    lulc_transition_rate[ok_transition] = (
        lulc_transition_count[ok_transition].astype(np.float32)
        / lulc_possible_transition_count[ok_transition].astype(np.float32)
    )

    # -------------------------------------------------------------------------
    # 5. Degradation-validation class based on transition history
    # -------------------------------------------------------------------------
    # Use global thresholds if defined in Section 4. Otherwise, fall back to
    # the 22-year-window logic:
    #   low transition      <= 2/21
    #   moderate transition <= 5/21
    low_transition_rate = globals().get("LOW_TRANSITION_RATE", 2 / 21)
    moderate_transition_rate = globals().get("MODERATE_TRANSITION_RATE", 5 / 21)

    # Thresholds are pixel-specific because possible_transition_count may vary
    # if some LULC years are missing or NoData.
    low_transition_threshold = np.floor(
        low_transition_rate * lulc_possible_transition_count.astype(np.float32)
    ).astype(np.int16)

    moderate_transition_threshold = np.floor(
        moderate_transition_rate * lulc_possible_transition_count.astype(np.float32)
    ).astype(np.int16)

    # For pixels with enough LULC history, allow at least:
    #   1 transition as "low"
    #   2 transitions as "moderate"
    # in shorter windows. This avoids classifying every short-window transition
    # as high transition too aggressively.
    low_transition_threshold = np.where(
        ok_transition,
        np.maximum(low_transition_threshold, 1),
        0
    )

    moderate_transition_threshold = np.where(
        ok_transition,
        np.maximum(moderate_transition_threshold, low_transition_threshold + 1),
        0
    )

    lulc_degradation_validation_class = np.zeros((H, W), dtype=np.uint8)

    # Class 1: no LULC transition
    class_1 = ok_transition & (lulc_transition_count == 0)

    # Class 2: low LULC transition
    class_2 = (
        ok_transition
        & (lulc_transition_count > 0)
        & (lulc_transition_count <= low_transition_threshold)
    )

    # Class 3: moderate LULC transition
    class_3 = (
        ok_transition
        & (lulc_transition_count > low_transition_threshold)
        & (lulc_transition_count <= moderate_transition_threshold)
    )

    # Class 4: high transition / likely transition-affected
    class_4 = (
        ok_transition
        & (lulc_transition_count > moderate_transition_threshold)
    )

    lulc_degradation_validation_class[class_1] = 1
    lulc_degradation_validation_class[class_2] = 2
    lulc_degradation_validation_class[class_3] = 3
    lulc_degradation_validation_class[class_4] = 4

    # -------------------------------------------------------------------------
    # 6. Return diagnostics
    # -------------------------------------------------------------------------
    return {
        "lulc_mode": lulc_mode,
        "lulc_mode_count": mode_count,
        "lulc_mode_proportion": lulc_mode_proportion,
        "lulc_unique_count": lulc_unique_count,

        "lulc_transition_count": lulc_transition_count,
        "lulc_possible_transition_count": lulc_possible_transition_count,
        "lulc_transition_rate": lulc_transition_rate,

        "lulc_first_last_same": lulc_first_last_same,
        "lulc_degradation_validation_class": lulc_degradation_validation_class,

        # Backward-compatible alias.
        # The preferred name and interpretation is now:
        # "lulc_degradation_validation_class".
        "lulc_stability_class": lulc_degradation_validation_class,
    }
# =============================================================================
# 10. MASK CREATION
# =============================================================================

def build_degradation_masks(stats: Dict[str, np.ndarray], mk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Build the configured NDVI Trend confirmation and degradation masks.

    Active criteria:
        - slope direction / physical slope threshold
        - OLS p-value
        - Mann-Kendall p-value and tau direction
        - Newey-West corrected p-value

    """

    slope = stats["slope"]
    p = stats["p"]
    nw_p = stats["nw_p"]
    mk_p = mk["mk_p"]
    mk_tau = mk["mk_tau"]

    valid_slope = np.isfinite(slope)

    DegrPotential = ((slope <= SLOPE_TH_PHYSICAL) & valid_slope).astype(np.uint8)
    ConfirmP_all = ((p < P_90) & np.isfinite(p)).astype(np.uint8)
    DegrP_neg = ((p < P_90) & (slope <= SLOPE_TH_PHYSICAL) & np.isfinite(p) & valid_slope).astype(np.uint8)
    ConfirmNW_all = ((nw_p < P_90) & np.isfinite(nw_p)).astype(np.uint8)
    DegrNW_neg = ((nw_p < P_90) & (slope <= SLOPE_TH_PHYSICAL) & np.isfinite(nw_p) & valid_slope).astype(np.uint8)
    ConfirmMK_all = ((mk_p < P_90) & np.isfinite(mk_p)).astype(np.uint8)
    DegrMK_neg = ((mk_p < P_90) & (mk_tau < 0) & np.isfinite(mk_p) & np.isfinite(mk_tau)).astype(np.uint8)
    ConfirmPMK_all = ((ConfirmP_all == 1) & (ConfirmMK_all == 1)).astype(np.uint8)
    DegrPMK_neg = ((DegrP_neg == 1) & (DegrMK_neg == 1)).astype(np.uint8)
    ConfirmPNW_all = ((ConfirmP_all == 1) & (ConfirmNW_all == 1)).astype(np.uint8)
    DegrPNW_neg = ((DegrP_neg == 1) & (DegrNW_neg == 1)).astype(np.uint8)
    ConfirmMKNW_all = ((ConfirmMK_all == 1) & (ConfirmNW_all == 1)).astype(np.uint8)
    DegrMKNW_neg = ((DegrMK_neg == 1) & (DegrNW_neg == 1)).astype(np.uint8)
    ConfirmPMKNW_all = ((ConfirmP_all == 1) & (ConfirmMK_all == 1) & (ConfirmNW_all == 1)).astype(np.uint8)
    # Strict supported decline used by the current workflow:
    #   OLS support + physical negative-slope threshold
    #   AND Mann-Kendall support with negative tau
    #   AND Newey-West support + physical negative-slope threshold.
    DegrPMKNW_neg = ((DegrP_neg == 1) & (DegrMK_neg == 1) & (DegrNW_neg == 1)).astype(np.uint8)
    return {
        "DegrPotential": DegrPotential,
        "ConfirmP_all": ConfirmP_all, 
        "DegrP_neg": DegrP_neg,
        "ConfirmNW_all": ConfirmNW_all, 
        "DegrNW_neg": DegrNW_neg,
        "ConfirmMK_all": ConfirmMK_all, 
        "DegrMK_neg": DegrMK_neg,
        "ConfirmPMK_all": ConfirmPMK_all, 
        "DegrPMK_neg": DegrPMK_neg,
        "ConfirmPNW_all": ConfirmPNW_all, 
        "DegrPNW_neg": DegrPNW_neg,
        "ConfirmMKNW_all": ConfirmMKNW_all, 
        "DegrMKNW_neg": DegrMKNW_neg,
        "ConfirmPMKNW_all": ConfirmPMKNW_all, 
        "DegrPMKNW_neg": DegrPMKNW_neg,
    }
# =============================================================================
# 11. BUILD CONTINENTAL TREND RASTERS
# =============================================================================

TREND_FLOAT_METRICS = {
    "NDVI_slope_per_year": "slope",
    "NDVI_intercept": "intercept",
    "NDVI_r": "r",
    "NDVI_R2": "r2",
    "NDVI_p": "p",
    "NDVI_NeweyWest_t": "nw_t",
    "NDVI_NeweyWest_p": "nw_p",
    "NDVI_MK_z": "mk_z",
    "NDVI_MK_p": "mk_p",
    "NDVI_MK_tau": "mk_tau",
    "LULC_mode_proportion": "lulc_mode_proportion",
}

TREND_UINT16_METRICS = {
    "NDVI_valid_n": "valid_n",
    "LULC_mode_count": "lulc_mode_count",
}

TREND_UINT8_METRICS = {
    "LULC_mode": "lulc_mode",
    "LULC_unique_count": "lulc_unique_count",
    "LULC_first_last_same": "lulc_first_last_same",
    "LULC_stability_class": "lulc_stability_class",
}


def _exists_and_nonempty(path: Path) -> bool:
    """
    Return True only for an existing, non-empty, readable one-band raster.

    This archive check does not inspect or modify raster values. It prevents a
    truncated/corrupt GeoTIFF from being treated as a completed Trend output.
    """
    if not (path.exists() and path.is_file() and path.stat().st_size > 0):
        return False

    try:
        with rasterio.open(path) as src:
            return (
                src.count == 1
                and src.width > 0
                and src.height > 0
                and src.crs is not None
            )
    except Exception:
        return False


def _expected_output_paths_for_window(window_name: str) -> List[Path]:
    """Build the full list of expected outputs for one trend window."""
    paths = []

    for metric in TREND_FLOAT_METRICS:
        paths.append(lulc_stability_path(window_name, metric) if metric.startswith("LULC_") else trend_raster_path(window_name, metric))

    for metric in TREND_UINT16_METRICS:
        paths.append(lulc_stability_path(window_name, metric) if metric.startswith("LULC_") else trend_raster_path(window_name, metric))

    for metric in TREND_UINT8_METRICS:
        paths.append(lulc_stability_path(window_name, metric))

    for mask_name in MASK_LABELS.keys():
        paths.append(mask_raster_path(window_name, mask_name))

    return paths


def missing_outputs_for_window(window_name: str) -> List[Path]:
    """Return all expected outputs that are missing or empty for one trend window."""
    return [p for p in _expected_output_paths_for_window(window_name) if not _exists_and_nonempty(p)]


def window_already_complete(window_name: str) -> bool:
    """True only when all expected outputs for a window exist and are non-empty."""
    return len(missing_outputs_for_window(window_name)) == 0


def _read_ndvi_window_as_float(src, win) -> np.ndarray:
    """
    Read one annual State-cache NDVI window as float32.

    Screening here is limited to declared/configured NoData and the configured
    physical NDVI range. No MOD13Q1 QA-band filtering is added at the Trend
    stage, preserving the production implementation.
    """
    arr = src.read(1, window=win).astype(np.float32)

    src_nodata = src.nodata
    if src_nodata is not None:
        arr[arr == src_nodata] = np.nan

    if NDVI_NODATA is not None:
        try:
            if not np.isnan(float(NDVI_NODATA)):
                arr[arr == NDVI_NODATA] = np.nan
        except Exception:
            arr[arr == NDVI_NODATA] = np.nan

    lo, hi = VALID_NDVI_RANGE
    arr[(arr < lo) | (arr > hi)] = np.nan

    return arr


def build_trend_rasters_for_window(window_name: str, years: List[int]) -> None:
    """
    Build all NDVI trend, degradation-mask, and LULC-stability rasters for one trend window.

    The function is windowed/tiled to avoid loading the full continental raster stack into memory.
    """
    mkdir(get_raster_dir(window_name))
    mkdir(get_diagnostic_dir(window_name))

    if REUSE_EXISTING_TREND_RASTERS and not FORCE_REBUILD_TREND_RASTERS:
        missing = missing_outputs_for_window(window_name)

        if len(missing) == 0:
            print(f"[reuse] {window_name}: all expected outputs already exist; skipping.")
            return

        print(f"[reuse-check] {window_name}: outputs are incomplete; rebuilding this window.")
        print(f"[reuse-check] missing or empty files: {len(missing):,}")
        for p in missing[:10]:
            print(f"  - {p}")
        if len(missing) > 10:
            print(f"  ... plus {len(missing) - 10:,} more")

    print("\n" + "=" * 100)
    print(f"[trend] building continental trend rasters: {window_name}")
    print(f"[trend] years: {years[0]}-{years[-1]} | N={len(years)} | TIME_TAG={TIME_TAG}")
    print("=" * 100)

    ndvi_paths = [ndvi_path(y) for y in years]
    lulc_paths = [lulc_path(y) for y in years]

    template_profile = validate_alignment(ndvi_paths + lulc_paths + [ecoregion_id_path(), country_id_path()])

    width = template_profile["width"]
    height = template_profile["height"]
    min_obs = min_valid_obs_for_window(len(years))

    print(f"[trend] raster size: {width:,} × {height:,}")
    print(f"[trend] min valid observations: {min_obs}/{len(years)}")

    manifest_rows = []

    with ExitStack() as stack:
        ndvi_srcs = [stack.enter_context(rasterio.open(p)) for p in ndvi_paths]
        lulc_srcs = [stack.enter_context(rasterio.open(p)) for p in lulc_paths]

        float_profile = output_profile_like(template_profile, "float32", FLOAT_NODATA)
        uint8_profile = output_profile_like(template_profile, "uint8", UINT8_NODATA)
        uint16_profile = output_profile_like(template_profile, "uint16", UINT16_NODATA)

        float_dsts = {}
        for out_metric in TREND_FLOAT_METRICS:
            out_path = lulc_stability_path(window_name, out_metric) if out_metric.startswith("LULC_") else trend_raster_path(window_name, out_metric)
            float_dsts[out_metric] = stack.enter_context(rasterio.open(out_path, "w", **float_profile))
            manifest_rows.append({"window": window_name, "time_tag": TIME_TAG, "type": "float32", "metric": out_metric, "path": str(out_path)})

        uint16_dsts = {}
        for out_metric in TREND_UINT16_METRICS:
            out_path = lulc_stability_path(window_name, out_metric) if out_metric.startswith("LULC_") else trend_raster_path(window_name, out_metric)
            uint16_dsts[out_metric] = stack.enter_context(rasterio.open(out_path, "w", **uint16_profile))
            manifest_rows.append({"window": window_name, "time_tag": TIME_TAG, "type": "uint16", "metric": out_metric, "path": str(out_path)})

        uint8_dsts = {}
        for out_metric in TREND_UINT8_METRICS:
            out_path = lulc_stability_path(window_name, out_metric)
            uint8_dsts[out_metric] = stack.enter_context(rasterio.open(out_path, "w", **uint8_profile))
            manifest_rows.append({"window": window_name, "time_tag": TIME_TAG, "type": "uint8", "metric": out_metric, "path": str(out_path)})

        mask_dsts = {}
        for mask_name in MASK_LABELS.keys():
            out_path = mask_raster_path(window_name, mask_name)
            mask_dsts[mask_name] = stack.enter_context(rasterio.open(out_path, "w", **uint8_profile))
            manifest_rows.append({"window": window_name, "time_tag": TIME_TAG, "type": "mask_uint8", "metric": mask_name, "label": MASK_LABELS.get(mask_name, mask_name), "path": str(out_path)})

        windows = list(iter_windows(width, height, BLOCK_SIZE))
        n_windows = len(windows)
        t0 = time.perf_counter()

        for i, win in enumerate(windows, start=1):

            # --------------------------------------------------------------
            # NDVI stack for this window
            # --------------------------------------------------------------
            ndvi_stack = [_read_ndvi_window_as_float(src, win) for src in ndvi_srcs]
            Y = np.stack(ndvi_stack, axis=0)

            stats = linear_stats_nan(Y, years, min_obs=min_obs)
            mk = mann_kendall_stats_nan(Y, min_obs=min_obs)
            masks = build_degradation_masks(stats, mk)

            # Defensive check: every configured mask should be produced.
            missing_mask_outputs = set(mask_dsts.keys()) - set(masks.keys())
            if missing_mask_outputs:
                raise KeyError(f"build_degradation_masks() did not return these configured masks: {sorted(missing_mask_outputs)}")

            # --------------------------------------------------------------
            # LULC stability stack for this window
            # --------------------------------------------------------------
            lulc_stack = [src.read(1, window=win).astype(np.uint8) for src in lulc_srcs]
            L = np.stack(lulc_stack, axis=0)
            lulc_diag = compute_lulc_stability(L)

            # --------------------------------------------------------------
            # Write trend outputs
            # --------------------------------------------------------------
            write_float_window(float_dsts["NDVI_slope_per_year"], stats["slope"], win)
            write_float_window(float_dsts["NDVI_intercept"], stats["intercept"], win)
            write_float_window(float_dsts["NDVI_r"], stats["r"], win)
            write_float_window(float_dsts["NDVI_R2"], stats["r2"], win)
            write_float_window(float_dsts["NDVI_p"], stats["p"], win)
            write_float_window(float_dsts["NDVI_NeweyWest_t"], stats["nw_t"], win)
            write_float_window(float_dsts["NDVI_NeweyWest_p"], stats["nw_p"], win)
            write_float_window(float_dsts["NDVI_MK_z"], mk["mk_z"], win)
            write_float_window(float_dsts["NDVI_MK_p"], mk["mk_p"], win)
            write_float_window(float_dsts["NDVI_MK_tau"], mk["mk_tau"], win)
            write_uint16_window(uint16_dsts["NDVI_valid_n"], stats["valid_n"], win)

            # --------------------------------------------------------------
            # Write degradation masks
            # --------------------------------------------------------------
            for mask_name, mask_dst in mask_dsts.items():
                write_uint8_window(mask_dst, masks[mask_name], win)

            # --------------------------------------------------------------
            # Write LULC-stability diagnostics
            # --------------------------------------------------------------
            write_float_window(float_dsts["LULC_mode_proportion"], lulc_diag["lulc_mode_proportion"], win)
            write_uint16_window(uint16_dsts["LULC_mode_count"], lulc_diag["lulc_mode_count"], win)

            for out_metric, internal_key in TREND_UINT8_METRICS.items():
                write_uint8_window(uint8_dsts[out_metric], lulc_diag[internal_key], win)

            if i % 50 == 0 or i == n_windows:
                print(f"[{window_name}] {i:,}/{n_windows:,} windows | elapsed {_fmt(time.perf_counter() - t0)}")

    manifest = pd.DataFrame(manifest_rows)
    save_csv(manifest, get_diagnostic_dir(window_name) / f"{PREFIX}_{window_name}_output_manifest_{TIME_TAG}.csv")

    print(f"[trend] completed {window_name}")
# =============================================================================
# 12. OPTIONAL SUMMARY TABLES
# =============================================================================

def configured_summary_masks() -> List[str]:
    """Return the Trend masks configured for summary-table generation."""
    return list(SUMMARY_MASKS)


def get_pixel_area_ha_from_profile(path: Path) -> float:
    """
    Estimate pixel area in hectares from the affine transform.

    This assumes the raster is in a projected coordinate reference system with
    metre-based units. If the raster is in degrees, this value will not represent
    true hectares.
    """
    with rasterio.open(path) as src:
        return abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d) / 10000.0


def update_group_accumulator(acc, group_cols, group_arrays, degraded, valid, pixel_area_ha, window_name, mask_name, summary_level):
    """
    Update a dictionary-based grouped accumulator for one mask/window/grouping.

    The accumulator stores pixel counts, not hectares. Hectares are calculated
    only once at the end using pixel_area_ha.

    Important:
        The 'valid' mask passed into this function now controls the denominator.
        In summarise_window(), all summary levels are restricted to ANALYSIS_LULC_GROUPS
        so total_pixels and total_ha represent the selected reporting/vegetated LULC domain.
    """
    if not np.any(valid):
        return

    group_values = [arr[valid].ravel().astype(np.int64) for arr in group_arrays]

    if len(group_values) == 0 or group_values[0].size == 0:
        return

    degraded_values = degraded[valid].astype(np.uint8).ravel()
    key_matrix = np.column_stack(group_values)

    unique_keys, inverse = np.unique(key_matrix, axis=0, return_inverse=True)

    total_counts = np.bincount(inverse).astype(np.float64)
    degraded_counts = np.bincount(inverse, weights=degraded_values.astype(np.float64))

    for key_values, total_count, degraded_count in zip(unique_keys, total_counts, degraded_counts):
        key = (window_name, mask_name, summary_level, *[int(v) for v in key_values])

        if key not in acc:
            acc[key] = {"total_pixels": 0.0, "degraded_pixels": 0.0}

        acc[key]["total_pixels"] += float(total_count)
        acc[key]["degraded_pixels"] += float(degraded_count)


def summary_required_paths(window_name: str) -> List[Path]:
    """
    Return all raster inputs required to summarise one trend window.

    Only masks explicitly listed in SUMMARY_MASKS are required.
    """
    paths = [
        ecoregion_id_path(),
        country_id_path(),
        lulc_stability_path(window_name, "LULC_mode"),
        lulc_stability_path(window_name, "LULC_stability_class"),
    ]

    for mask_name in configured_summary_masks():
        paths.append(mask_raster_path(window_name, mask_name))

    return paths


def precheck_summary_inputs(window_name: str) -> pd.DataFrame:
    """
    Check whether all required summary inputs exist, are non-empty, and are readable.
    """
    rows = []

    for p in summary_required_paths(window_name):
        row = {
            "trend_window": window_name,
            "file_name": p.name,
            "path": str(p),
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 6) if p.exists() else 0.0,
            "readable": False,
            "width": None,
            "height": None,
            "dtype": None,
            "nodata": None,
            "error": "",
        }

        if p.exists() and p.stat().st_size > 0:
            try:
                with rasterio.open(p) as src:
                    row["readable"] = True
                    row["width"] = src.width
                    row["height"] = src.height
                    row["dtype"] = src.dtypes[0]
                    row["nodata"] = src.nodata
            except Exception as e:
                row["error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)


def write_summary_precheck_report(window_name: str) -> pd.DataFrame:
    """
    Write a per-window summary-input precheck table and return it.
    """
    mkdir(get_diagnostic_dir(window_name))
    report = precheck_summary_inputs(window_name)

    out_csv = get_diagnostic_dir(window_name) / f"{PREFIX}_{window_name}_summary_input_precheck_{TIME_TAG}.csv"
    save_csv(report, out_csv)

    bad = report[(~report["exists"]) | (~report["readable"]) | (report["size_mb"] <= 0)].copy()

    if not bad.empty:
        print(f"[summary precheck warning] {window_name}: {len(bad):,} required inputs are missing/unreadable.")
        print(bad[["file_name", "exists", "readable", "size_mb", "path"]].to_string(index=False))
    else:
        print(f"[summary precheck] {window_name}: all required summary inputs are present and readable.")

    return report


def summarise_window(window_name: str) -> pd.DataFrame:
    """
    Summarise degradation-mask rasters by ecoregion, country, LULC mode, and
    LULC-stability class.

    Critical denominator correction:
        total_pixels and total_ha are now based only on selected reporting LULC pixels:
            valid_reporting_lulc = valid_eco & np.isin(lulc, ANALYSIS_LULC_GROUPS)

        This means the summary denominator mirrors the selected vegetated/reporting domain
        rather than all valid ecoregion pixels.
    """
    print("\n" + "=" * 100)
    print(f"[summary] summarising trend rasters: {window_name}")
    print("=" * 100)

    if not SUMMARY_MASKS:
        raise ValueError("SUMMARY_MASKS is empty. At least one mask name is required for summary tables.")

    requested_summary_masks = configured_summary_masks()

    if not requested_summary_masks:
        raise ValueError("No SUMMARY_MASKS are configured for summary tables.")

    mkdir(get_summary_dir(window_name))
    write_summary_precheck_report(window_name)

    lulc_mode_path = lulc_stability_path(window_name, "LULC_mode")
    lulc_stability_class_path = lulc_stability_path(window_name, "LULC_stability_class")

    missing_summary_masks = [name for name in requested_summary_masks if not mask_raster_path(window_name, name).exists()]
    available_summary_masks = [name for name in requested_summary_masks if mask_raster_path(window_name, name).exists()]

    if missing_summary_masks:
        print(f"[summary warning] {window_name}: these SUMMARY_MASKS are missing and will be skipped: {missing_summary_masks}")

    if not available_summary_masks:
        raise FileNotFoundError(f"No requested SUMMARY_MASKS are available for {window_name}.")

    base_mask = mask_raster_path(window_name, available_summary_masks[0])

    required_summary_inputs = [base_mask, ecoregion_id_path(), country_id_path(), lulc_mode_path, lulc_stability_class_path]

    for p in required_summary_inputs:
        if not p.exists():
            raise FileNotFoundError(f"Cannot summarise. Missing required input: {p}")

    validate_alignment(required_summary_inputs)

    with rasterio.open(base_mask) as tmp:
        width, height = tmp.width, tmp.height

    pixel_ha = get_pixel_area_ha_from_profile(base_mask)
    print(f"[summary] pixel area: {pixel_ha:.6f} ha")
    print(f"[summary] denominator basis: selected ANALYSIS_LULC_GROUPS only")
    print(f"[summary] ANALYSIS_LULC_GROUPS: {ANALYSIS_LULC_GROUPS}")

    acc = {}

    with ExitStack() as stack:
        eco_src = stack.enter_context(rasterio.open(ecoregion_id_path()))
        country_src = stack.enter_context(rasterio.open(country_id_path()))
        lulc_src = stack.enter_context(rasterio.open(lulc_mode_path))
        stab_src = stack.enter_context(rasterio.open(lulc_stability_class_path))

        mask_srcs = {name: stack.enter_context(rasterio.open(mask_raster_path(window_name, name))) for name in available_summary_masks}

        windows = list(iter_windows(width, height, BLOCK_SIZE))
        n_windows = len(windows)
        t0 = time.perf_counter()

        for i, win in enumerate(windows, start=1):
            eco = eco_src.read(1, window=win).astype(np.int32)
            country = country_src.read(1, window=win).astype(np.int32)
            lulc = lulc_src.read(1, window=win).astype(np.uint8)
            stability = stab_src.read(1, window=win).astype(np.uint8)

            # -----------------------------------------------------------------
            # Correct denominator:
            # All summaries are restricted to selected analysis/vegetated LULC.
            # This fixes total_pixels -> total_ha so it no longer counts all
            # valid ecoregion pixels.
            # -----------------------------------------------------------------
            valid_eco = eco > 0

            # Production denominator retained unchanged:
            # selected window-modal LULC domain only. This is intentionally NOT
            # intersected with NDVI_valid_n, because doing so would change the
            # reported degradation percentages.
            valid_reporting_lulc = valid_eco & np.isin(lulc, ANALYSIS_LULC_GROUPS)
            valid_country_reporting_lulc = valid_reporting_lulc & (country > 0)
            valid_stability_reporting_lulc = valid_reporting_lulc & (stability > 0)

            for mask_name, mask_src in mask_srcs.items():
                degraded = mask_src.read(1, window=win).astype(np.uint8) == 1

                update_group_accumulator(acc, ["ECO_ID"], [eco], degraded, valid_reporting_lulc, pixel_ha, window_name, mask_name, "ecoregion")

                update_group_accumulator(acc, ["COUNTRY_ID"], [country], degraded, valid_country_reporting_lulc, pixel_ha, window_name, mask_name, "country")

                update_group_accumulator(acc, ["ECO_ID", "COUNTRY_ID"], [eco, country], degraded, valid_country_reporting_lulc, pixel_ha, window_name, mask_name, "ecoregion_country")

                update_group_accumulator(acc, ["ECO_ID", "LULC_group"], [eco, lulc], degraded, valid_reporting_lulc, pixel_ha, window_name, mask_name, "ecoregion_lulc_mode")

                update_group_accumulator(acc, ["ECO_ID", "COUNTRY_ID", "LULC_group"], [eco, country, lulc], degraded, valid_country_reporting_lulc, pixel_ha, window_name, mask_name, "ecoregion_country_lulc_mode")

                update_group_accumulator(acc, ["ECO_ID", "LULC_stability_class"], [eco, stability], degraded, valid_stability_reporting_lulc, pixel_ha, window_name, mask_name, "ecoregion_lulc_stability")

            if i % 100 == 0 or i == n_windows:
                print(f"[summary {window_name}] {i:,}/{n_windows:,} windows | elapsed {_fmt(time.perf_counter() - t0)}")

    rows = []

    for key, vals in acc.items():
        window, mask_name, level, *group_values = key

        total_pixels = vals["total_pixels"]
        degraded_pixels = vals["degraded_pixels"]

        total_ha = total_pixels * pixel_ha
        degraded_ha = degraded_pixels * pixel_ha
        pct = degraded_ha / total_ha * 100.0 if total_ha > 0 else 0.0

        row = {
            "trend_window": window,
            "time_tag": TIME_TAG,
            "mask_name": mask_name,
            "mask_label": MASK_LABELS.get(mask_name, mask_name),
            "summary_level": level,
            "total_area_basis": "selected_ANALYSIS_LULC_GROUPS",
            "total_pixels": total_pixels,
            "degraded_pixels": degraded_pixels,
            "pixel_area_ha": pixel_ha,
            "total_ha": total_ha,
            "degraded_ha": degraded_ha,
            "degraded_pct": pct,
        }

        if level == "ecoregion":
            row["ECO_ID"] = group_values[0]

        elif level == "country":
            row["COUNTRY_ID"] = group_values[0]

        elif level == "ecoregion_country":
            row["ECO_ID"], row["COUNTRY_ID"] = group_values

        elif level == "ecoregion_lulc_mode":
            row["ECO_ID"], row["LULC_group"] = group_values

        elif level == "ecoregion_country_lulc_mode":
            row["ECO_ID"], row["COUNTRY_ID"], row["LULC_group"] = group_values

        elif level == "ecoregion_lulc_stability":
            row["ECO_ID"], row["LULC_stability_class"] = group_values

        rows.append(row)

    df = pd.DataFrame(rows)

    eco_lookup_path = LOOKUP_DIR / "ecoregion_lookup.csv"
    country_lookup_path = LOOKUP_DIR / "country_lookup.csv"

    if not df.empty:
        if "ECO_ID" in df.columns and eco_lookup_path.exists():
            df = df.merge(pd.read_csv(eco_lookup_path), on="ECO_ID", how="left")

        if "COUNTRY_ID" in df.columns and country_lookup_path.exists():
            df = df.merge(pd.read_csv(country_lookup_path), on="COUNTRY_ID", how="left")

        if "LULC_group" in df.columns:
            df["LULC_label"] = df["LULC_group"].map(LULC_GROUP_LABELS)

        if "LULC_stability_class" in df.columns:
            df["LULC_stability_class"] = df["LULC_stability_class"].map(LULC_STABILITY_LABELS)

    out_csv = get_summary_dir(window_name) / f"{PREFIX}_{window_name}_summary_tables_{TIME_TAG}.csv"
    save_csv(df, out_csv)

    print(f"[summary] saved: {out_csv}")
    return df


# =============================================================================
# 12B. CHARTS
# =============================================================================

def combined_summary_path() -> Path:
    return TREND_OUT_ROOT / f"{PREFIX}_ALL_WINDOWS_summary_tables_{TIME_TAG}.csv"


def get_summary_masks() -> List[str]:
    """Return the configured Trend masks used in summaries and charts."""
    return list(SUMMARY_MASKS)


def get_available_summary_masks(all_df: pd.DataFrame) -> List[str]:
    """Return configured summary masks that are actually present in the table."""
    available = set(all_df["mask_name"].astype(str)) if "mask_name" in all_df.columns else set()
    return [m for m in get_summary_masks() if m in available]


def resolve_chart_main_mask(mask_order: List[str]) -> str | None:
    """Choose the configured main chart mask, with sensible current-mask fallbacks."""
    if not mask_order:
        return None

    if CHART_MAIN_MASK in mask_order:
        return CHART_MAIN_MASK

    for candidate in [
        "DegrPMKNW_neg",
        "ConfirmPMKNW_all",
        "DegrMKNW_neg",
        "DegrPNW_neg",
        "DegrPMK_neg",
        "DegrNW_neg",
        "DegrMK_neg",
        "DegrP_neg",
        "DegrPotential",
    ]:
        if candidate in mask_order:
            print(f"[charts] CHART_MAIN_MASK={CHART_MAIN_MASK} not available; using {candidate}.")
            return candidate

    return mask_order[-1]


def read_trend_summary_table() -> pd.DataFrame:
    """
    Read the combined Trend summary table.

    If the combined table does not exist, available per-window summary tables
    are concatenated. The archive workflow expects current mask names only.
    """
    combined = combined_summary_path()

    if combined.exists():
        print(f"[charts] reading combined summary: {combined}")
        df = pd.read_csv(combined)
    else:
        frames = []
        for window_name in TREND_WINDOWS:
            p = get_summary_dir(window_name) / f"{PREFIX}_{window_name}_summary_tables_{TIME_TAG}.csv"
            if p.exists():
                print(f"[charts] reading window summary: {p}")
                frames.append(pd.read_csv(p))
            else:
                print(f"[charts warning] missing window summary: {p}")

        if not frames:
            raise FileNotFoundError("No trend summary table found. Run RUN_SUMMARY_TABLES=True first.")

        df = pd.concat(frames, ignore_index=True)

    if "total_area_basis" not in df.columns:
        df["total_area_basis"] = "selected_ANALYSIS_LULC_GROUPS_assumed"
        print("[charts warning] total_area_basis column missing. If this table was created before the Block 12 denominator fix, rerun RUN_SUMMARY_TABLES=True.")

    int_cols = ["ECO_ID", "ECO_NAME_CODE", "COUNTRY_ID", "LULC_group", "LULC_stability_class"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    num_cols = ["total_pixels", "degraded_pixels", "pixel_area_ha", "total_ha", "degraded_ha", "degraded_pct"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_chart_if_needed(fig, out_png: Path, dpi: int = 800) -> None:
    mkdir(out_png.parent)

    if out_png.exists() and not FORCE_REBUILD_CHARTS:
        print(f"[skip chart] {out_png}")
        plt.close(fig)
        return

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[write chart] {out_png}")


def aggregate_trend_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate degraded and total area, then calculate degraded percentage using the corrected selected-LULC denominator."""
    if df.empty:
        return pd.DataFrame()

    required = ["total_ha", "degraded_ha"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[charts] missing required columns for aggregation: {missing}")
        return pd.DataFrame()

    out = df.groupby(group_cols, dropna=False)[["total_ha", "degraded_ha"]].sum().reset_index()
    out["degraded_pct"] = np.where(out["total_ha"] > 0, out["degraded_ha"] / out["total_ha"] * 100.0, 0.0)

    return out


def plot_grouped_degraded_percent(df: pd.DataFrame, group_col: str, bar_col: str, out_png: Path, title: str, xlabel: str, ylabel: str = "Degraded share of selected vegetated/reporting LULC area (%)", group_order: List[str] | None = None, bar_order: List[str] | None = None, figsize: Tuple[float, float] = (14, 7), rotation: int = 90) -> None:
    """Grouped bar chart for degraded percentage using the corrected selected-LULC denominator."""
    if df.empty:
        print(f"[charts] empty dataframe, skipping: {out_png.name}")
        return

    required = [group_col, bar_col, "degraded_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[charts] missing columns {missing}, skipping: {out_png.name}")
        return

    tmp = df.copy()
    tmp[group_col] = tmp[group_col].astype(str)
    tmp[bar_col] = tmp[bar_col].astype(str)

    group_order = list(tmp[group_col].drop_duplicates()) if group_order is None else [str(x) for x in group_order]
    bar_order = list(tmp[bar_col].drop_duplicates()) if bar_order is None else [str(x) for x in bar_order]

    idx = pd.MultiIndex.from_product([group_order, bar_order], names=[group_col, bar_col])
    plot_df = tmp.set_index([group_col, bar_col]).reindex(idx)[["degraded_pct"]].fillna(0.0).reset_index()

    x = np.arange(len(group_order))
    n_bars = max(1, len(bar_order))
    total_width = 0.78
    bar_width = min(0.28, total_width / n_bars)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=figsize)

    for j, b in enumerate(bar_order):
        vals = plot_df[plot_df[bar_col] == b].set_index(group_col).reindex(group_order)["degraded_pct"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x + offsets[j], vals, width=bar_width, label=str(b), edgecolor="black", linewidth=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(title=bar_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")

    save_chart_if_needed(fig, out_png)


def plot_stacked_degraded_composition(df: pd.DataFrame, group_col: str, stack_col: str, value_col: str, out_png: Path, title: str, xlabel: str, group_order: List[str] | None = None, stack_order: List[str] | None = None, figsize: Tuple[float, float] = (12, 7), rotation: int = 45) -> None:
    """Stacked percentage chart based on degraded area composition."""
    if df.empty:
        print(f"[charts] empty dataframe, skipping: {out_png.name}")
        return

    tmp = df.copy()
    tmp[group_col] = tmp[group_col].astype(str)
    tmp[stack_col] = tmp[stack_col].astype(str)

    group_order = list(tmp[group_col].drop_duplicates()) if group_order is None else [str(x) for x in group_order]
    stack_order = list(tmp[stack_col].drop_duplicates()) if stack_order is None else [str(x) for x in stack_order]

    p = tmp.pivot_table(index=group_col, columns=stack_col, values=value_col, aggfunc="sum", fill_value=0.0).reindex(index=group_order, columns=stack_order).fillna(0.0)
    p_pct = p.div(p.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    p_pct = p_pct.fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    p_pct.plot(kind="bar", stacked=True, ax=ax, edgecolor="black", linewidth=0.2)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of degraded area (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(title=stack_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")

    save_chart_if_needed(fig, out_png)


def make_trend_charts(all_df: pd.DataFrame) -> None:
    """
    Generate chart set for continental, country, and ecoregion Trend results.

    Charts use the masks explicitly configured in SUMMARY_MASKS and the selected ANALYSIS_LULC_GROUPS denominator.
    """
    if all_df.empty:
        print("[charts] trend summary table is empty.")
        return

    chart_dir = mkdir(get_chart_dir())
    window_order = list(TREND_WINDOWS.keys())
    mask_order = get_available_summary_masks(all_df)
    main_mask = resolve_chart_main_mask(mask_order)

    if main_mask is None:
        print("[charts] no configured summary mask found for charting.")
        return

    sort_window = CHART_SORT_WINDOW if CHART_SORT_WINDOW in window_order else window_order[-1]

    cont = all_df[(all_df["summary_level"].astype(str) == "ecoregion") & (all_df["mask_name"].astype(str).isin(mask_order))].copy()
    cont = aggregate_trend_summary(cont, ["trend_window", "mask_name"])

    if not cont.empty:
        plot_grouped_degraded_percent(cont, group_col="trend_window", bar_col="mask_name", group_order=window_order, bar_order=mask_order, out_png=chart_dir / f"{PREFIX}_ALL_continental_mask_sensitivity_by_window_{TIME_TAG}.png", title=f"Continental NDVI degradation by trend window and confirmation mask ({TIME_TAG})", xlabel="Trend window", figsize=(13, 7), rotation=0)

    lulc = all_df[(all_df["summary_level"].astype(str) == "ecoregion_lulc_mode") & (all_df["mask_name"].astype(str) == main_mask)].copy()

    if not lulc.empty:
        ag = aggregate_trend_summary(lulc, ["trend_window", "LULC_group", "LULC_label"])
        rank = ag[ag["trend_window"].astype(str) == sort_window].sort_values("degraded_pct", ascending=False)
        lulc_order = rank["LULC_label"].astype(str).tolist() or sorted(ag["LULC_label"].astype(str).unique())
        plot_grouped_degraded_percent(ag, group_col="LULC_label", bar_col="trend_window", group_order=lulc_order, bar_order=window_order, out_png=chart_dir / f"{PREFIX}_ALL_LULC_grouped_windows_{main_mask}_{TIME_TAG}.png", title=f"NDVI degradation by selected LULC group and trend window — {main_mask}", xlabel="Selected LULC group", figsize=(13, 7), rotation=45)

    country = all_df[(all_df["summary_level"].astype(str) == "country") & (all_df["mask_name"].astype(str) == main_mask)].copy()

    if not country.empty:
        ag = aggregate_trend_summary(country, ["COUNTRY_ID", "ISO_A3", "trend_window"])
        ag = ag[ag["ISO_A3"].astype(str).str.len() > 0]
        rank = ag[ag["trend_window"].astype(str) == sort_window].sort_values("degraded_pct", ascending=False)
        country_order = rank["ISO_A3"].astype(str).tolist()
        if TOP_N_COUNTRIES is not None:
            country_order = country_order[:TOP_N_COUNTRIES]
        plot_grouped_degraded_percent(ag, group_col="ISO_A3", bar_col="trend_window", group_order=country_order, bar_order=window_order, out_png=chart_dir / f"{PREFIX}_ALL_country_grouped_windows_{main_mask}_{TIME_TAG}.png", title=f"Country ranking of NDVI degradation across trend windows — {main_mask}", xlabel="Country", figsize=(max(14, len(country_order) * 0.45), 8), rotation=90)

    eco = all_df[(all_df["summary_level"].astype(str) == "ecoregion") & (all_df["mask_name"].astype(str) == main_mask)].copy()

    if not eco.empty:
        ag = aggregate_trend_summary(eco, ["ECO_ID", "ECO_NAME_CODE", "ECO_NAME", "trend_window"])
        ag["eco_label"] = ag["ECO_NAME_CODE"].astype(str) + " | " + ag["ECO_NAME"].astype(str)
        rank = ag[ag["trend_window"].astype(str) == sort_window].sort_values("degraded_pct", ascending=False)
        eco_order = rank["eco_label"].astype(str).tolist()
        if TOP_N_ECOREGIONS is not None:
            eco_order = eco_order[:TOP_N_ECOREGIONS]
        plot_grouped_degraded_percent(ag, group_col="eco_label", bar_col="trend_window", group_order=eco_order, bar_order=window_order, out_png=chart_dir / f"{PREFIX}_ALL_ecoregion_grouped_windows_{main_mask}_{TIME_TAG}.png", title=f"Ecoregion ranking of NDVI degradation across trend windows — {main_mask}", xlabel="Ecoregion", figsize=(max(14, len(eco_order) * 0.55), 8), rotation=90)

    country_masks = all_df[(all_df["summary_level"].astype(str) == "country") & (all_df["trend_window"].astype(str) == sort_window) & (all_df["mask_name"].astype(str).isin(mask_order))].copy()

    if not country_masks.empty:
        ag = aggregate_trend_summary(country_masks, ["COUNTRY_ID", "ISO_A3", "mask_name"])
        ag = ag[ag["ISO_A3"].astype(str).str.len() > 0]
        if not country.empty:
            rank = aggregate_trend_summary(country[country["trend_window"].astype(str) == sort_window], ["ISO_A3"]).sort_values("degraded_pct", ascending=False)
            country_order = rank["ISO_A3"].astype(str).tolist()
        else:
            country_order = sorted(ag["ISO_A3"].astype(str).unique())
        if TOP_N_COUNTRIES is not None:
            country_order = country_order[:TOP_N_COUNTRIES]
        plot_grouped_degraded_percent(ag, group_col="ISO_A3", bar_col="mask_name", group_order=country_order, bar_order=mask_order, out_png=chart_dir / f"{PREFIX}_ALL_country_mask_sensitivity_{sort_window}_{TIME_TAG}.png", title=f"Country mask-sensitivity for NDVI degradation — {sort_window}", xlabel="Country", figsize=(max(14, len(country_order) * 0.45), 8), rotation=90)

    eco_masks = all_df[(all_df["summary_level"].astype(str) == "ecoregion") & (all_df["trend_window"].astype(str) == sort_window) & (all_df["mask_name"].astype(str).isin(mask_order))].copy()

    if not eco_masks.empty:
        ag = aggregate_trend_summary(eco_masks, ["ECO_ID", "ECO_NAME_CODE", "ECO_NAME", "mask_name"])
        ag["eco_label"] = ag["ECO_NAME_CODE"].astype(str) + " | " + ag["ECO_NAME"].astype(str)
        rank = ag[ag["mask_name"].astype(str) == main_mask].sort_values("degraded_pct", ascending=False)
        eco_order = rank["eco_label"].astype(str).tolist()
        if TOP_N_ECOREGIONS is not None:
            eco_order = eco_order[:TOP_N_ECOREGIONS]
        plot_grouped_degraded_percent(ag, group_col="eco_label", bar_col="mask_name", group_order=eco_order, bar_order=mask_order, out_png=chart_dir / f"{PREFIX}_ALL_ecoregion_mask_sensitivity_{sort_window}_{TIME_TAG}.png", title=f"Ecoregion mask-sensitivity for NDVI degradation — {sort_window}", xlabel="Ecoregion", figsize=(max(14, len(eco_order) * 0.55), 8), rotation=90)

    validation = all_df[(all_df["summary_level"].astype(str) == "ecoregion_lulc_stability") & (all_df["mask_name"].astype(str) == main_mask)].copy()

    if not validation.empty:
        ag = validation.groupby(["trend_window", "LULC_stability_class"], dropna=False)["degraded_ha"].sum().reset_index()
        ag["validation_label"] = ag["LULC_stability_class"].map(SHORT_GENUINITY_LABELS).fillna(ag["LULC_stability_class"].astype(str) + " | unclassified transition level")
        validation_order = ag[["LULC_stability_class", "validation_label"]].drop_duplicates().sort_values("LULC_stability_class")["validation_label"].astype(str).tolist()
        plot_stacked_degraded_composition(ag, group_col="trend_window", stack_col="validation_label", value_col="degraded_ha", group_order=window_order, stack_order=validation_order, out_png=chart_dir / f"{PREFIX}_ALL_degraded_area_validation_composition_{main_mask}_{TIME_TAG}.png", title=f"Composition of degraded area by LULC-transition validation class — {main_mask}", xlabel="Trend window", figsize=(13, 7), rotation=0)

    print(f"[charts] outputs saved under: {chart_dir}")


def make_country_detail_charts(all_df: pd.DataFrame) -> None:
    """
    Create detailed Trend charts for each individual country using configured masks only.
    """
    if all_df.empty:
        print("[country charts] trend summary table is empty.")
        return

    chart_root = mkdir(get_chart_dir() / "country_details")
    window_order = list(TREND_WINDOWS.keys())
    mask_order = get_available_summary_masks(all_df)
    main_mask = resolve_chart_main_mask(mask_order)
    sort_window = CHART_SORT_WINDOW if CHART_SORT_WINDOW in window_order else window_order[-1]

    if main_mask is None:
        print("[country charts] no available configured mask found for charting.")
        return

    country_base = all_df[all_df["summary_level"].astype(str) == "country"].copy()
    if country_base.empty or "ISO_A3" not in country_base.columns:
        print("[country charts] no country-level summary rows found.")
        return

    all_codes = country_base["ISO_A3"].dropna().astype(str).str.strip()
    all_codes = [c for c in sorted(all_codes.unique()) if c]

    if COUNTRY_DETAIL_CODES:
        wanted = {str(c).strip() for c in COUNTRY_DETAIL_CODES if str(c).strip()}
        country_codes = [c for c in all_codes if c in wanted]
    else:
        country_codes = all_codes

    print(f"[country charts] countries to process: {len(country_codes)}")

    for iso in country_codes:
        cdir = mkdir(chart_root / iso)
        print(f"[country charts] {iso}")

        sub = all_df[(all_df["summary_level"].astype(str) == "country") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["mask_name"].astype(str).isin(mask_order))].copy()

        if not sub.empty:
            ag = aggregate_trend_summary(sub, ["trend_window", "mask_name"])
            plot_grouped_degraded_percent(ag, group_col="trend_window", bar_col="mask_name", group_order=window_order, bar_order=mask_order, out_png=cdir / f"{PREFIX}_{iso}_mask_sensitivity_by_window_{TIME_TAG}.png", title=f"{iso}: NDVI degradation by trend window and confirmation mask", xlabel="Trend window", figsize=(12, 7), rotation=0)

        sub = all_df[(all_df["summary_level"].astype(str) == "ecoregion_country") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["mask_name"].astype(str) == main_mask)].copy()

        if not sub.empty:
            ag = aggregate_trend_summary(sub, ["ECO_ID", "ECO_NAME_CODE", "ECO_NAME", "trend_window"])
            ag["eco_label"] = ag["ECO_NAME_CODE"].astype(str) + " | " + ag["ECO_NAME"].astype(str)
            rank = ag[ag["trend_window"].astype(str) == sort_window].sort_values("degraded_pct", ascending=False)
            eco_order = rank["eco_label"].astype(str).tolist()
            if TOP_N_ECOREGIONS_PER_COUNTRY is not None:
                eco_order = eco_order[:TOP_N_ECOREGIONS_PER_COUNTRY]
            plot_grouped_degraded_percent(ag, group_col="eco_label", bar_col="trend_window", group_order=eco_order, bar_order=window_order, out_png=cdir / f"{PREFIX}_{iso}_ecoregion_grouped_windows_{main_mask}_{TIME_TAG}.png", title=f"{iso}: ecoregion ranking across trend windows — {main_mask}", xlabel="Ecoregion", figsize=(max(12, len(eco_order) * 0.55), 8), rotation=90)

        sub = all_df[(all_df["summary_level"].astype(str) == "ecoregion_country") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["trend_window"].astype(str) == sort_window) & (all_df["mask_name"].astype(str).isin(mask_order))].copy()

        if not sub.empty:
            ag = aggregate_trend_summary(sub, ["ECO_ID", "ECO_NAME_CODE", "ECO_NAME", "mask_name"])
            ag["eco_label"] = ag["ECO_NAME_CODE"].astype(str) + " | " + ag["ECO_NAME"].astype(str)
            rank = ag[ag["mask_name"].astype(str) == main_mask].sort_values("degraded_pct", ascending=False)
            eco_order = rank["eco_label"].astype(str).tolist()
            if TOP_N_ECOREGIONS_PER_COUNTRY is not None:
                eco_order = eco_order[:TOP_N_ECOREGIONS_PER_COUNTRY]
            plot_grouped_degraded_percent(ag, group_col="eco_label", bar_col="mask_name", group_order=eco_order, bar_order=mask_order, out_png=cdir / f"{PREFIX}_{iso}_ecoregion_mask_sensitivity_{sort_window}_{TIME_TAG}.png", title=f"{iso}: ecoregion mask sensitivity — {sort_window}", xlabel="Ecoregion", figsize=(max(12, len(eco_order) * 0.55), 8), rotation=90)

        sub = all_df[(all_df["summary_level"].astype(str) == "ecoregion_country_lulc_mode") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["mask_name"].astype(str) == main_mask)].copy()

        if not sub.empty:
            ag = aggregate_trend_summary(sub, ["LULC_group", "LULC_label", "trend_window"])
            rank = ag[ag["trend_window"].astype(str) == sort_window].sort_values("degraded_pct", ascending=False)
            lulc_order = rank["LULC_label"].astype(str).tolist() or sorted(ag["LULC_label"].astype(str).unique())
            plot_grouped_degraded_percent(ag, group_col="LULC_label", bar_col="trend_window", group_order=lulc_order, bar_order=window_order, out_png=cdir / f"{PREFIX}_{iso}_LULC_grouped_windows_{main_mask}_{TIME_TAG}.png", title=f"{iso}: NDVI degradation by selected LULC group and trend window — {main_mask}", xlabel="Selected LULC group", figsize=(12, 7), rotation=45)

        sub = all_df[(all_df["summary_level"].astype(str) == "ecoregion_country_lulc_mode") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["trend_window"].astype(str) == sort_window) & (all_df["mask_name"].astype(str).isin(mask_order))].copy()

        if not sub.empty:
            ag = aggregate_trend_summary(sub, ["LULC_group", "LULC_label", "mask_name"])
            rank = ag[ag["mask_name"].astype(str) == main_mask].sort_values("degraded_pct", ascending=False)
            lulc_order = rank["LULC_label"].astype(str).tolist() or sorted(ag["LULC_label"].astype(str).unique())
            plot_grouped_degraded_percent(ag, group_col="LULC_label", bar_col="mask_name", group_order=lulc_order, bar_order=mask_order, out_png=cdir / f"{PREFIX}_{iso}_LULC_mask_sensitivity_{sort_window}_{TIME_TAG}.png", title=f"{iso}: selected-LULC mask sensitivity — {sort_window}", xlabel="Selected LULC group", figsize=(12, 7), rotation=45)

        sub = all_df[(all_df["summary_level"].astype(str) == "ecoregion_country_lulc_stability") & (all_df["ISO_A3"].astype(str) == iso) & (all_df["mask_name"].astype(str) == main_mask)].copy()

        if not sub.empty:
            ag = sub.groupby(["trend_window", "LULC_stability_class"], dropna=False)["degraded_ha"].sum().reset_index()
            ag["validation_label"] = ag["LULC_stability_class"].map(SHORT_GENUINITY_LABELS).fillna(ag["LULC_stability_class"].astype(str) + " | unclassified transition level")
            validation_order = ag[["LULC_stability_class", "validation_label"]].drop_duplicates().sort_values("LULC_stability_class")["validation_label"].astype(str).tolist()
            plot_stacked_degraded_composition(ag, group_col="trend_window", stack_col="validation_label", value_col="degraded_ha", group_order=window_order, stack_order=validation_order, out_png=cdir / f"{PREFIX}_{iso}_degraded_area_validation_composition_{main_mask}_{TIME_TAG}.png", title=f"{iso}: composition of degraded area by LULC-transition validation class — {main_mask}", xlabel="Trend window", figsize=(12, 7), rotation=0)

    print(f"[country charts] outputs saved under: {chart_root}")


def run_chart_only_mode() -> None:
    print("\n" + "=" * 100)
    print("[charts only] creating Trend charts from existing summary tables")
    print("=" * 100)

    all_df = read_trend_summary_table()
    make_trend_charts(all_df)

    if RUN_COUNTRY_DETAIL_CHARTS:
        make_country_detail_charts(all_df)



# =============================================================================
# 12C. TREND DIRECTION DATA PRODUCTS
# =============================================================================

TREND_DIRECTION_LABELS = {
    0: "NoData / excluded / insufficient trend information",
    1: "Strict confirmed degradation",
    2: "Neutral / no strict confirmed directional trend",
    3: "Strict confirmed improvement",
}

IMPROVEMENT_MASK_LABELS = {
    "ImproveP_pos": "Positive Trend confirmed by OLS p-value",
    "ImproveNW_pos": "Positive Trend confirmed by Newey-West p-value",
    "ImproveMK_pos": "Positive Trend confirmed by Mann-Kendall",
    "ImprovePNW_pos": "Positive Trend confirmed by OLS p-value and Newey-West",
    "ImprovePMK_pos": "Positive Trend confirmed by OLS p-value and Mann-Kendall",
    "ImproveMKNW_pos": "Positive Trend confirmed by Mann-Kendall and Newey-West",
    "ImprovePMKNW_pos": "Strict positive Trend confirmed by OLS, Mann-Kendall and Newey-West",
}


def trend_direction_product_dir() -> Path:
    return mkdir(TREND_OUT_ROOT / "DATA_PRODUCTS" / "TREND_DIRECTION" / TIME_TAG)


def trend_direction_product_path(window_name: str, metric: str) -> Path:
    return (
        trend_direction_product_dir()
        / f"{PREFIX}_{window_name}_{metric}_{TIME_TAG}_EA250m.tif"
    )


def trend_direction_required_inputs(window_name: str) -> Dict[str, Path]:
    """
    Return raster inputs needed to build positive-direction and three-class
    Trend products from the already-computed Trend statistics.
    """
    return {
        "slope": trend_raster_path(window_name, "NDVI_slope_per_year"),
        "p": trend_raster_path(window_name, "NDVI_p"),
        "nw_p": trend_raster_path(window_name, "NDVI_NeweyWest_p"),
        "mk_p": trend_raster_path(window_name, "NDVI_MK_p"),
        "mk_tau": trend_raster_path(window_name, "NDVI_MK_tau"),
        "lulc_mode": lulc_stability_path(window_name, "LULC_mode"),
        "strict_degradation": mask_raster_path(window_name, "DegrPMKNW_neg"),
    }


def trend_direction_outputs_complete(window_name: str) -> bool:
    required = [
        trend_direction_product_path(window_name, "ImprovePMKNW_pos"),
        trend_direction_product_path(window_name, "TrendDirection3Class_PMKNW"),
    ]

    if WRITE_COMPONENT_IMPROVEMENT_MASKS:
        required.extend(
            trend_direction_product_path(window_name, mask_name)
            for mask_name in IMPROVEMENT_MASK_LABELS
        )

    return all(_exists_and_nonempty(p) for p in required)


def write_trend_direction_lookups() -> None:
    product_dir = trend_direction_product_dir()

    direction_lookup = pd.DataFrame([
        {"class_code": code, "class_label": label}
        for code, label in TREND_DIRECTION_LABELS.items()
    ])
    save_csv(
        direction_lookup,
        product_dir / f"{PREFIX}_TrendDirection3Class_lookup_{TIME_TAG}.csv",
    )

    improvement_lookup = pd.DataFrame([
        {"mask_name": name, "mask_label": label}
        for name, label in IMPROVEMENT_MASK_LABELS.items()
    ])
    save_csv(
        improvement_lookup,
        product_dir / f"{PREFIX}_Improvement_mask_lookup_{TIME_TAG}.csv",
    )


def summarise_existing_direction_raster(window_name: str) -> pd.DataFrame:
    """
    Summarise an existing three-class direction raster without rebuilding it.
    """
    out_direction = trend_direction_product_path(
        window_name,
        "TrendDirection3Class_PMKNW",
    )

    if not _exists_and_nonempty(out_direction):
        return pd.DataFrame()

    pixel_ha = get_pixel_area_ha_from_profile(out_direction)
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    with rasterio.open(out_direction) as src:
        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            values, counts = np.unique(arr, return_counts=True)
            for value, count in zip(values, counts):
                value = int(value)
                if value in class_counts:
                    class_counts[value] += int(count)

    valid_pixels = class_counts[1] + class_counts[2] + class_counts[3]

    rows = []
    for class_code, pixels in class_counts.items():
        rows.append({
            "trend_window": window_name,
            "product_type": "TrendDirection3Class_PMKNW",
            "class_code": class_code,
            "class_label": TREND_DIRECTION_LABELS[class_code],
            "pixels": pixels,
            "area_ha": pixels * pixel_ha,
            "pct_of_valid_analysis_domain": (
                pixels / valid_pixels * 100.0
                if class_code != 0 and valid_pixels > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_trend_direction_products_for_window(window_name: str) -> pd.DataFrame:
    """
    Build strict confirmed-improvement masks and a three-class Trend direction
    product from the current Trend rasters.

    Positive-direction logic mirrors the strict negative logic:

        ImproveP_pos:
            OLS p < P_90 and slope >= +abs(SLOPE_TH_PHYSICAL)

        ImproveNW_pos:
            Newey-West p < P_90 and slope >= +abs(SLOPE_TH_PHYSICAL)

        ImproveMK_pos:
            Mann-Kendall p < P_90 and tau > 0

        ImprovePMKNW_pos:
            ImproveP_pos & ImproveMK_pos & ImproveNW_pos

    Three-class coding within the valid analysis domain:
        1 = strict confirmed degradation
        2 = neutral / no strict confirmed directional Trend
        3 = strict confirmed improvement

    A pixel is eligible for the three-class product only where the OLS slope is
    finite and the window-modal LULC belongs to ANALYSIS_LULC_GROUPS.
    """
    print("\n" + "=" * 100)
    print(f"[trend direction] building products for {window_name}")
    print("=" * 100)

    if (
        REUSE_EXISTING_DIRECTION_PRODUCTS
        and not FORCE_REBUILD_DIRECTION_PRODUCTS
        and trend_direction_outputs_complete(window_name)
    ):
        print(f"[reuse] {window_name}: Trend-direction products already complete.")
        return summarise_existing_direction_raster(window_name)

    paths = trend_direction_required_inputs(window_name)
    missing = [p for p in paths.values() if not _exists_and_nonempty(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required Trend-direction inputs for {window_name}:\n"
            + "\n".join(str(p) for p in missing)
        )

    template_profile = validate_alignment(list(paths.values()))
    width = template_profile["width"]
    height = template_profile["height"]
    profile_u8 = output_profile_like(template_profile, "uint8", UINT8_NODATA)
    pixel_ha = get_pixel_area_ha_from_profile(paths["slope"])

    out_strict_improve = trend_direction_product_path(
        window_name,
        "ImprovePMKNW_pos",
    )
    out_direction = trend_direction_product_path(
        window_name,
        "TrendDirection3Class_PMKNW",
    )

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    improvement_counts = {name: 0 for name in IMPROVEMENT_MASK_LABELS}

    with ExitStack() as stack:
        src_slope = stack.enter_context(rasterio.open(paths["slope"]))
        src_p = stack.enter_context(rasterio.open(paths["p"]))
        src_nw_p = stack.enter_context(rasterio.open(paths["nw_p"]))
        src_mk_p = stack.enter_context(rasterio.open(paths["mk_p"]))
        src_mk_tau = stack.enter_context(rasterio.open(paths["mk_tau"]))
        src_lulc = stack.enter_context(rasterio.open(paths["lulc_mode"]))
        src_degr = stack.enter_context(rasterio.open(paths["strict_degradation"]))

        dst_improve = stack.enter_context(
            rasterio.open(out_strict_improve, "w", **profile_u8)
        )
        dst_direction = stack.enter_context(
            rasterio.open(out_direction, "w", **profile_u8)
        )

        component_dsts = {}
        if WRITE_COMPONENT_IMPROVEMENT_MASKS:
            for mask_name in IMPROVEMENT_MASK_LABELS:
                out_path = trend_direction_product_path(window_name, mask_name)
                component_dsts[mask_name] = stack.enter_context(
                    rasterio.open(out_path, "w", **profile_u8)
                )

        windows = list(iter_windows(width, height, BLOCK_SIZE))
        n_windows = len(windows)
        t0 = time.perf_counter()

        for i, win in enumerate(windows, start=1):
            slope = _read_ndvi_window_as_float(src_slope, win)
            p = _read_ndvi_window_as_float(src_p, win)
            nw_p = _read_ndvi_window_as_float(src_nw_p, win)
            mk_p = _read_ndvi_window_as_float(src_mk_p, win)
            mk_tau = _read_ndvi_window_as_float(src_mk_tau, win)
            lulc = src_lulc.read(1, window=win).astype(np.uint8)
            strict_degr_existing = src_degr.read(1, window=win).astype(np.uint8)

            valid = np.isfinite(slope) & np.isin(lulc, ANALYSIS_LULC_GROUPS)

            improve_p = (
                valid
                & np.isfinite(p)
                & (p < P_90)
                & (slope >= SLOPE_TH_PHYSICAL_POS)
            )
            improve_nw = (
                valid
                & np.isfinite(nw_p)
                & (nw_p < P_90)
                & (slope >= SLOPE_TH_PHYSICAL_POS)
            )
            improve_mk = (
                valid
                & np.isfinite(mk_p)
                & (mk_p < P_90)
                & np.isfinite(mk_tau)
                & (mk_tau > 0)
            )

            improve_pnw = improve_p & improve_nw
            improve_pmk = improve_p & improve_mk
            improve_mknw = improve_mk & improve_nw
            improve_strict = improve_p & improve_mk & improve_nw

            strict_degr = valid & (strict_degr_existing == 1)

            direction = np.zeros(slope.shape, dtype=np.uint8)
            direction[valid] = 2
            direction[strict_degr] = 1
            direction[improve_strict] = 3

            overlap = strict_degr & improve_strict
            if np.any(overlap):
                # This should not occur because strict decline and improvement
                # require opposite slope directions. Neutral is used defensively.
                direction[overlap] = 2

            dst_improve.write(improve_strict.astype(np.uint8), 1, window=win)
            dst_direction.write(direction, 1, window=win)

            component_masks = {
                "ImproveP_pos": improve_p,
                "ImproveNW_pos": improve_nw,
                "ImproveMK_pos": improve_mk,
                "ImprovePNW_pos": improve_pnw,
                "ImprovePMK_pos": improve_pmk,
                "ImproveMKNW_pos": improve_mknw,
                "ImprovePMKNW_pos": improve_strict,
            }

            for mask_name, mask_bool in component_masks.items():
                improvement_counts[mask_name] += int(mask_bool.sum())
                if WRITE_COMPONENT_IMPROVEMENT_MASKS:
                    component_dsts[mask_name].write(
                        mask_bool.astype(np.uint8),
                        1,
                        window=win,
                    )

            values, counts = np.unique(direction, return_counts=True)
            for value, count in zip(values, counts):
                class_counts[int(value)] += int(count)

            if i % 50 == 0 or i == n_windows:
                print(
                    f"[trend direction {window_name}] "
                    f"{i:,}/{n_windows:,} windows | "
                    f"elapsed {_fmt(time.perf_counter() - t0)}"
                )

    valid_pixels = class_counts[1] + class_counts[2] + class_counts[3]
    rows = []

    for class_code, pixels in class_counts.items():
        rows.append({
            "trend_window": window_name,
            "product_type": "TrendDirection3Class_PMKNW",
            "class_code": class_code,
            "class_label": TREND_DIRECTION_LABELS[class_code],
            "pixels": pixels,
            "area_ha": pixels * pixel_ha,
            "pct_of_valid_analysis_domain": (
                pixels / valid_pixels * 100.0
                if class_code != 0 and valid_pixels > 0
                else np.nan
            ),
        })

    for mask_name, pixels in improvement_counts.items():
        rows.append({
            "trend_window": window_name,
            "product_type": "improvement_mask",
            "class_code": 1,
            "class_label": IMPROVEMENT_MASK_LABELS[mask_name],
            "mask_name": mask_name,
            "pixels": pixels,
            "area_ha": pixels * pixel_ha,
            "pct_of_valid_analysis_domain": (
                pixels / valid_pixels * 100.0 if valid_pixels > 0 else np.nan
            ),
        })

    print(f"[write] {out_strict_improve}")
    print(f"[write] {out_direction}")
    return pd.DataFrame(rows)


def build_all_trend_direction_products() -> pd.DataFrame:
    """
    Build/reuse Trend-direction products for all configured Trend windows and
    write one combined area-summary table.
    """
    write_trend_direction_lookups()

    parts = [
        build_trend_direction_products_for_window(window_name)
        for window_name in TREND_WINDOWS
    ]
    parts = [df for df in parts if df is not None and not df.empty]
    summary = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    out_csv = (
        trend_direction_product_dir()
        / f"{PREFIX}_TrendDirection3Class_area_summary_{TIME_TAG}.csv"
    )
    save_csv(summary, out_csv)
    return summary


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    T0 = time.perf_counter()
    mkdir(TREND_OUT_ROOT)

    print("\n" + "#" * 100)
    print("CONTINENTAL NDVI TREND ASSESSMENT FROM STATE CACHE")
    print("#" * 100)
    print(f"Start time: {_now()}")
    print(f"STATE_CACHE_DIR: {STATE_CACHE_DIR}")
    print(f"TREND_OUT_ROOT: {TREND_OUT_ROOT}")
    print(f"TIME_TAG: {TIME_TAG}")
    print(f"TREND_WINDOWS: {TREND_WINDOWS}")
    print(f"RUN_BUILD_TREND_RASTERS: {RUN_BUILD_TREND_RASTERS}")
    print(f"RUN_SUMMARY_TABLES: {RUN_SUMMARY_TABLES}")
    print(f"RUN_CHARTS_ONLY: {RUN_CHARTS_ONLY}")
    print(f"RUN_CHARTS: {RUN_CHARTS}")
    print(f"REUSE_EXISTING_TREND_RASTERS: {REUSE_EXISTING_TREND_RASTERS}")
    print(f"FORCE_REBUILD_TREND_RASTERS: {FORCE_REBUILD_TREND_RASTERS}")
    print(f"FORCE_REBUILD_CHARTS: {FORCE_REBUILD_CHARTS}")
    print(f"SCIPY_ERFC_AVAILABLE: {SCIPY_ERFC_AVAILABLE}")
    print(f"SUMMARY_MASKS: {get_summary_masks()}")
    print(f"RUN_TREND_DIRECTION_PRODUCTS: {RUN_TREND_DIRECTION_PRODUCTS}")
    print(f"ANALYSIS_LULC_GROUPS: {ANALYSIS_LULC_GROUPS}")
    print("#" * 100)

    if RUN_CHARTS_ONLY:
        run_chart_only_mode()
        print("\n" + "#" * 100)
        print("TREND CHART-ONLY RUN FINISHED")
        print("#" * 100)
        print(f"End time: {_now()}")
        print(f"Elapsed: {_fmt(time.perf_counter() - T0)}")
        print(f"Outputs: {TREND_OUT_ROOT}")
        print("#" * 100)
        return

    validate_cache_inputs()

    analysis_lulc_labels = [LULC_GROUP_LABELS.get(k, str(k)) for k in ANALYSIS_LULC_GROUPS]

    config = pd.DataFrame([
        ("TIME_TAG", TIME_TAG),
        ("STATE_CACHE_DIR", STATE_CACHE_DIR),
        ("TREND_OUT_ROOT", TREND_OUT_ROOT),
        ("TREND_WINDOWS", TREND_WINDOWS),
        ("SLOPE_TH_PHYSICAL", SLOPE_TH_PHYSICAL),
        ("R2_THRESHOLD", R2_THRESHOLD),
        ("P_90", P_90),
        ("NW_MAX_LAG", NW_MAX_LAG),
        ("MIN_VALID_OBS_FRACTION", MIN_VALID_OBS_FRACTION),
        ("MIN_VALID_OBS_ABSOLUTE", MIN_VALID_OBS_ABSOLUTE),
        ("MIN_VALID_OBS_BY_WINDOW", {k: min_valid_obs_for_window(len(v)) for k, v in TREND_WINDOWS.items()}),
        ("OLS_P_VALUE_REFERENCE", "two-sided finite-sample Student t, df=n-2"),
        ("NEWEY_WEST_P_VALUE_REFERENCE", "two-sided asymptotic standard normal"),
        ("MANN_KENDALL_VARIANCE", "continuity-corrected S; no additional tie correction"),
        ("STRICT_NEGATIVE_MASK", "DegrPMKNW_neg = DegrP_neg & DegrMK_neg & DegrNW_neg"),
        ("SUMMARY_DENOMINATOR_NOTE", "selected ANALYSIS_LULC_GROUPS; not additionally restricted by NDVI_valid_n"),
        ("STRICT_STABILITY_PROP", STRICT_STABILITY_PROP),
        ("HIGH_STABILITY_PROP", HIGH_STABILITY_PROP),
        ("MODERATE_STABILITY_PROP", MODERATE_STABILITY_PROP),
        ("RUN_BUILD_TREND_RASTERS", RUN_BUILD_TREND_RASTERS),
        ("RUN_SUMMARY_TABLES", RUN_SUMMARY_TABLES),
        ("RUN_CHARTS_ONLY", RUN_CHARTS_ONLY),
        ("RUN_CHARTS", RUN_CHARTS),
        ("RUN_COUNTRY_DETAIL_CHARTS", RUN_COUNTRY_DETAIL_CHARTS),
        ("COUNTRY_DETAIL_CODES", COUNTRY_DETAIL_CODES),
        ("FORCE_REBUILD_CHARTS", FORCE_REBUILD_CHARTS),
        ("CHART_MAIN_MASK", CHART_MAIN_MASK),
        ("CHART_MAIN_MASK_RESOLVED_RULE", "configured mask with current-mask fallback"),
        ("CHART_SORT_WINDOW", CHART_SORT_WINDOW),
        ("SUMMARY_MASKS", get_summary_masks()),
        ("SUMMARY_TOTAL_AREA_BASIS", "selected_ANALYSIS_LULC_GROUPS"),
        ("ANALYSIS_LULC_GROUPS", ANALYSIS_LULC_GROUPS),
        ("ANALYSIS_LULC_LABELS", analysis_lulc_labels),
        ("REUSE_EXISTING_TREND_RASTERS", REUSE_EXISTING_TREND_RASTERS),
        ("FORCE_REBUILD_TREND_RASTERS", FORCE_REBUILD_TREND_RASTERS),
        ("RUN_TREND_DIRECTION_PRODUCTS", RUN_TREND_DIRECTION_PRODUCTS),
        ("WRITE_COMPONENT_IMPROVEMENT_MASKS", WRITE_COMPONENT_IMPROVEMENT_MASKS),
        ("REUSE_EXISTING_DIRECTION_PRODUCTS", REUSE_EXISTING_DIRECTION_PRODUCTS),
        ("FORCE_REBUILD_DIRECTION_PRODUCTS", FORCE_REBUILD_DIRECTION_PRODUCTS),
        ("SLOPE_TH_PHYSICAL_POS", SLOPE_TH_PHYSICAL_POS),
        ("BLOCK_SIZE", BLOCK_SIZE),
    ], columns=["setting", "value"])

    save_csv(config, TREND_OUT_ROOT / f"{PREFIX}_run_configuration_{TIME_TAG}.csv")

    if RUN_BUILD_TREND_RASTERS:
        for window_name, years in TREND_WINDOWS.items():
            build_trend_rasters_for_window(window_name, years)

    if RUN_TREND_DIRECTION_PRODUCTS:
        build_all_trend_direction_products()

    if RUN_SUMMARY_TABLES:
        summary_tables = []
        summary_status_rows = []

        print("\n" + "#" * 100)
        print("SUMMARY TABLE GENERATION")
        print("#" * 100)

        for window_name in TREND_WINDOWS:
            print("\n" + "=" * 100)
            print(f"[summary main] starting: {window_name}")
            print("=" * 100)

            t_summary = time.perf_counter()

            try:
                df_summary = summarise_window(window_name)
                n_rows = 0 if df_summary is None else len(df_summary)
                status = "ok_empty" if n_rows == 0 else "ok"

                summary_status_rows.append({"trend_window": window_name, "status": status, "rows": n_rows, "elapsed": _fmt(time.perf_counter() - t_summary), "error_type": "", "error_message": ""})

                if df_summary is not None and not df_summary.empty:
                    summary_tables.append(df_summary)

                print(f"[summary main] finished: {window_name} | rows={n_rows:,} | status={status}")

            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e)

                summary_status_rows.append({"trend_window": window_name, "status": "failed", "rows": 0, "elapsed": _fmt(time.perf_counter() - t_summary), "error_type": err_type, "error_message": err_msg})

                print(f"[summary main ERROR] {window_name} failed: {err_type}: {err_msg}")
                traceback.print_exc()

        summary_status = pd.DataFrame(summary_status_rows)
        summary_status_csv = TREND_OUT_ROOT / f"{PREFIX}_summary_generation_status_{TIME_TAG}.csv"
        save_csv(summary_status, summary_status_csv)

        if summary_tables:
            all_summary = pd.concat(summary_tables, ignore_index=True)
            combined_summary_csv = TREND_OUT_ROOT / f"{PREFIX}_ALL_WINDOWS_summary_tables_{TIME_TAG}.csv"
            save_csv(all_summary, combined_summary_csv)
            print(f"[summary main] combined summary rows: {len(all_summary):,}")
            print(f"[summary main] combined summary table: {combined_summary_csv}")

        else:
            print("[summary main warning] No non-empty summary tables were produced.")

        failed = summary_status[summary_status["status"] == "failed"]
        if not failed.empty:
            print("\n[summary main warning] Some windows failed. See status CSV:")
            print(summary_status_csv)
            print(failed[["trend_window", "error_type", "error_message"]].to_string(index=False))
            raise RuntimeError("One or more summary windows failed. See summary_generation_status CSV for details.")

    if RUN_CHARTS:
        all_df_for_charts = read_trend_summary_table()
        make_trend_charts(all_df_for_charts)

    print("\n" + "#" * 100)
    print("TREND ASSESSMENT FINISHED")
    print("#" * 100)
    print(f"End time: {_now()}")
    print(f"Elapsed: {_fmt(time.perf_counter() - T0)}")
    print(f"Outputs: {TREND_OUT_ROOT}")
    print("#" * 100)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
# %%
