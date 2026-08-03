# %%
"""
Author: Evariste Rutebuka
Date: 2025-2026

MODIS NDVI downloader using ORNL API
Robust Africa/full-AOI version with empty-subset NoData tile handling.

Background
----------
This script acquires the MODIS MOD13Q1 NDVI observations used as the
Earth-observation productivity input for the subsequent State, Trend and
Performance workflow. It performs data acquisition and mosaicking only.
Quality screening, annual NDVI aggregation, common-grid alignment and STP
construction are handled in later workflow stages.

Main purpose
------------
- Download MOD13Q1 NDVI from ORNL API.
- Build a request grid over a large AOI, such as Africa.
- Download all request tiles for each MODIS acquisition date.
- Convert ORNL JSON response to Int16 GeoTIFF where value = NDVI x 10000.
- Mosaic all request tiles into one final AOI raster per MODIS date.
- Resume interrupted runs safely.
- Stop before mosaicking if any required tile is missing.
- Export request-footprint diagnostics before downloading.
- If ORNL returns metadata but subset = [], create a valid all-NoData tile.

Key steps
---------
1. Read the AOI and build an overlapping request grid in the native MODIS
   sinusoidal coordinate system.
2. Verify that the request footprints cover the AOI before downloading.
3. Query available MOD13Q1 dates within the requested period.
4. Download all request subsets for each date with retries for failures.
5. Convert returned NDVI values to Int16 NDVI x 10000 and preserve NoData.
6. Stop if any required tile remains unavailable after retry passes.
7. Mosaic the date-specific request tiles and crop to the AOI.
8. Validate the completed raster and write a completion marker.
9. Remove temporary tiles unless KEEP_TILES = True.

Important
---------
- The request grid is an acquisition device only; it is not an analytical
  spatial unit used by the State, Trend or Performance calculations.
- MOD13Q1 quality-assurance filtering is NOT performed in this script and
  must be applied in the later preprocessing stage before annual aggregation.

Output
------
For each MODIS date:
    <MODIS_OUT>/<BAND>_int1e4_<YYYY-MM-DD>.tif

Each output pixel is:
    Int16 NDVI x 10000
    Example: NDVI = 0.4586 is stored as 4586

Expected valid stored MOD13Q1 NDVI range:
    -2000 to 10000

NoData:
    -32768

"""
#%%
# ============================================================
# 0. imports
# ============================================================

import os
import shutil
import json
import time
import math
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import requests
from osgeo import gdal, osr
from shapely.geometry import Point, Polygon

gdal.UseExceptions()


# ============================================================
# 1. User settings — edit these
# ============================================================

MAIN_FOLDER = r"C:\PATH\TO\Paper\Codes\Africa"
COMMON_AOI_INPUTS = os.path.join(MAIN_FOLDER, "Polygons")
AOI_BUFFERED = os.path.join(COMMON_AOI_INPUTS, "Africa.gpkg") # this is buffered Africa boundary. 
AOI_LAYER: Optional[str] = None
MODIS_OUT = os.path.join(MAIN_FOLDER, "modis_download", "NDVI")

PRODUCT = "MOD13Q1"
BAND = "250m_16_days_NDVI"
START_DATE = "2001-01-01"
END_DATE   = "2022-12-31"


GRID_STEP_M = 100_000
KM_ABOVE_BELOW = 55
KM_LEFT_RIGHT = 55
MAX_WORKERS = 38 
KEEP_TILES = False     
CROP_TO_AOI = True
APPLY_SCALE = True
TILE_MAX_RETRIES = 4
MISSING_TILE_RETRY_PASSES = 2
STOP_ON_FAILED_TILES = True
CREATE_NODATA_TILE_FOR_EMPTY_SUBSET = True
STOP_IF_GRID_UNCOVERED = True
COVERAGE_TOLERANCE_KM2 = 1.0
REUSE_UNVERIFIED_OUTPUT = False
PROGRESS_EVERY = 250
APPROX_MODIS_DATES_2000_2025 = 596
EXPORT_GRID_FOOTPRINTS = True
MOVE_EDGE_POINTS_INSIDE_AOI = False


# ============================================================
# 2. Constants
# ============================================================

HEADER = {
    "Accept": "application/json",
    "User-Agent": "LandDegradationMODISDownloader/1.0"
}
# request-grid construction and downloaded-tile georeferencing.
PROJ4_SINUSOIDAL = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

OUT_MULT = 10_000
OUT_NODATA = -32768
EXPECTED_MODIS_SCALE = 0.0001
VALID_STORED_MIN = -2000
VALID_STORED_MAX = 10000
# Temporary tiles: lighter compression because they are deleted after mosaic in production.
TEMP_TILE_CO = [
    "COMPRESS=LZW",
    "PREDICTOR=2",
    "TILED=YES",
    "BIGTIFF=IF_SAFER"
]
# Final mosaic: stronger compression because these are retained.
FINAL_CO = [
    "COMPRESS=ZSTD",
    "ZSTD_LEVEL=12",
    "PREDICTOR=2",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "BIGTIFF=YES"
]

# ============================================================
# 3. Utility functions
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)

def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_vector(path: str) -> gpd.GeoDataFrame:
    if AOI_LAYER:
        return gpd.read_file(path, layer=AOI_LAYER)
    return gpd.read_file(path)

def output_done_marker(out_tif: str) -> str:
    return out_tif + ".done.json"

def is_completed_output(out_tif: str) -> bool:
    marker = output_done_marker(out_tif)
    if not (
        os.path.exists(out_tif)
        and os.path.getsize(out_tif) > 0
        and os.path.exists(marker)
    ):
        return False
    
    return ndvi_raster_is_valid(out_tif)

def mark_completed_output(
    out_tif: str,
    calendar_date: str,
    num_tiles: int,
    round_secs: float
) -> None:
    marker = {
        "output": out_tif,
        "calendar_date": calendar_date,
        "num_tiles": int(num_tiles),
        "output_size_gb": bytes_to_gb(os.path.getsize(out_tif)),
        "elapsed_seconds": float(round_secs),
        "elapsed_minutes": float(round_secs / 60),
        "completed_at": now_str(),
        "note": "Completed output structurally validated. NDVI stored as Int16 values = NDVI x 10000."
    }
    save_json(output_done_marker(out_tif), marker)


def tile_path(tiles_folder: str, tile_id: int) -> str:
    return os.path.join(tiles_folder, f"tile_{tile_id}.tif")


def tile_is_valid(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def ndvi_raster_is_valid(path: str) -> bool:
    """Lightweight structural validation for completed NDVI GeoTIFFs."""
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None or ds.RasterCount != 1:
            return False
        if ds.RasterXSize <= 0 or ds.RasterYSize <= 0:
            return False
        if not ds.GetProjection():
            return False

        band = ds.GetRasterBand(1)
        if band is None or band.DataType != gdal.GDT_Int16:
            return False
        if band.GetNoDataValue() != OUT_NODATA:
            return False

        ds = None
        return True
    except Exception:
        return False

def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))

def delete_shapefile_sidecars(path: str) -> None:
    base = os.path.splitext(path)[0]
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        f = base + ext
        if os.path.exists(f):
            os.remove(f)

def delete_if_exists(path: str) -> None:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)

def geometry_union(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.unary_union


# ============================================================
# 4. Main
# ============================================================

def main():
    run_start_dt = datetime.now()
    run_start = time.perf_counter()

    print("\n" + "=" * 78)
    print("MODIS NDVI ORNL downloader — robust Africa/full-AOI version")
    print("=" * 78)
    print(f"START TIME: {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AOI: {AOI_BUFFERED}")
    print(f"AOI_LAYER: {AOI_LAYER}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output folder: {MODIS_OUT}")
    print("=" * 78 + "\n")

    grid_shp = os.path.join(MAIN_FOLDER, "aoi", "aoi_grid.shp")
    os.makedirs(os.path.dirname(grid_shp), exist_ok=True)

    grid = make_data_request_grid(
        aoi_shapefile=AOI_BUFFERED,
        step_size=GRID_STEP_M,
        target_file=grid_shp,
        exclude_nonoverlapping=True,
        move_edge_points_inside_aoi=MOVE_EDGE_POINTS_INSIDE_AOI,
        export_footprints=EXPORT_GRID_FOOTPRINTS,
    )

    if grid.empty:
        raise RuntimeError("Grid is empty. Check AOI path/layer/CRS.")

    os.makedirs(MODIS_OUT, exist_ok=True)

    download_modis_data(
        grid_points_shapefile=grid_shp,
        start_date=START_DATE,
        end_date=END_DATE,
        product=PRODUCT,
        band=BAND,
        target_folder=MODIS_OUT,
        kmAboveBelow=KM_ABOVE_BELOW,
        kmLeftRight=KM_LEFT_RIGHT,
        apply_scale=APPLY_SCALE,
        keep_tiles=KEEP_TILES,
        crop=CROP_TO_AOI,
        aoi_vector=AOI_BUFFERED,
    )

    run_end_dt = datetime.now()
    total_secs = time.perf_counter() - run_start

    print("\n" + "=" * 78)
    print("RUN TIMING")
    print("=" * 78)
    print(f"START TIME: {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"END TIME:   {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ELAPSED:    {timedelta(seconds=int(total_secs))}")
    print("=" * 78)

# ============================================================
# 5. Download + mosaic
# ============================================================
def download_modis_data(
    grid_points_shapefile: str,
    start_date: str,
    end_date: str,
    product: str,
    band: str,
    target_folder: str,
    kmAboveBelow: int = 55,
    kmLeftRight: int = 55,
    apply_scale: bool = True,
    keep_tiles: bool = False,
    redownload: bool = False,
    **kwargs
):
    grid_points = gpd.read_file(grid_points_shapefile).to_crs("EPSG:4326").reset_index(drop=True)
    num_tiles = len(grid_points)

    print(f"[setup] grid tiles: {num_tiles} | MAX_WORKERS={MAX_WORKERS}")
    print(
        f"[setup] grid step: {GRID_STEP_M:,} m | "
        f"request half-size: {kmAboveBelow} km vertical, {kmLeftRight} km horizontal"
    )

    xmin, ymin, xmax, ymax = grid_points.total_bounds
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2)

    resp = requests.get(
        f"https://modis.ornl.gov/rst/api/v1/{product}/dates"
        f"?longitude={center[0]}&latitude={center[1]}",
        headers=HEADER,
        timeout=60
    )
    resp.raise_for_status()
    available_dates = resp.json().get("dates", [])

    dates = [d for d in available_dates if start_date <= d["calendar_date"] <= end_date]

    if not dates:
        print("[setup] No MODIS dates found in requested range.")
        return

    print(f"[setup] MODIS dates in range: {len(dates)}")
    for d in dates:
        print(f"        {d['calendar_date']} | {d['modis_date']}")

    target_crs = None
    aoi_vector = kwargs.get("aoi_vector")
    crop = bool(kwargs.get("crop", True))

    if crop and aoi_vector:
        aoi_gdf = read_vector(aoi_vector)
        if aoi_gdf.crs:
            epsg = aoi_gdf.crs.to_epsg()
            target_crs = f"EPSG:{epsg}" if epsg else aoi_gdf.crs.to_wkt()
        else:
            target_crs = "EPSG:4326"

    batch_start_dt = datetime.now()
    batch_start = time.perf_counter()

    completed_outputs = []

    for d in dates:
        modis_date = d["modis_date"]
        calendar_date = d["calendar_date"]

        out_tif = os.path.join(target_folder, f"{band}_int1e4_{calendar_date}.tif")
        tmp_out_tif = out_tif + ".tmp.tif"

        round_start_dt = datetime.now()
        round_start = time.perf_counter()

        print("\n" + "-" * 78)
        print(f"[{calendar_date}] START TIME: {round_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[{calendar_date}] START | workers={MAX_WORKERS} | tiles={num_tiles}")
        print("-" * 78)

        if not redownload and is_completed_output(out_tif):
            print(f"[{calendar_date}] SKIP | completed output exists: {out_tif}")
            completed_outputs.append(out_tif)
            continue

        if os.path.exists(out_tif) and not is_completed_output(out_tif):
            if REUSE_UNVERIFIED_OUTPUT:
                print(
                    f"[{calendar_date}] WARNING | output exists but no .done marker; "
                    "reusing because REUSE_UNVERIFIED_OUTPUT=True"
                )
                completed_outputs.append(out_tif)
                continue
            old_name = out_tif + f".unverified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
            print(f"[{calendar_date}] WARNING | output exists but no .done marker; renaming to:")
            print(f"                 {old_name}")
            os.replace(out_tif, old_name)

        if os.path.exists(tmp_out_tif):
            os.remove(tmp_out_tif)

        tiles_folder = os.path.join(target_folder, calendar_date)
        os.makedirs(tiles_folder, exist_ok=True)

        # ----------------------------------------------------
        # download phase with retry passes
        # ----------------------------------------------------
        dl_start = time.perf_counter()
        failed_records_all = []

        for pass_id in range(1, MISSING_TILE_RETRY_PASSES + 2):
            missing_ids = [
                i for i in range(num_tiles)
                if not tile_is_valid(tile_path(tiles_folder, i))
            ]

            if not missing_ids:
                print(f"[{calendar_date}] all tiles already available before pass {pass_id}")
                break

            if pass_id == 1:
                print(f"[{calendar_date}] download pass {pass_id}: requesting {len(missing_ids)} tiles")
            else:
                print(f"[{calendar_date}] retry pass {pass_id}: requesting {len(missing_ids)} missing tiles")

            pass_failed = download_tile_batch(
                grid_points=grid_points,
                tile_ids=missing_ids,
                tiles_folder=tiles_folder,
                product=product,
                band=band,
                modis_date=modis_date,
                kmAboveBelow=kmAboveBelow,
                kmLeftRight=kmLeftRight,
                apply_scale=apply_scale,
                calendar_date=calendar_date,
                pass_id=pass_id
            )

            failed_records_all.extend(pass_failed)

            remaining_missing = [
                i for i in range(num_tiles)
                if not tile_is_valid(tile_path(tiles_folder, i))
            ]

            print(f"[{calendar_date}] after pass {pass_id}: remaining missing tiles = {len(remaining_missing)}")

            if not remaining_missing:
                break

            if pass_id <= MISSING_TILE_RETRY_PASSES:
                sleep_seconds = 10 * pass_id
                print(f"[{calendar_date}] sleeping {sleep_seconds}s before next retry pass")
                sleep(sleep_seconds)

        dl_secs = time.perf_counter() - dl_start

        tile_files = [
            os.path.join(tiles_folder, fn)
            for fn in os.listdir(tiles_folder)
            if fn.startswith("tile_")
            and fn.endswith(".tif")
            and os.path.getsize(os.path.join(tiles_folder, fn)) > 0
        ]

        missing_ids = [
            i for i in range(num_tiles)
            if not tile_is_valid(tile_path(tiles_folder, i))
        ]

        print(f"[{calendar_date}] download phase: {dl_secs:.1f}s ({dl_secs/60:,.2f} min)")
        print(f"[{calendar_date}] valid tiles: {len(tile_files)} / {num_tiles}")
        print(f"[{calendar_date}] missing tiles: {len(missing_ids)}")

        if failed_records_all or missing_ids:
            failed_log = os.path.join(tiles_folder, f"FAILED_TILES_{calendar_date}.json")
            save_json(failed_log, {
                "calendar_date": calendar_date,
                "modis_date": modis_date,
                "num_tiles": num_tiles,
                "valid_tiles": len(tile_files),
                "missing_tiles": missing_ids,
                "failed_records": failed_records_all,
                "created_at": now_str()
            })
            print(f"[{calendar_date}] failed/missing tile log: {failed_log}")

        if missing_ids and STOP_ON_FAILED_TILES:
            raise RuntimeError(
                f"{calendar_date}: {len(missing_ids)} of {num_tiles} tiles are missing. "
                f"Stopping before mosaic to avoid incomplete output. "
                f"See failed log in: {tiles_folder}"
            )

        if not tile_files:
            print(f"[{calendar_date}] WARN | no tiles; skipping date")
            continue

        # ----------------------------------------------------
        # build VRT and final mosaic
        # ----------------------------------------------------
        vrt_file = os.path.join(tiles_folder, "tiles.vrt")
        if os.path.exists(vrt_file):
            os.remove(vrt_file)

        print(f"[{calendar_date}] building VRT from {len(tile_files)} tiles")

        vrt_opts = gdal.BuildVRTOptions(
            srcNodata=OUT_NODATA,
            VRTNodata=OUT_NODATA,
            hideNodata=False
        )

        vrt_ds = gdal.BuildVRT(vrt_file, tile_files, options=vrt_opts)

        if vrt_ds is None:
            raise RuntimeError(f"{calendar_date}: GDAL BuildVRT failed.")

        vrt_ds = None

        print(f"[{calendar_date}] writing final mosaic")

        if crop and aoi_vector:
            warp_kwargs = dict(
                format="GTiff",
                multithread=True,
                creationOptions=FINAL_CO,
                warpOptions=["NUM_THREADS=ALL_CPUS", "CUTLINE_ALL_TOUCHED=TRUE"],
                cutlineDSName=aoi_vector,
                cropToCutline=True,
                dstSRS=target_crs,
                outputType=gdal.GDT_Int16,
                dstNodata=OUT_NODATA
            )

            if AOI_LAYER:
                warp_kwargs["cutlineLayer"] = AOI_LAYER

            warp_opts = gdal.WarpOptions(**warp_kwargs)

            result = gdal.Warp(
                destNameOrDestDS=tmp_out_tif,
                srcDSOrSrcDSTab=vrt_file,
                options=warp_opts
            )
        else:
            result = gdal.Warp(
                destNameOrDestDS=tmp_out_tif,
                srcDSOrSrcDSTab=vrt_file,
                format="GTiff",
                options=FINAL_CO,
                outputType=gdal.GDT_Int16,
                dstNodata=OUT_NODATA
            )

        if result is None:
            raise RuntimeError(f"{calendar_date}: GDAL Warp failed.")

        result = None

        if not os.path.exists(tmp_out_tif) or os.path.getsize(tmp_out_tif) == 0:
            raise RuntimeError(
                f"{calendar_date}: temporary output was not created correctly: {tmp_out_tif}"
            )
        if not ndvi_raster_is_valid(tmp_out_tif):
            raise RuntimeError(
                f"{calendar_date}: temporary output failed raster validation: "
                f"{tmp_out_tif}"
            )

        os.replace(tmp_out_tif, out_tif)

        round_secs = time.perf_counter() - round_start
        mark_completed_output(out_tif, calendar_date, num_tiles, round_secs)
        completed_outputs.append(out_tif)

        out_gb = bytes_to_gb(os.path.getsize(out_tif))
        print(f"[{calendar_date}] output size: {out_gb:,.3f} GB")

        if not keep_tiles:
            print(f"[{calendar_date}] deleting temporary tile folder")
            shutil.rmtree(tiles_folder, ignore_errors=True)

        round_end_dt = datetime.now()

        print(f"[{calendar_date}] END TIME: {round_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"[{calendar_date}] DONE | elapsed: {round_secs/60:,.2f} min "
            f"(~{round_secs/max(1, num_tiles):.2f} s/tile @ {MAX_WORKERS} workers)"
        )

    batch_secs = time.perf_counter() - batch_start
    batch_end_dt = datetime.now()

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    print(f"Batch start: {batch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Batch end:   {batch_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dates processed/requested: {len(completed_outputs)} / {len(dates)}")
    print(f"Total elapsed: {timedelta(seconds=int(batch_secs))}")

    if completed_outputs:
        total_bytes = sum(os.path.getsize(fp) for fp in completed_outputs if os.path.exists(fp))
        total_gb = bytes_to_gb(total_bytes)
        avg_gb = total_gb / len(completed_outputs)
        avg_secs = batch_secs / len(completed_outputs)

        print(f"Total output size: {total_gb:,.3f} GB")
        print(f"Average size per MODIS date: {avg_gb:,.3f} GB")
        print(f"Average time per MODIS date: {timedelta(seconds=int(avg_secs))}")
        print(f"Estimated 2000–2025  size: {avg_gb * APPROX_MODIS_DATES_2000_2025:,.1f} GB")
        print(f"Estimated 2000–2025 time: {format_eta(avg_secs * APPROX_MODIS_DATES_2000_2025)}")

    print("=" * 78)

# ============================================================
# 6. Tile download
# ============================================================

def download_tile_batch(
    grid_points: gpd.GeoDataFrame,
    tile_ids: List[int],
    tiles_folder: str,
    product: str,
    band: str,
    modis_date: str,
    kmAboveBelow: int,
    kmLeftRight: int,
    apply_scale: bool,
    calendar_date: str,
    pass_id: int
) -> List[Dict[str, Any]]:

    failed_records = []
    empty_subset_count = 0
    pass_start = time.perf_counter()
    completed = 0
    total = len(tile_ids)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}

        for tile_id in tile_ids:
            row = grid_points.loc[tile_id]
            future = ex.submit(
                download_tile,
                row=row,
                tile_id=tile_id,
                tiles_folder=tiles_folder,
                product=product,
                band=band,
                modis_date=modis_date,
                kmAboveBelow=kmAboveBelow,
                kmLeftRight=kmLeftRight,
                apply_scale=apply_scale
            )
            futures[future] = tile_id

        for f in as_completed(futures):
            result = f.result()
            completed += 1

            if not result.get("ok", False):
                failed_records.append(result)
            elif result.get("status") == "empty_subset":
                empty_subset_count += 1

            if completed % PROGRESS_EVERY == 0 or completed == total:
                elapsed = time.perf_counter() - pass_start
                rate = completed / max(elapsed, 1)
                remaining = total - completed
                eta = remaining / rate if rate > 0 else math.nan

                print(
                    f"[{calendar_date}] pass {pass_id} progress: "
                    f"{completed}/{total} ({completed/total*100:5.1f}%) | "
                    f"elapsed {format_eta(elapsed)} | ETA {format_eta(eta)} | "
                    f"failed so far {len(failed_records)} | "
                    f"empty subsets {empty_subset_count}"
                )

    if empty_subset_count:
        print(
            f"[{calendar_date}] pass {pass_id}: "
            f"legitimate empty-subset NoData tiles = {empty_subset_count}"
        )

    return failed_records


def download_tile(
    row,
    tile_id: int,
    tiles_folder: str,
    product: str,
    band: str,
    modis_date: str,
    kmAboveBelow: int,
    kmLeftRight: int,
    apply_scale: bool
) -> Dict[str, Any]:

    target_file = tile_path(tiles_folder, tile_id)

    if tile_is_valid(target_file):
        return {
            "ok": True,
            "tile": tile_id,
            "file": target_file,
            "status": "exists"
        }

    if os.path.exists(target_file) and os.path.getsize(target_file) == 0:
        os.remove(target_file)

    tmp_file = target_file + ".tmp.tif"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    lat, lon = row.geometry.y, row.geometry.x
    url = build_request_url(
        product, lat, lon, band,
        modis_date, modis_date,
        kmAboveBelow, kmLeftRight
    )

    last_error = None
    status_code = None

    for attempt in range(1, TILE_MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADER, timeout=180)
            status_code = r.status_code

            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}: {r.text[:500]}"
                sleep(1.5 * attempt)
                continue

            try:
                data = r.json()
            except Exception as e:
                last_error = f"JSON parse failed: {e}"
                sleep(1.5 * attempt)
                continue

            empty_subset = modis_data_to_geotiff(
                data, tmp_file, apply_scale=apply_scale
            )

            if not tile_is_valid(tmp_file):
                last_error = "temporary GeoTIFF was not created or is empty"
                sleep(1.5 * attempt)
                continue

            os.replace(tmp_file, target_file)

            if tile_is_valid(target_file):
                sleep(0.03)
                return {
                    "ok": True,
                    "tile": tile_id,
                    "file": target_file,
                    "lat": float(lat),
                    "lon": float(lon),
                    "attempts": attempt,
                    "status": "empty_subset" if empty_subset else "downloaded"
                }

            last_error = "final GeoTIFF was not created or is empty"

        except Exception as e:
            last_error = str(e)

        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass

        sleep(1.5 * attempt)

    return {
        "ok": False,
        "tile": tile_id,
        "lat": float(lat),
        "lon": float(lon),
        "url": url,
        "status_code": status_code,
        "error": last_error,
        "attempts": TILE_MAX_RETRIES
    }
def build_request_url(
    product,
    lat,
    lon,
    band,
    start_date,
    end_date,
    km_vert,
    km_horiz
):
    return (
        f"https://modis.ornl.gov/rst/api/v1/{product}/subset"
        f"?latitude={lat}&longitude={lon}&band={band}"
        f"&startDate={start_date}&endDate={end_date}"
        f"&kmAboveBelow={km_vert}&kmLeftRight={km_horiz}"
    )


# ============================================================
# 7. ORNL JSON to Int16 GeoTIFF
# ============================================================

def modis_data_to_geotiff(data, target_file, apply_scale=True):
    """
    Convert ORNL MODIS JSON subset to Int16 GeoTIFF.
    """

    def is_numeric(x):
        try:
            float(x)
            return True
        except Exception:
            return False

    # Required metadata
    nrows = int(data["nrows"])
    ncols = int(data["ncols"])

    subset = data.get("subset", [])
    empty_subset = not bool(subset)

    if empty_subset:
        if CREATE_NODATA_TILE_FOR_EMPTY_SUBSET:
            stored = np.full((nrows, ncols), OUT_NODATA, dtype=np.int16)
        else:
            preview = json.dumps(data)[:500] if isinstance(data, dict) else str(data)[:500]
            raise ValueError(f"No 'subset' in API response. Response preview: {preview}")
    else:
        raw = np.array(subset[0]["data"], dtype=float).reshape(nrows, ncols)

        fill_mask = raw <= -3000
            
        if apply_scale:
            scale = data.get("scale", "")
            if not is_numeric(scale):
                raise ValueError("MODIS response does not contain a valid scale factor.")
            scale = float(scale)
            if not np.isclose(scale, EXPECTED_MODIS_SCALE):
                raise ValueError(
                    f"Unexpected MOD13Q1 NDVI scale factor: {scale}; "
                    f"expected {EXPECTED_MODIS_SCALE}."
                )
            physical = raw * scale
            stored = np.rint(physical * OUT_MULT).astype(np.int32)
        else:
            stored = np.rint(raw).astype(np.int32)

        valid_mask = ~fill_mask
        if np.any(
            (stored[valid_mask] < VALID_STORED_MIN)
            | (stored[valid_mask] > VALID_STORED_MAX)
        ):
            bad_min = int(stored[valid_mask].min())
            bad_max = int(stored[valid_mask].max())
            raise ValueError(
                "Stored MOD13Q1 NDVI values fall outside the expected valid "
                f"range [{VALID_STORED_MIN}, {VALID_STORED_MAX}]: "
                f"observed {bad_min} to {bad_max}."
            )

        stored[fill_mask] = OUT_NODATA
        stored = stored.astype(np.int16)

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(target_file, ncols, nrows, 1, gdal.GDT_Int16, options=TEMP_TILE_CO)

    if ds is None:
        raise RuntimeError(
            f"GDAL failed to create tile GeoTIFF: {target_file} | "
            f"ncols={ncols}, nrows={nrows}"
        )

    xll = float(data["xllcorner"])
    yll = float(data["yllcorner"]) + nrows * float(data["cellsize"])
    cell = float(data["cellsize"])

    ds.SetGeoTransform([xll, cell, 0, yll, 0, -cell])

    srs = osr.SpatialReference()
    srs.ImportFromProj4(PROJ4_SINUSOIDAL)
    ds.SetProjection(srs.ExportToWkt())

    b = ds.GetRasterBand(1)
    b.WriteArray(stored)
    b.SetNoDataValue(OUT_NODATA)
    b.FlushCache()

    ds.FlushCache()
    ds = None

    return empty_subset


# ============================================================
# 8. Robust grid builder with coverage diagnostics
# ============================================================

def make_data_request_grid(
    aoi_shapefile: str,
    step_size: int,
    target_file: Optional[str] = None,
    exclude_nonoverlapping: bool = True,
    move_edge_points_inside_aoi: bool = False,
    export_footprints: bool = True,
) -> gpd.GeoDataFrame:
    """
    Build a robust regular request grid over the AOI in MODIS sinusoidal space.
    """

    aoi = read_vector(aoi_shapefile)
    if aoi.empty:
        raise ValueError("AOI is empty.")
    if aoi.crs is None:
        raise ValueError("AOI has no CRS. Please define CRS before running.")
    aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notnull()].copy()
    aoi["geometry"] = aoi.geometry.buffer(0)

    proj_aoi = aoi.to_crs(PROJ4_SINUSOIDAL)
    aoi_union = geometry_union(proj_aoi)

    minx, miny, maxx, maxy = proj_aoi.total_bounds

    half_x = KM_LEFT_RIGHT * 1000.0
    half_y = KM_ABOVE_BELOW * 1000.0

    # Expanded bounds ensure edges are not missed.
    start_x = minx - half_x
    end_x   = maxx + half_x
    start_y = miny - half_y
    end_y   = maxy + half_y

    xs = np.arange(start_x, end_x + step_size, step_size)
    ys = np.arange(start_y, end_y + step_size, step_size)

    print("[grid] projected AOI bounds used directly in MODIS sinusoidal CRS")
    print(f"[grid] candidate columns={len(xs)}, rows={len(ys)}, cells={len(xs) * len(ys)}")

    points = []
    footprints = []
    moved_points = 0

    for x in xs:
        for y in ys:
            center_pt = Point(x, y)

            request_footprint = Polygon([
                (x - half_x, y - half_y),
                (x - half_x, y + half_y),
                (x + half_x, y + half_y),
                (x + half_x, y - half_y),
            ])

            if exclude_nonoverlapping and not request_footprint.intersects(aoi_union):
                continue

            request_pt = center_pt

            if move_edge_points_inside_aoi and not aoi_union.contains(center_pt):
                inter = request_footprint.intersection(aoi_union)
                if not inter.is_empty:
                    request_pt = inter.representative_point()
                    moved_points += 1

                    x2, y2 = request_pt.x, request_pt.y
                    request_footprint = Polygon([
                        (x2 - half_x, y2 - half_y),
                        (x2 - half_x, y2 + half_y),
                        (x2 + half_x, y2 + half_y),
                        (x2 + half_x, y2 - half_y),
                    ])

            points.append(request_pt)
            footprints.append(request_footprint)

    grid_points = gpd.GeoDataFrame(geometry=points, crs=PROJ4_SINUSOIDAL)
    footprint_gdf = gpd.GeoDataFrame(geometry=footprints, crs=PROJ4_SINUSOIDAL)

    print(f"[grid] retained request points: {len(grid_points)}")
    print(f"[grid] edge/ocean centres moved inside AOI intersection: {moved_points}")

    if grid_points.empty:
        raise RuntimeError("No request points retained. Check AOI and grid settings.")

    # Coverage diagnostic
    coverage_union = geometry_union(footprint_gdf)
    uncovered = aoi_union.difference(coverage_union)

    if uncovered.is_empty:
        uncovered_area_km2 = 0.0
    else:
        uncovered_area_km2 = uncovered.area / 1e6

    if uncovered_area_km2 <= COVERAGE_TOLERANCE_KM2:
        print(
            f"[grid-check] PASS: request footprints cover the AOI "
            f"(uncovered area {uncovered_area_km2:,.4f} km² <= tolerance)."
        )
    else:
        print(f"[grid-check] WARNING: uncovered AOI area = {uncovered_area_km2:,.2f} km²")
        print("[grid-check] Inspect exported uncovered AOI file before downloading.")

    grid_points_out = grid_points.to_crs(aoi.crs)

    if target_file:
        target_dir = os.path.dirname(target_file)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        delete_shapefile_sidecars(target_file)
        grid_points_out.to_file(target_file)

        if export_footprints:
            base = os.path.splitext(target_file)[0]

            footprint_file = base + "_request_footprints.gpkg"
            uncovered_file = base + "_uncovered_aoi.gpkg"

            delete_if_exists(footprint_file)
            footprint_gdf.to_crs(aoi.crs).to_file(
                footprint_file,
                layer="request_footprints",
                driver="GPKG"
            )
            print(f"[grid-check] request footprints written: {footprint_file}")

            if uncovered_area_km2 > COVERAGE_TOLERANCE_KM2:
                delete_if_exists(uncovered_file)
                uncovered_gdf = gpd.GeoDataFrame(
                    {"area_km2": [uncovered_area_km2]},
                    geometry=[uncovered],
                    crs=PROJ4_SINUSOIDAL
                ).to_crs(aoi.crs)

                uncovered_gdf.to_file(
                    uncovered_file,
                    layer="uncovered_aoi",
                    driver="GPKG"
                )
                print(f"[grid-check] uncovered AOI written: {uncovered_file}")

                if STOP_IF_GRID_UNCOVERED:
                    raise RuntimeError(
                        f"Grid footprints do not fully cover AOI. "
                        f"Uncovered area = {uncovered_area_km2:,.2f} km². "
                        f"Inspect: {uncovered_file}"
                    )
    return grid_points_out

# ============================================================
# 9. Optional full-coverage date checker
# ============================================================

def get_full_coverage_dates(grid: gpd.GeoDataFrame) -> List[Dict[str, str]]:
    """
    Find dates where every grid point reports data.
    WARNING: slow — it queries the dates endpoint for every point.
    """
    available = {}

    for i, pt in grid.iterrows():
        lat, lon = pt.geometry.y, pt.geometry.x

        r = requests.get(
            f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates"
            f"?longitude={lon}&latitude={lat}",
            headers=HEADER,
            timeout=60
        )
        r.raise_for_status()
        available[i] = r.json().get("dates", [])
        sleep(0.1)

    counts = {}

    for dates in available.values():
        for d in dates:
            cal = d["calendar_date"]
            counts[cal] = counts.get(cal, 0) + 1

    return [d for d, c in counts.items() if c == len(grid)]


# ============================================================
# 10. run
# ============================================================

if __name__ == "__main__":
    main()

# %%