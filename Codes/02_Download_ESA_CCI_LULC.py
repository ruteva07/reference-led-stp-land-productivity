# %%
"""
Author: Evariste Rutebuka
Date: 2025-2026
Download ESA CCI / C3S 300 m annual land-cover maps for Africa.

Background
----------
This script acquires the annual land-cover record used to provide land-cover
context for the productivity-based State, Trend and Performance analyses.
Annual land-cover classifications are obtained from the Copernicus Climate
Data Store (CDS) satellite land-cover product and the `lccs_class` variable
is extracted for subsequent analysis.

Purpose
-------
- Download annual ESA CCI / C3S land-cover maps from CDS.
- Study period: 2001-2022.
- Spatial subset: Africa bounding box.
- Extract the annual `lccs_class` layer.
- Export one GeoTIFF per year.
- Optionally crop/mask each GeoTIFF to the Africa study-area polygon.
- Verify that all requested years were completed successfully.

Main variable of interest
-------------------------
lccs_class:
    Annual land-cover class per pixel using the FAO LCCS legend.

Role in the workflow
--------------------
This script performs preliminary data acquisition, extraction and spatial
subsetting only. Harmonisation to broader land-cover classes, temporal
persistence assessment, alignment to the common analytical grid, and use in
State, Trend and Performance analyses are performed in downstream scripts.

Source-version handling
-----------------------
- Years up to 2015 use ESA CCI / C3S version v2_0_7cds.
- Years from 2016 onward use C3S version v2_1_1.
- This source-version split is required only to request the correct annual
  file. It is not treated as an analytical breakpoint.

Key steps
---------
1. Read and validate the study-area polygon.
2. Determine the correct CDS source version for each year.
3. Download the annual land-cover archive for the Africa bounding box.
4. Extract the annual NetCDF file.
5. Locate and extract the `lccs_class` variable.
6. Confirm that the annual layer is two-dimensional and geographic.
7. Export the categorical layer as an unsigned-byte GeoTIFF.
8. Optionally crop/mask the GeoTIFF to the Africa AOI.
9. Check at the end of the run that no requested year is missing.

"""

from pathlib import Path
import os
import zipfile
import shutil
import cdsapi
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd
import rasterio
from rasterio.mask import mask


# ============================================================
# 1. user settings
# ============================================================

# Edit only the project root for another machine/environment.
MAIN_FOLDER = Path(r"C:\PATH\TO\Paper\Codes\Africa")
COMMON_AOI_INPUTS = MAIN_FOLDER / "Polygons"  # AOI folder containing Africa.gpkg
AOI_PATH = COMMON_AOI_INPUTS / "Africa.gpkg"
        
AOI_LAYER = None  # set if your GPKG has a specific layer

OUT_DIR = MAIN_FOLDER / "cci_c3s_lulc_300m_africa"
ZIP_DIR = OUT_DIR / "zips"
NC_DIR = OUT_DIR / "netcdf"
TIF_DIR = OUT_DIR / "geotiff_lccs_class"
MASKED_TIF_DIR = OUT_DIR / "geotiff_lccs_class_aoi_masked"

for d in [ZIP_DIR, NC_DIR, TIF_DIR, MASKED_TIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Full study period used by the analytical workflow.
START_YEAR = 2001
END_YEAR = 2022
AFRICA_BBOX = [38, -26, -35, 60]
CROP_TO_AOI = True
KEEP_ZIPS = False
LULC_DTYPE = "uint8"
LULC_NODATA = 0

# ============================================================
# 2. helper functions
# ============================================================
def version_for_year(year: int) -> str:
    """
    Return the CDS land-cover product version for a given year.

    Important:
    ----------
    This year split is NOT an analytical breakpoint.
    It does not define the degradation-analysis baseline, moving-window logic,
    or LULC stability assessment.

    The split only reflects how the ESA CCI / C3S land-cover archive is
    organised in the CDS download system:

        - v2_0_7cds : historical ESA CCI land-cover years up to 2015
        - v2_1_1    : C3S continuation from 2016 onward
    """
    if year <= 2015:
        return "v2_0_7cds"

    return "v2_1_1"


def open_aoi(path: Path, layer=None) -> gpd.GeoDataFrame:
    """
    Read, validate and dissolve the study-area geometry.
    """
    if layer:
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError("AOI is empty.")

    if gdf.crs is None:
        raise ValueError("AOI CRS is missing.")

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

    if gdf.empty:
        raise ValueError("AOI contains no valid non-empty geometries.")

    # Retain the original light geometry-repair approach.
    gdf["geometry"] = gdf.geometry.buffer(0)

    return gdf.dissolve().reset_index(drop=True)


def validate_lulc_tif(path: Path) -> None:
    """
    Perform lightweight structural validation of an annual LULC GeoTIFF.

    This does not change raster values. It verifies that the expected
    categorical output exists and has the basic structure required downstream.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty GeoTIFF: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError(
                f"Expected one raster band in {path.name}; found {src.count}."
            )

        if src.width <= 0 or src.height <= 0:
            raise RuntimeError(f"Invalid raster dimensions in {path.name}.")

        if src.crs is None:
            raise RuntimeError(f"Missing CRS in {path.name}.")

        if src.dtypes[0] != LULC_DTYPE:
            raise RuntimeError(
                f"Unexpected data type in {path.name}: {src.dtypes[0]}; "
                f"expected {LULC_DTYPE}."
            )

        if src.nodata != LULC_NODATA:
            raise RuntimeError(
                f"Unexpected NoData in {path.name}: {src.nodata}; "
                f"expected {LULC_NODATA}."
            )


def extract_first_nc(zip_path: Path, out_nc: Path) -> Path:
    """
    Extract the annual NetCDF file from the CDS zip archive.
    """
    if out_nc.exists() and out_nc.stat().st_size > 0:
        return out_nc

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Not a valid zip archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        nc_members = [
            m for m in z.namelist()
            if m.lower().endswith((".nc", ".nc4", ".netcdf"))
        ]

        if not nc_members:
            raise RuntimeError(f"No NetCDF found in zip: {zip_path}")

        # Do not silently choose one file if the archive structure changes.
        if len(nc_members) != 1:
            raise RuntimeError(
                f"Expected exactly one NetCDF in {zip_path.name}; "
                f"found {len(nc_members)}: {nc_members}"
            )

        member = nc_members[0]

        tmp_dir = out_nc.parent / f"_tmp_extract_{out_nc.stem}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            z.extract(member, tmp_dir)
            extracted = tmp_dir / member

            if not extracted.exists() or extracted.stat().st_size == 0:
                raise RuntimeError(f"Extracted NetCDF is missing or empty: {extracted}")

            out_nc.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(out_nc))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_nc


def find_lccs_variable(ds: xr.Dataset) -> str:
    """
    Find the annual land-cover classification variable.

    Known names are supported explicitly. If none are present, fail rather than
    selecting an unrelated variable simply because its name contains 'class'.
    """
    preferred = ["lccs_class", "land_cover_lccs", "lc_class"]

    for v in preferred:
        if v in ds.data_vars:
            return v

    raise RuntimeError(
        "Could not find the expected LCCS land-cover variable. "
        f"Available variables: {list(ds.data_vars)}"
    )


def nc_lccs_to_geotiff(nc_path: Path, out_tif: Path) -> Path:
    """
    Convert lccs_class from NetCDF to GeoTIFF.
    """
    if out_tif.exists() and out_tif.stat().st_size > 0:
        validate_lulc_tif(out_tif)
        return out_tif

    ds = xr.open_dataset(nc_path, decode_times=False)

    try:
        var_name = find_lccs_variable(ds)
        da = ds[var_name]

        # Remove singleton dimensions such as annual time if present.
        da = da.squeeze(drop=True)

        # Annual land-cover output should contain only the two spatial dimensions.
        if da.ndim != 2:
            raise RuntimeError(
                f"Expected a two-dimensional annual land-cover layer after "
                f"squeezing singleton dimensions; got dims={da.dims}, "
                f"shape={da.shape}."
            )

        # Try to identify spatial dimension names.
        # CCI/C3S files are usually lat/lon in Plate Carrée.
        if "lon" in da.dims and "lat" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        elif "longitude" in da.dims and "latitude" in da.dims:
            da = da.rio.set_spatial_dims(
                x_dim="longitude", y_dim="latitude", inplace=False
            )
        else:
            raise RuntimeError(f"Unexpected dimensions for {var_name}: {da.dims}")

        # Assign CRS if missing. The source grid is geographic / Plate Carrée.
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326", inplace=False)

        # Ensure north-up orientation.
        y_dim = da.rio.y_dim
        if da[y_dim][0] < da[y_dim][-1]:
            da = da.sortby(y_dim, ascending=False)

        # Confirm that any explicit source fill value is compatible with the
        # class-0 NoData convention used by this workflow.
        source_nodata = da.rio.nodata
        if source_nodata is None:
            source_nodata = da.encoding.get("_FillValue", None)
        if source_nodata is None:
            source_nodata = da.attrs.get("_FillValue", None)

        if source_nodata is not None and int(source_nodata) != LULC_NODATA:
            raise RuntimeError(
                f"Unexpected source NoData/_FillValue for {var_name}: "
                f"{source_nodata}; expected {LULC_NODATA}. "
                "Inspect the source product before continuing."
            )

        da = da.rio.write_nodata(LULC_NODATA, inplace=False)

        # Export as compressed categorical GeoTIFF.
        da.rio.to_raster(
            out_tif,
            dtype=LULC_DTYPE,
            nodata=LULC_NODATA,
            compress="ZSTD",
            tiled=True,
            predictor=2,
            BIGTIFF="IF_SAFER"
        )

    finally:
        ds.close()

    validate_lulc_tif(out_tif)
    return out_tif


def crop_tif_to_aoi(in_tif: Path, out_tif: Path, aoi: gpd.GeoDataFrame) -> Path:
    """
    Crop/mask a GeoTIFF to the AOI polygon.
    """
    if out_tif.exists() and out_tif.stat().st_size > 0:
        validate_lulc_tif(out_tif)
        return out_tif

    with rasterio.open(in_tif) as src:
        aoi_src = aoi.to_crs(src.crs)
        geom = [aoi_src.geometry.iloc[0].__geo_interface__]

        arr, transform = mask(
            src,
            geom,
            crop=True,
            filled=True,
            nodata=LULC_NODATA
        )

        profile = src.profile.copy()
        profile.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=transform,
            dtype=LULC_DTYPE,
            nodata=LULC_NODATA,
            compress="ZSTD",
            tiled=True,
            predictor=2,
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr.astype(LULC_DTYPE, copy=False))

    validate_lulc_tif(out_tif)
    return out_tif


def download_one_year(client: cdsapi.Client, year: int) -> Path:
    """
    Download one annual land-cover zip from CDS.
    """
    version = version_for_year(year)

    zip_path = ZIP_DIR / f"CCI_C3S_LC_{year}_{version}_Africa.zip"

    if zip_path.exists() and zip_path.stat().st_size > 0:
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                f"[{year}] existing file is not a valid zip archive: {zip_path}"
            )

        print(f"[{year}] zip exists: {zip_path.name}")
        return zip_path

    request = {
        "variable": "all",
        "year": [str(year)],
        "version": [version],
        "area": AFRICA_BBOX,
    }

    print(f"[{year}] downloading | version={version}")
    print(f"       request={request}")

    client.retrieve(
        "satellite-land-cover",
        request
    ).download(str(zip_path))

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise RuntimeError(f"[{year}] download failed or empty zip: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"[{year}] downloaded file is not a valid zip archive: {zip_path}"
        )

    return zip_path


# ============================================================
# 3. main
# ============================================================

def main():
    
   # Initialise the Copernicus Climate Data Store (CDS) API client.
    # Requires a valid personal CDS API token configured locally in ~/.cdsapirc.
    # Credentials are intentionally kept outside this script and should not be
    # included in the public code archive. See the CDS API setup instructions
    # if authentication fails with a "Missing/incomplete configuration file" error.
    client = cdsapi.Client()

    aoi = open_aoi(AOI_PATH, AOI_LAYER) if CROP_TO_AOI else None

    completed = []
    completed_years = []
    failed_years = {}

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            version = version_for_year(year)

            zip_path = download_one_year(client, year)

            nc_path = NC_DIR / f"CCI_C3S_LC_{year}_{version}_Africa.nc"
            extract_first_nc(zip_path, nc_path)

            tif_path = TIF_DIR / f"CCI_C3S_LCCS_Class_{year}_Africa_bbox.tif"
            nc_lccs_to_geotiff(nc_path, tif_path)

            if CROP_TO_AOI:
                masked_path = MASKED_TIF_DIR / f"CCI_C3S_LCCS_Class_{year}_Africa_AOI.tif"
                crop_tif_to_aoi(tif_path, masked_path, aoi)
                completed.append(masked_path)
                completed_years.append(year)
                print(f"[{year}] done → {masked_path}")
            else:
                completed.append(tif_path)
                completed_years.append(year)
                print(f"[{year}] done → {tif_path}")

            if not KEEP_ZIPS and zip_path.exists():
                zip_path.unlink()

        except Exception as e:
            failed_years[year] = str(e)
            print(f"\n[{year}] FAILED: {e}\n")
            print("Continuing to next year...\n")

    print("\nCompleted years:")
    for p in completed:
        print(" -", p)

    # Allow every year to be attempted, but do not silently accept an incomplete
    # annual record at the end of the production run.
    expected_years = set(range(START_YEAR, END_YEAR + 1))
    missing_years = sorted(expected_years - set(completed_years))

    if missing_years:
        print("\nFAILED / MISSING YEARS:")
        for year in missing_years:
            reason = failed_years.get(year, "output not completed")
            print(f" - {year}: {reason}")

        raise RuntimeError(
            "Land-cover acquisition is incomplete. "
            f"Missing years: {missing_years}"
        )

    print(
        f"\nAll requested land-cover years completed successfully: "
        f"{START_YEAR}-{END_YEAR}"
    )


if __name__ == "__main__":
    main()

# %%
