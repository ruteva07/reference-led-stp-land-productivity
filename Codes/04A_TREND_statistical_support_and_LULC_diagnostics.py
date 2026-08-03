# -*- coding: utf-8 -*-
"""
TREND statistical-support and LULC-transition diagnostics
=========================================================

Author
------
Evariste Rutebuka,

Role in the archive
-------------------
This is a post-processing / supplementary-analysis script. It does not estimate
NDVI Trend and it does not modify the canonical Trend classification generated
by the core Trend workflow.

It consumes the full-period 2001–2022 Trend masks and reusable reporting-zone
rasters to quantify:

1. the area satisfying progressively stronger negative-Trend support rules;
2. the proportion of the common included 2022 land-cover domain represented
   by those negative masks;
3. retention of each statistically supported negative mask relative to the
   potential negative-slope population;
4. country, grouped-land-cover and country × land-cover summaries; and
5. the broad land-cover-transition context of declining pixels.

Trend inputs
------------
The five full-period masks are:

    potential
        Physical negative-slope criterion only.

    ols
        Negative slope with finite-sample OLS Student-t support.

    mann_kendall
        Mann–Kendall-supported negative direction.

    newey_west
        Negative slope with Newey–West support.

    strict_joint
        Joint OLS + Mann–Kendall + Newey–West supported decline.

There is no standalone OLS-associated Z-score criterion in this diagnostic
workflow. Mann–Kendall statistics remain part of the upstream MK test.

Reporting denominator
---------------------
The primary denominator is the total area of included 2022 country × grouped
land-cover zones. Sparse/bare and water/snow/ice are excluded from both the
numerator and denominator.

This denominator is intentionally a stable reporting-domain denominator. It is
not additionally restricted to pixels with sufficient Trend observations.
Accordingly, the primary percentage is:

    negative_share_of_valid_area_pct =
        negative-mask pixels / included 2022 reporting-domain pixels × 100

Here, "valid area" refers to the valid included reporting-zone domain, not to a
separate Trend-validity mask. This behaviour is retained because it matches the
production summaries reported in the analysis.

A secondary diagnostic is:

    retention_from_potential_pct =
        negative pixels under a support rule /
        potential-negative pixels × 100

LULC-transition context
-----------------------
The supplementary figure reports the broad transition class already generated
by the core Trend workflow. Because the core workflow scales transition
thresholds to the number of valid consecutive LULC year-pairs, the class labels
should be interpreted as transition-frequency categories rather than as exact
transition counts for every pixel. For a complete 2001–2022 LULC history they
approximately correspond to:

    class 1 : no detected broad transition
    class 2 : low transition frequency (about 1–2 transitions)
    class 3 : moderate transition frequency (about 3–5 transitions)
    class 4 : high transition frequency (more than about 5 transitions)

Panel (b) of the supplementary figure is normalised within declining pixels
having a valid transition class.

Upstream requirements
---------------------
Run after:
    - the core Trend workflow has produced the full-period Trend masks and
      LULC-transition class raster; and
    - the STP reusable-zone preparation has produced country and country × LULC
      zone rasters and lookup tables.


"""

from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
# =============================================================================
# 1. ARCHIVE PATH SETTINGS
# =============================================================================

# Edit this one path when reproducing the analysis on another machine.

MAIN_FOLDER= Path(r"C:\PATH\TO\Paper\Codes\Africa")
STP_DATABASE_ROOT = MAIN_FOLDER / "STP_Supplementary_Database"

TREND_ROOT = (
    MAIN_FOLDER
    / "TREND_NDVI_Africa_From_StateCache"
    / "Annual"
)
TREND_FULL_RASTER_DIR = TREND_ROOT / "Trend_2001_2022" / "rasters"

REUSABLE_ZONE_DIR = (
    STP_DATABASE_ROOT
    / "outputs"
    / "reusable_zones"
)

TREND_RASTERS = {
    "potential": (
        TREND_FULL_RASTER_DIR
        / "TrendNDVI_Trend_2001_2022_DegrPotential_Annual_EA250m.tif"
    ),
    "ols": (
        TREND_FULL_RASTER_DIR
        / "TrendNDVI_Trend_2001_2022_DegrP_neg_Annual_EA250m.tif"
    ),
    "newey_west": (
        TREND_FULL_RASTER_DIR
        / "TrendNDVI_Trend_2001_2022_DegrNW_neg_Annual_EA250m.tif"
    ),
    "mann_kendall": (
        TREND_FULL_RASTER_DIR
        / "TrendNDVI_Trend_2001_2022_DegrMK_neg_Annual_EA250m.tif"
    ),
    "strict_joint": (
        TREND_FULL_RASTER_DIR
        / "TrendNDVI_Trend_2001_2022_DegrPMKNW_neg_Annual_EA250m.tif"
    ),
}

COUNTRY_RASTER = REUSABLE_ZONE_DIR / "country_id_EA250m.tif"
COUNTRY_LULC_RASTER = REUSABLE_ZONE_DIR / "country_lulc_2022_zone_id.tif"
COUNTRY_LOOKUP = REUSABLE_ZONE_DIR / "country_lookup.csv"
COUNTRY_LULC_LOOKUP = REUSABLE_ZONE_DIR / "country_lulc_lookup.csv"


# =============================================================================
# 2. OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = (
    STP_DATABASE_ROOT
    / "outputs"
    / "trend_diagnostics"
    / "statistical_support_and_LULC_2001_2022"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WRITE_EXCEL_SUMMARY = True

CREATE_TREND_SUPPORT_LULC_FIGURE = True
SHOW_TREND_SUPPORT_LULC_FIGURE = False
WRITE_TREND_SUPPORT_LULC_PNG = True
WRITE_TREND_SUPPORT_LULC_PDF = True
WRITE_TREND_SUPPORT_LULC_TABLE = True

FIGURE_DPI = 500
FIGURE_SIZE = (15.5, 8.5)
FONT_FAMILY = "DejaVu Sans"

FIGURE_OUTPUT_PNG = (
    OUTPUT_DIR / "Fig_S4_Trend_Statistical_Support_LULC_2001_2022.png"
)
FIGURE_OUTPUT_PDF = (
    OUTPUT_DIR / "Fig_S4_Trend_Statistical_Support_LULC_2001_2022.pdf"
)
FIGURE_OUTPUT_TABLE = (
    OUTPUT_DIR / "Fig_S4_Trend_Statistical_Support_LULC_2001_2022.csv"
)

# The transition-class raster is generated by the same full-period Trend
# workflow as the current statistical-support masks.
LULC_TRANSITION_CLASS_RASTER = (
    TREND_RASTERS["strict_joint"].parent
    / "TrendNDVI_Trend_2001_2022_LULC_stability_class_Code_Annual_EA250m.tif"
)
LULC_TRANSITION_CLASS_RASTER_FALLBACK = (
    TREND_RASTERS["strict_joint"].parent
    / "TrendNDVI_Trend_2001_2022_LULC_stability_class_Annual_EA250m.tif"
)


# =============================================================================
# 3. ANALYSIS SETTINGS
# =============================================================================

WINDOW_KEY = "full_2001_2022"
WINDOW = "2001-2022"

# Every supplied Trend-support raster uses value 1 for the negative class.
TARGET_VALUE = 1

# Excluded from both numerator and denominator.
EXCLUDED_LULC_NAMES = {
    "Sparse/bare",
    "Water/snow/ice",
}

STATISTIC_LABELS = {
    "potential": "Potential_negative",
    "ols": "OLS_P_negative",
    "mann_kendall": "MK_negative",
    "newey_west": "NW_negative",
    "strict_joint": "ALL_STAT_negative",
}

STATISTIC_ORDER = {
    "potential": 1,
    "ols": 2,
    "mann_kendall": 3,
    "newey_west": 4,
    "strict_joint": 5,
}

PLOT_STATISTIC_ORDER = [
    "potential",
    "ols",
    "mann_kendall",
    "newey_west",
    "strict_joint",
]

PLOT_STATISTIC_LABELS = {
    "potential": "Negative slope\nthreshold",
    "ols": "OLS-supported\ndecline",
    "mann_kendall": "Mann–Kendall-supported\ndecline",
    "newey_west": "Newey–West-supported\ndecline",
    "strict_joint": "Joint OLS + MK + NW\nsupported decline",
}

TRANSITION_ORDER = [1, 2, 3, 4]
TRANSITION_LABELS = {
    1: "No broad LULC transition",
    2: "Low transition frequency",
    3: "Moderate transition frequency",
    4: "High transition frequency",
}
TRANSITION_DETAILS = {
    1: "no detected transition",
    2: "low transition rate; ≈1–2 transitions for a complete history",
    3: "moderate transition rate; ≈3–5 transitions for a complete history",
    4: "high transition rate; >≈5 transitions for a complete history",
}

TRANSITION_COLORS = {
    1: "#666666",
    2: "#56B4E9",
    3: "#E69F00",
    4: "#D55E00",
}
FUNNEL_BAR_COLOR = "#0072B2"


# =============================================================================
# 4. VALIDATION HELPERS
# =============================================================================

def require_files(paths: list[Path]) -> None:
    """Fail early when a required input is missing or empty."""
    missing = [
        path
        for path in paths
        if (not path.exists()) or (not path.is_file()) or path.stat().st_size == 0
    ]

    if missing:
        raise FileNotFoundError(
            "The following required files were not found or were empty:\n"
            + "\n".join(str(path) for path in missing)
        )


def assert_same_grid(
    reference: rasterio.io.DatasetReader,
    candidate: rasterio.io.DatasetReader,
    candidate_name: str,
) -> None:
    problems = []

    if reference.width != candidate.width:
        problems.append(
            f"width {candidate.width} != {reference.width}"
        )

    if reference.height != candidate.height:
        problems.append(
            f"height {candidate.height} != {reference.height}"
        )

    if reference.transform != candidate.transform:
        problems.append("transform differs")

    if reference.crs != candidate.crs:
        problems.append(
            f"CRS {candidate.crs} != {reference.crs}"
        )

    if problems:
        raise ValueError(
            f"Raster alignment failed for {candidate_name}: "
            + "; ".join(problems)
        )


def infer_lookup_column(
    lookup: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in lookup.columns:
            return column

    raise KeyError(
        f"Could not identify {description}. "
        f"Tried {candidates}. "
        f"Available columns: {lookup.columns.tolist()}"
    )


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


# =============================================================================
# 5. LOOKUP PREPARATION
# =============================================================================

def prepare_country_lulc_lookup(
    country_lookup: pd.DataFrame,
    country_lulc_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare the country × LULC lookup and exclude non-vegetated classes.
    """

    zone_id_column = infer_lookup_column(
        country_lulc_lookup,
        [
            "zone_id",
            "country_lulc_zone_id",
        ],
        "country × LULC zone identifier",
    )

    country_id_column = infer_lookup_column(
        country_lulc_lookup,
        [
            "country_seq",
            "country_id",
        ],
        "country identifier in country_lulc_lookup.csv",
    )

    country_lookup_id = infer_lookup_column(
        country_lookup,
        [
            "country_seq",
            "country_id",
            "zone_id",
        ],
        "country identifier in country_lookup.csv",
    )

    if "lulc_name" not in country_lulc_lookup.columns:
        raise KeyError(
            "country_lulc_lookup.csv must contain 'lulc_name'. "
            f"Available columns: {country_lulc_lookup.columns.tolist()}"
        )

    lookup = country_lulc_lookup.copy()
    lookup["lulc_name"] = clean_text(
        lookup["lulc_name"]
    )

    lookup = lookup.loc[
        ~lookup["lulc_name"].isin(
            EXCLUDED_LULC_NAMES
        )
    ].copy()

    # Add country attributes when they are absent from the combined lookup.
    country_columns_to_add = [
        column
        for column in [
            "ISO_A3",
            "NAM_0",
        ]
        if (
            column in country_lookup.columns
            and column not in lookup.columns
        )
    ]

    if country_columns_to_add:
        country_attributes = country_lookup[
            [country_lookup_id]
            + country_columns_to_add
        ].drop_duplicates(
            subset=[country_lookup_id]
        )

        if country_lookup_id != country_id_column:
            country_attributes = (
                country_attributes.rename(
                    columns={
                        country_lookup_id:
                        country_id_column
                    }
                )
            )

        lookup = lookup.merge(
            country_attributes,
            on=country_id_column,
            how="left",
            validate="many_to_one",
        )

    required_attributes = [
        country_id_column,
        "lulc_code",
        "lulc_name",
    ]

    missing_attributes = [
        column
        for column in required_attributes
        if column not in lookup.columns
    ]

    if missing_attributes:
        raise KeyError(
            "The combined lookup is missing required attributes: "
            f"{missing_attributes}"
        )

    lookup[zone_id_column] = pd.to_numeric(
        lookup[zone_id_column],
        errors="coerce",
    )

    lookup = lookup.dropna(
        subset=[zone_id_column]
    ).copy()

    lookup[zone_id_column] = (
        lookup[zone_id_column].astype(np.int64)
    )

    if lookup[zone_id_column].duplicated().any():
        duplicates = (
            lookup.loc[
                lookup[zone_id_column].duplicated(
                    keep=False
                ),
                zone_id_column,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "country_lulc_lookup.csv contains duplicate zone identifiers: "
            f"{duplicates[:20]}"
        )

    return lookup, zone_id_column


# =============================================================================
# 6. BLOCK-WISE ZONE COUNTS
# =============================================================================

def count_total_zone_pixels(
    zone_raster_path: Path,
    allowed_zone_ids: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """
    Count all included country × LULC pixels.

    These counts provide the stable total included land-cover denominator for
    each country × LULC zone. Sparse/bare and Water/snow/ice are excluded
    through allowed_zone_ids.
    """

    allowed_zone_ids = np.asarray(
        allowed_zone_ids,
        dtype=np.int64,
    )

    allowed_zone_ids.sort()

    counts: dict[int, int] = {}

    with rasterio.open(
        zone_raster_path
    ) as zone_src:
        pixel_area_m2 = abs(
            zone_src.transform.a
            * zone_src.transform.e
        )

        for _, window in zone_src.block_windows(1):
            zone = zone_src.read(
                1,
                window=window,
                masked=True,
            )

            zone_mask = np.ma.getmaskarray(zone)

            valid = (
                (~zone_mask)
                & (zone.data > 0)
                & np.isin(
                    zone.data,
                    allowed_zone_ids,
                )
            )

            if not np.any(valid):
                continue

            values = zone.data[valid].astype(
                np.int64,
                copy=False,
            )

            unique_ids, block_counts = np.unique(
                values,
                return_counts=True,
            )

            for zone_id, count in zip(
                unique_ids,
                block_counts,
            ):
                counts[int(zone_id)] = (
                    counts.get(
                        int(zone_id),
                        0,
                    )
                    + int(count)
                )

    result = pd.DataFrame(
        {
            "zone_id": list(counts.keys()),
            "total_valid_pixel_count": list(
                counts.values()
            ),
        }
    )

    pixel_area_ha = pixel_area_m2 / 10_000.0

    result["total_valid_area_ha"] = (
        result["total_valid_pixel_count"]
        * pixel_area_ha
    )

    result["total_valid_area_mha"] = (
        result["total_valid_area_ha"]
        / 1_000_000.0
    )

    return result, pixel_area_ha


def count_negative_pixels_by_zone(
    trend_raster_path: Path,
    zone_raster_path: Path,
    allowed_zone_ids: np.ndarray,
    statistic_key: str,
) -> pd.DataFrame:
    """
    Count negative Trend pixels by included country × LULC zone.
    """

    allowed_zone_ids = np.asarray(
        allowed_zone_ids,
        dtype=np.int64,
    )

    allowed_zone_ids.sort()

    counts: dict[int, int] = {}

    with (
        rasterio.open(
            zone_raster_path
        ) as zone_src,
        rasterio.open(
            trend_raster_path
        ) as trend_src,
    ):
        assert_same_grid(
            zone_src,
            trend_src,
            trend_raster_path.name,
        )

        pixel_area_m2 = abs(
            zone_src.transform.a
            * zone_src.transform.e
        )

        for _, window in zone_src.block_windows(1):
            zone = zone_src.read(
                1,
                window=window,
                masked=True,
            )

            trend = trend_src.read(
                1,
                window=window,
                masked=True,
            )

            valid = (
                (~np.ma.getmaskarray(zone))
                & (~np.ma.getmaskarray(trend))
                & (zone.data > 0)
                & np.isin(
                    zone.data,
                    allowed_zone_ids,
                )
                & (trend.data == TARGET_VALUE)
            )

            if not np.any(valid):
                continue

            values = zone.data[valid].astype(
                np.int64,
                copy=False,
            )

            unique_ids, block_counts = np.unique(
                values,
                return_counts=True,
            )

            for zone_id, count in zip(
                unique_ids,
                block_counts,
            ):
                counts[int(zone_id)] = (
                    counts.get(
                        int(zone_id),
                        0,
                    )
                    + int(count)
                )

    result = pd.DataFrame(
        {
            "zone_id": list(counts.keys()),
            "negative_pixel_count": list(
                counts.values()
            ),
        }
    )

    pixel_area_ha = (
        pixel_area_m2 / 10_000.0
    )

    result["negative_area_ha"] = (
        result["negative_pixel_count"]
        * pixel_area_ha
    )

    result["negative_area_mha"] = (
        result["negative_area_ha"]
        / 1_000_000.0
    )

    result["window_key"] = WINDOW_KEY
    result["window"] = WINDOW
    result["statistic_key"] = statistic_key
    result["applied_statistic"] = (
        STATISTIC_LABELS[statistic_key]
    )
    result["statistic_order"] = (
        STATISTIC_ORDER[statistic_key]
    )
    result["trend_direction"] = "Negative"

    return result


# =============================================================================
# 7. COUNTRY × LULC BASE TABLE
# =============================================================================

def build_country_lulc_summary(
    lookup: pd.DataFrame,
    zone_id_column: str,
) -> pd.DataFrame:
    allowed_zone_ids = (
        lookup[zone_id_column]
        .astype(np.int64)
        .unique()
    )

    print(
        "Counting total included country × LULC area..."
    )

    denominators, _ = count_total_zone_pixels(
        COUNTRY_LULC_RASTER,
        allowed_zone_ids,
    )

    denominators = denominators.rename(
        columns={
            "zone_id": zone_id_column
        }
    )

    # Keep all included lookup zones, including zones with zero raster pixels.
    zone_base = lookup.merge(
        denominators,
        on=zone_id_column,
        how="left",
        validate="one_to_one",
    )

    for column in [
        "total_valid_pixel_count",
        "total_valid_area_ha",
        "total_valid_area_mha",
    ]:
        zone_base[column] = (
            zone_base[column]
            .fillna(0)
        )

    all_statistics = []

    for statistic_key, raster_path in TREND_RASTERS.items():
        print(
            f"Counting {statistic_key} negative pixels "
            "by country × LULC..."
        )

        negative = count_negative_pixels_by_zone(
            raster_path,
            COUNTRY_LULC_RASTER,
            allowed_zone_ids,
            statistic_key,
        )

        negative = negative.rename(
            columns={
                "zone_id": zone_id_column
            }
        )

        summary = zone_base.merge(
            negative,
            on=zone_id_column,
            how="left",
            validate="one_to_one",
        )

        for column in [
            "negative_pixel_count",
            "negative_area_ha",
            "negative_area_mha",
        ]:
            summary[column] = (
                summary[column]
                .fillna(0)
            )

        summary["window_key"] = WINDOW_KEY
        summary["window"] = WINDOW
        summary["statistic_key"] = statistic_key
        summary["applied_statistic"] = (
            STATISTIC_LABELS[statistic_key]
        )
        summary["statistic_order"] = (
            STATISTIC_ORDER[statistic_key]
        )
        summary["trend_direction"] = "Negative"

        summary[
            "negative_share_of_valid_area_pct"
        ] = np.where(
            summary[
                "total_valid_pixel_count"
            ] > 0,
            (
                100.0
                * summary[
                    "negative_pixel_count"
                ]
                / summary[
                    "total_valid_pixel_count"
                ]
            ),
            np.nan,
        )

        summary["reporting_unit_type"] = (
            "country_lulc"
        )

        all_statistics.append(summary)

    result = pd.concat(
        all_statistics,
        ignore_index=True,
        sort=False,
    )

    preferred = [
        zone_id_column,
        "country_seq",
        "ISO_A3",
        "NAM_0",
        "lulc_code",
        "lulc_name",
        "window_key",
        "window",
        "statistic_key",
        "applied_statistic",
        "statistic_order",
        "trend_direction",
        "total_valid_pixel_count",
        "total_valid_area_ha",
        "total_valid_area_mha",
        "negative_pixel_count",
        "negative_area_ha",
        "negative_area_mha",
        "negative_share_of_valid_area_pct",
        "reporting_unit_type",
    ]

    result = result[
        [
            column
            for column in preferred
            if column in result.columns
        ]
        + [
            column
            for column in result.columns
            if column not in preferred
        ]
    ]

    return result.sort_values(
        [
            "NAM_0",
            "lulc_code",
            "statistic_order",
        ],
        kind="stable",
    )


# =============================================================================
# 8. AGGREGATE TO COUNTRY AND LULC
# =============================================================================

def aggregate_reporting_level(
    country_lulc_summary: pd.DataFrame,
    group_columns: list[str],
    reporting_unit_type: str,
) -> pd.DataFrame:
    """
    Aggregate numerator and denominator from country × LULC zones.

    Because the same denominator appears once per statistic in the long table,
    aggregation is carried out separately within each statistic.
    """

    grouping = (
        group_columns
        + [
            "window_key",
            "window",
            "statistic_key",
            "applied_statistic",
            "statistic_order",
            "trend_direction",
        ]
    )

    result = (
        country_lulc_summary.groupby(
            grouping,
            as_index=False,
            observed=True,
        )
        .agg(
            total_valid_pixel_count=(
                "total_valid_pixel_count",
                "sum",
            ),
            total_valid_area_ha=(
                "total_valid_area_ha",
                "sum",
            ),
            total_valid_area_mha=(
                "total_valid_area_mha",
                "sum",
            ),
            negative_pixel_count=(
                "negative_pixel_count",
                "sum",
            ),
            negative_area_ha=(
                "negative_area_ha",
                "sum",
            ),
            negative_area_mha=(
                "negative_area_mha",
                "sum",
            ),
        )
    )

    result[
        "negative_share_of_valid_area_pct"
    ] = np.where(
        result["total_valid_pixel_count"] > 0,
        (
            100.0
            * result["negative_pixel_count"]
            / result["total_valid_pixel_count"]
        ),
        np.nan,
    )

    result["reporting_unit_type"] = (
        reporting_unit_type
    )

    return result.sort_values(
        group_columns
        + ["statistic_order"],
        kind="stable",
    )


# =============================================================================
# 9. RETENTION RELATIVE TO POTENTIAL
# =============================================================================

def add_retention_from_potential(
    table: pd.DataFrame,
    unit_columns: list[str],
) -> pd.DataFrame:
    """
    Add a secondary diagnostic showing each statistic relative to Potential.

    The primary percentage remains negative_share_of_valid_area_pct, based on
    the total included land-cover denominator.
    """

    potential = (
        table.loc[
            table["statistic_key"]
            == "potential",
            unit_columns
            + [
                "negative_pixel_count",
                "negative_area_ha",
                "negative_area_mha",
            ],
        ]
        .rename(
            columns={
                "negative_pixel_count":
                "potential_negative_pixel_count",
                "negative_area_ha":
                "potential_negative_area_ha",
                "negative_area_mha":
                "potential_negative_area_mha",
            }
        )
    )

    result = table.merge(
        potential,
        on=unit_columns,
        how="left",
        validate="many_to_one",
    )

    result[
        "retention_from_potential_pct"
    ] = np.where(
        result[
            "potential_negative_pixel_count"
        ] > 0,
        (
            100.0
            * result["negative_pixel_count"]
            / result[
                "potential_negative_pixel_count"
            ]
        ),
        np.nan,
    )

    return result


# =============================================================================
# 10. WIDE TABLES
# =============================================================================

def build_wide_table(
    long_table: pd.DataFrame,
    index_columns: list[str],
) -> pd.DataFrame:
    """
    Produce a plotting-ready wide table with Mha and percentages.
    """

    denominator = (
        long_table[
            index_columns
            + [
                "total_valid_pixel_count",
                "total_valid_area_ha",
                "total_valid_area_mha",
            ]
        ]
        .drop_duplicates(
            subset=index_columns
        )
    )

    mha = (
        long_table.pivot_table(
            index=index_columns,
            columns="applied_statistic",
            values="negative_area_mha",
            aggfunc="first",
            fill_value=0.0,
        )
        .add_suffix("_mha")
        .reset_index()
    )

    share = (
        long_table.pivot_table(
            index=index_columns,
            columns="applied_statistic",
            values="negative_share_of_valid_area_pct",
            aggfunc="first",
        )
        .add_suffix("_pct_valid_area")
        .reset_index()
    )

    retention = (
        long_table.pivot_table(
            index=index_columns,
            columns="applied_statistic",
            values="retention_from_potential_pct",
            aggfunc="first",
        )
        .add_suffix("_pct_potential")
        .reset_index()
    )

    result = (
        denominator
        .merge(
            mha,
            on=index_columns,
            how="left",
        )
        .merge(
            share,
            on=index_columns,
            how="left",
        )
        .merge(
            retention,
            on=index_columns,
            how="left",
        )
    )

    return result


# =============================================================================
# 11. OPTIONAL COUNTRY-RASTER ALIGNMENT CHECK
# =============================================================================

def validate_country_raster_alignment() -> None:
    """
    Confirm that the supplied country raster shares the same grid.

    Country summaries are intentionally aggregated from country × LULC zones
    so that excluded LULC classes are removed consistently from numerators and
    denominators.
    """

    with (
        rasterio.open(
            COUNTRY_LULC_RASTER
        ) as reference,
        rasterio.open(
            COUNTRY_RASTER
        ) as candidate,
    ):
        assert_same_grid(
            reference,
            candidate,
            COUNTRY_RASTER.name,
        )



# =============================================================================
# 15. SUPPLEMENTARY FIGURE: STATISTICAL SUPPORT + LULC-TRANSITION CONTEXT
# =============================================================================

def resolve_lulc_transition_class_raster() -> Path:
    """Return the current full-period LULC-transition classification raster."""
    if LULC_TRANSITION_CLASS_RASTER.exists():
        return LULC_TRANSITION_CLASS_RASTER

    if LULC_TRANSITION_CLASS_RASTER_FALLBACK.exists():
        print(
            "[figure warning] Using legacy LULC transition-class raster name:\n"
            f"  {LULC_TRANSITION_CLASS_RASTER_FALLBACK}"
        )
        return LULC_TRANSITION_CLASS_RASTER_FALLBACK

    raise FileNotFoundError(
        "Neither the preferred nor fallback LULC transition-class raster "
        "was found:\n"
        f"  {LULC_TRANSITION_CLASS_RASTER}\n"
        f"  {LULC_TRANSITION_CLASS_RASTER_FALLBACK}"
    )


def build_continental_support_summary(
    country_lulc_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the existing country × LULC summaries to one Africa-wide row
    per statistical-support formulation.

    This deliberately uses the summaries produced above from TREND_RASTERS,
    so panel (a) cannot drift from the most recently configured Trend rasters.
    """
    grouped = (
        country_lulc_table.groupby(
            [
                "statistic_key",
                "applied_statistic",
                "statistic_order",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            total_valid_pixel_count=("total_valid_pixel_count", "sum"),
            total_valid_area_ha=("total_valid_area_ha", "sum"),
            negative_pixel_count=("negative_pixel_count", "sum"),
            negative_area_ha=("negative_area_ha", "sum"),
        )
    )

    grouped["total_valid_area_mha"] = (
        grouped["total_valid_area_ha"] / 1_000_000.0
    )
    grouped["negative_area_mha"] = (
        grouped["negative_area_ha"] / 1_000_000.0
    )
    grouped["negative_share_of_valid_area_pct"] = np.where(
        grouped["total_valid_area_ha"] > 0,
        100.0
        * grouped["negative_area_ha"]
        / grouped["total_valid_area_ha"],
        np.nan,
    )

    grouped = (
        grouped.set_index("statistic_key")
        .reindex(PLOT_STATISTIC_ORDER)
        .reset_index()
    )

    missing = grouped.loc[
        grouped["negative_area_ha"].isna(), "statistic_key"
    ].tolist()
    if missing:
        raise ValueError(
            "Continental support summary is missing required statistics: "
            f"{missing}"
        )

    # All formulations must use the same included vegetated denominator.
    denominators = grouped["total_valid_area_ha"].to_numpy(dtype=float)
    if not np.allclose(
        denominators,
        denominators[0],
        rtol=0.0,
        atol=max(1.0, denominators[0] * 1e-10),
    ):
        raise ValueError(
            "The statistical formulations do not share the same vegetated "
            "denominator. Do not plot until the summaries are reconciled."
        )

    # Logical nesting checks. The joint mask is defined as the intersection of
    # OLS, Mann–Kendall and Newey–West support; it therefore cannot exceed any
    # of those individual masks. Each supported mask must also be contained in
    # the potential negative-slope population.
    areas = grouped.set_index("statistic_key")["negative_area_ha"]
    potential = float(areas["potential"])
    joint = float(areas["strict_joint"])

    violations = []
    for key in ["ols", "mann_kendall", "newey_west"]:
        individual = float(areas[key])
        if individual > potential + 1e-6:
            violations.append(
                f"{key} ({individual / 1e6:.3f} Mha) exceeds potential "
                f"({potential / 1e6:.3f} Mha)"
            )
        if joint > individual + 1e-6:
            violations.append(
                f"joint ({joint / 1e6:.3f} Mha) exceeds {key} "
                f"({individual / 1e6:.3f} Mha)"
            )

    if violations:
        details = "\n  - ".join(violations)
        raise ValueError(
            "Trend statistical-support rasters are internally inconsistent. "
            "Because the joint mask is an intersection, the following cannot "
            f"occur:\n  - {details}\n"
            "Rebuild or verify the current Trend rasters before generating "
            "the figure."
        )

    return grouped


def count_transition_context_for_declining_pixels(
    allowed_zone_ids: np.ndarray,
    support_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count full-period LULC-transition classes inside each declining Trend mask.

    The same included country × LULC domain used by the summary workflow is
    imposed here, so sparse/bare and water/snow/ice remain excluded.
    """
    transition_path = resolve_lulc_transition_class_raster()
    require_files(
        [
            COUNTRY_LULC_RASTER,
            transition_path,
            *[TREND_RASTERS[key] for key in PLOT_STATISTIC_ORDER],
        ]
    )

    allowed_zone_ids = np.asarray(allowed_zone_ids, dtype=np.int64)
    allowed_zone_ids.sort()

    counts = {
        key: np.zeros(len(TRANSITION_ORDER) + 1, dtype=np.int64)
        for key in PLOT_STATISTIC_ORDER
    }
    total_negative_direct = {
        key: 0 for key in PLOT_STATISTIC_ORDER
    }

    with ExitStack() as stack:
        zone_src = stack.enter_context(rasterio.open(COUNTRY_LULC_RASTER))
        transition_src = stack.enter_context(rasterio.open(transition_path))
        trend_srcs = {
            key: stack.enter_context(rasterio.open(TREND_RASTERS[key]))
            for key in PLOT_STATISTIC_ORDER
        }

        assert_same_grid(zone_src, transition_src, transition_path.name)
        for key, src in trend_srcs.items():
            assert_same_grid(zone_src, src, TREND_RASTERS[key].name)

        pixel_area_ha = abs(
            zone_src.transform.a * zone_src.transform.e
        ) / 10_000.0

        for _, window in zone_src.block_windows(1):
            zone = zone_src.read(1, window=window, masked=True)
            transition = transition_src.read(1, window=window, masked=True)

            included_domain = (
                (~np.ma.getmaskarray(zone))
                & (zone.data > 0)
                & np.isin(zone.data, allowed_zone_ids)
            )

            transition_valid = (
                (~np.ma.getmaskarray(transition))
                & np.isin(transition.data, TRANSITION_ORDER)
            )

            for key, src in trend_srcs.items():
                trend = src.read(1, window=window, masked=True)
                negative = (
                    included_domain
                    & (~np.ma.getmaskarray(trend))
                    & (trend.data == TARGET_VALUE)
                )

                total_negative_direct[key] += int(np.count_nonzero(negative))

                use = negative & transition_valid
                if not np.any(use):
                    continue

                block_counts = np.bincount(
                    transition.data[use].astype(np.int64),
                    minlength=max(TRANSITION_ORDER) + 1,
                )
                counts[key][:] += block_counts[: len(counts[key])]

    # Verify that direct raster counts reproduce panel (a)'s summary counts.
    summary_counts = support_summary.set_index("statistic_key")[
        "negative_pixel_count"
    ]
    mismatches = []
    for key in PLOT_STATISTIC_ORDER:
        summary_count = int(round(float(summary_counts[key])))
        direct_count = int(total_negative_direct[key])
        if summary_count != direct_count:
            mismatches.append(
                f"{key}: summary={summary_count:,}, raster={direct_count:,}"
            )

    if mismatches:
        raise ValueError(
            "Panel (a) summary counts do not match the directly read Trend "
            "rasters:\n  - "
            + "\n  - ".join(mismatches)
        )

    rows = []
    for key in PLOT_STATISTIC_ORDER:
        classified_count = int(sum(counts[key][code] for code in TRANSITION_ORDER))
        total_count = int(total_negative_direct[key])
        coverage_pct = (
            100.0 * classified_count / total_count
            if total_count > 0 else np.nan
        )

        if total_count > 0 and coverage_pct < 99.0:
            print(
                "[figure warning] LULC-transition coverage for "
                f"{key} is {coverage_pct:.2f}% of declining pixels. "
                "Panel (b) will be normalised within pixels having a valid "
                "transition class."
            )

        for code in TRANSITION_ORDER:
            count = int(counts[key][code])
            rows.append(
                {
                    "statistic_key": key,
                    "statistical_support_rule": PLOT_STATISTIC_LABELS[key].replace("\n", " "),
                    "transition_code": code,
                    "transition_class": TRANSITION_LABELS[code],
                    "transition_detail": TRANSITION_DETAILS[code],
                    "transition_pixel_count": count,
                    "transition_area_mha": count * pixel_area_ha / 1_000_000.0,
                    "transition_pct_within_classified_declining": (
                        100.0 * count / classified_count
                        if classified_count > 0 else np.nan
                    ),
                    "declining_pixel_count": total_count,
                    "transition_classified_declining_pixel_count": classified_count,
                    "transition_class_coverage_pct": coverage_pct,
                }
            )

    return pd.DataFrame(rows)


def make_trend_support_lulc_figure(
    support_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
) -> None:
    """Create the two-panel supplementary Trend support/LULC figure."""
    plt.rcParams["font.family"] = FONT_FAMILY

    support = (
        support_summary.set_index("statistic_key")
        .reindex(PLOT_STATISTIC_ORDER)
    )

    composition = (
        transition_summary.pivot(
            index="statistic_key",
            columns="transition_code",
            values="transition_pct_within_classified_declining",
        )
        .reindex(index=PLOT_STATISTIC_ORDER, columns=TRANSITION_ORDER)
        .fillna(0.0)
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [0.88, 1.45], "wspace": 0.28},
        facecolor="white",
    )

    # Panel (a): absolute area and common-domain share.
    x = np.arange(len(PLOT_STATISTIC_ORDER))
    total_mha = support["negative_area_mha"].to_numpy(dtype=float)
    total_pct = support["negative_share_of_valid_area_pct"].to_numpy(dtype=float)

    bars = ax_a.bar(
        x,
        total_mha,
        width=0.68,
        color=FUNNEL_BAR_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )

    max_area = max(float(np.nanmax(total_mha)), 1.0)
    for bar, area, pct in zip(bars, total_mha, total_pct):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_area * 0.022,
            f"{area:.1f} Mha\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.7,
            fontweight="semibold",
            linespacing=1.15,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [PLOT_STATISTIC_LABELS[key] for key in PLOT_STATISTIC_ORDER],
        fontsize=8.2,
        rotation=30,
        ha="right",
    )
    ax_a.set_ylabel("Declining area (Mha)", fontsize=10)
    ax_a.set_title(
        "(a) Declining area under statistical-support rules",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=9,
    )
    ax_a.grid(axis="y", linewidth=0.45, alpha=0.30)
    ax_a.set_axisbelow(True)
    ax_a.set_ylim(0, max_area * 1.20)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)

    # Panel (b): 100% stacked transition context.
    y = np.arange(len(PLOT_STATISTIC_ORDER))
    left = np.zeros(len(PLOT_STATISTIC_ORDER), dtype=float)

    for code in TRANSITION_ORDER:
        values = composition[code].to_numpy(dtype=float)
        ax_b.barh(
            y,
            values,
            left=left,
            height=0.64,
            color=TRANSITION_COLORS[code],
            edgecolor="white",
            linewidth=0.7,
            label=TRANSITION_LABELS[code],
        )

        for row_index, (segment_left, value) in enumerate(zip(left, values)):
            if value >= 4.0:
                text_color = "white" if code in [1, 4] else "black"
                ax_b.text(
                    segment_left + value / 2,
                    row_index,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="semibold",
                    color=text_color,
                )
        left += values

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(
        [PLOT_STATISTIC_LABELS[key].replace("\n", " ") for key in PLOT_STATISTIC_ORDER],
        fontsize=8.8,
    )
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Composition of declining area (%)", fontsize=10)
    ax_b.set_title(
        "(b) Broad LULC-transition context of declining pixels",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=9,
    )
    ax_b.grid(axis="x", linewidth=0.45, alpha=0.30)
    ax_b.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax_b.spines[spine].set_visible(False)

    handles = [
        Patch(
            facecolor=TRANSITION_COLORS[code],
            edgecolor="none",
            label=f"{TRANSITION_LABELS[code]} ({TRANSITION_DETAILS[code]})",
        )
        for code in TRANSITION_ORDER
    ]
    ax_b.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        "Statistical support and LULC-transition context of declining "
        "productivity, 2001–2022",
        fontsize=15.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        (
            "Panel (a) percentages use the common valid vegetated domain; "
            "panel (b) is normalised within declining pixels with a valid "
            "LULC-transition class."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.23)

    if WRITE_TREND_SUPPORT_LULC_PNG:
        fig.savefig(
            FIGURE_OUTPUT_PNG,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.08,
        )
        print(f"[write] {FIGURE_OUTPUT_PNG}")

    if WRITE_TREND_SUPPORT_LULC_PDF:
        fig.savefig(
            FIGURE_OUTPUT_PDF,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.08,
        )
        print(f"[write] {FIGURE_OUTPUT_PDF}")

    if SHOW_TREND_SUPPORT_LULC_FIGURE:
        plt.show()

    plt.close(fig)



# =============================================================================
# 16. MAIN
# =============================================================================

def main() -> None:
    """Run the full-period Trend statistical-support diagnostic workflow."""

    run_configuration = pd.DataFrame(
        [
            ("WINDOW_KEY", WINDOW_KEY),
            ("WINDOW", WINDOW),
            ("TARGET_VALUE", TARGET_VALUE),
            ("EXCLUDED_LULC_NAMES", sorted(EXCLUDED_LULC_NAMES)),
            ("TREND_ROOT", TREND_ROOT),
            ("REUSABLE_ZONE_DIR", REUSABLE_ZONE_DIR),
            ("OUTPUT_DIR", OUTPUT_DIR),
            ("CREATE_TREND_SUPPORT_LULC_FIGURE", CREATE_TREND_SUPPORT_LULC_FIGURE),
            ("WRITE_TREND_SUPPORT_LULC_PNG", WRITE_TREND_SUPPORT_LULC_PNG),
            ("WRITE_TREND_SUPPORT_LULC_PDF", WRITE_TREND_SUPPORT_LULC_PDF),
            ("WRITE_TREND_SUPPORT_LULC_TABLE", WRITE_TREND_SUPPORT_LULC_TABLE),
            ("WRITE_EXCEL_SUMMARY", WRITE_EXCEL_SUMMARY),
            (
                "PRIMARY_DENOMINATOR",
                "included 2022 country × LULC reporting-domain pixels; "
                "Sparse/bare and Water/snow/ice excluded",
            ),
            (
                "RETENTION_DENOMINATOR",
                "potential-negative pixels within the same reporting unit",
            ),
        ],
        columns=["setting", "value"],
    )
    run_configuration.to_csv(
        OUTPUT_DIR / "trend_support_run_configuration_2001_2022.csv",
        index=False,
    )

    # =============================================================================
    # 12. RUN
    # =============================================================================

    required_paths = (
        list(TREND_RASTERS.values())
        + [
            COUNTRY_RASTER,
            COUNTRY_LULC_RASTER,
            COUNTRY_LOOKUP,
            COUNTRY_LULC_LOOKUP,
        ]
    )

    require_files(required_paths)

    validate_country_raster_alignment()

    country_lookup = pd.read_csv(
        COUNTRY_LOOKUP
    )

    country_lulc_lookup = pd.read_csv(
        COUNTRY_LULC_LOOKUP
    )

    lookup, zone_id_column = (
        prepare_country_lulc_lookup(
            country_lookup,
            country_lulc_lookup,
        )
    )

    country_lulc_summary = (
        build_country_lulc_summary(
            lookup,
            zone_id_column,
        )
    )

    country_summary = aggregate_reporting_level(
        country_lulc_summary,
        [
            "country_seq",
            "ISO_A3",
            "NAM_0",
        ],
        "country",
    )

    lulc_summary = aggregate_reporting_level(
        country_lulc_summary,
        [
            "lulc_code",
            "lulc_name",
        ],
        "lulc",
    )

    country_lulc_summary = (
        add_retention_from_potential(
            country_lulc_summary,
            [
                zone_id_column,
                "window_key",
                "window",
            ],
        )
    )

    country_summary = add_retention_from_potential(
        country_summary,
        [
            "country_seq",
            "ISO_A3",
            "NAM_0",
            "window_key",
            "window",
        ],
    )

    lulc_summary = add_retention_from_potential(
        lulc_summary,
        [
            "lulc_code",
            "lulc_name",
            "window_key",
            "window",
        ],
    )

    country_wide = build_wide_table(
        country_summary,
        [
            "country_seq",
            "ISO_A3",
            "NAM_0",
            "window",
        ],
    )

    country_lulc_wide = build_wide_table(
        country_lulc_summary,
        [
            zone_id_column,
            "country_seq",
            "ISO_A3",
            "NAM_0",
            "lulc_code",
            "lulc_name",
            "window",
        ],
    )

    lulc_wide = build_wide_table(
        lulc_summary,
        [
            "lulc_code",
            "lulc_name",
            "window",
        ],
    )


    # =============================================================================
    # 13. EXPORT
    # =============================================================================

    country_summary.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_country_2001_2022_long.csv",
        index=False,
    )

    country_wide.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_country_2001_2022_wide.csv",
        index=False,
    )

    country_lulc_summary.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_country_LULC_2001_2022_long.csv",
        index=False,
    )

    country_lulc_wide.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_country_LULC_2001_2022_wide.csv",
        index=False,
    )

    lulc_summary.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_LULC_2001_2022_long.csv",
        index=False,
    )

    lulc_wide.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_by_LULC_2001_2022_wide.csv",
        index=False,
    )


    if WRITE_EXCEL_SUMMARY:
        with pd.ExcelWriter(
            OUTPUT_DIR
            / "trend_negative_share_summary_2001_2022.xlsx",
            engine="openpyxl",
        ) as writer:
            country_summary.to_excel(
                writer,
                sheet_name="Country_long",
                index=False,
            )

            country_wide.to_excel(
                writer,
                sheet_name="Country_wide",
                index=False,
            )

            country_lulc_summary.to_excel(
                writer,
                sheet_name="Country_LULC_long",
                index=False,
            )

            country_lulc_wide.to_excel(
                writer,
                sheet_name="Country_LULC_wide",
                index=False,
            )

            lulc_summary.to_excel(
                writer,
                sheet_name="LULC_long",
                index=False,
            )

            lulc_wide.to_excel(
                writer,
                sheet_name="LULC_wide",
                index=False,
            )



    # =============================================================================
    # 14. DIAGNOSTICS
    # =============================================================================

    diagnostic_columns = [
        "reporting_unit_type",
        "applied_statistic",
        "negative_share_of_valid_area_pct",
        "retention_from_potential_pct",
    ]

    diagnostics = pd.concat(
        [
            country_summary.assign(
                diagnostic_unit=country_summary[
                    "NAM_0"
                ]
            ),
            lulc_summary.assign(
                diagnostic_unit=lulc_summary[
                    "lulc_name"
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    diagnostics = diagnostics.loc[
        (
            diagnostics[
                "negative_share_of_valid_area_pct"
            ] > 100
        )
        | (
            diagnostics[
                "retention_from_potential_pct"
            ] > 100
        ),
        [
            "reporting_unit_type",
            "diagnostic_unit",
            "applied_statistic",
            "negative_share_of_valid_area_pct",
            "retention_from_potential_pct",
        ],
    ]

    diagnostics.to_csv(
        OUTPUT_DIR
        / "trend_negative_share_diagnostics_above_100pct.csv",
        index=False,
    )


    if CREATE_TREND_SUPPORT_LULC_FIGURE:
        print("\nCreating statistical-support and LULC-transition supplementary figure...")

        support_summary = build_continental_support_summary(country_lulc_summary)
        allowed_zone_ids_for_figure = (
            lookup[zone_id_column].astype(np.int64).unique()
        )
        transition_summary = count_transition_context_for_declining_pixels(
            allowed_zone_ids_for_figure,
            support_summary,
        )

        figure_table = transition_summary.merge(
            support_summary[
                [
                    "statistic_key",
                    "negative_area_mha",
                    "negative_share_of_valid_area_pct",
                    "total_valid_area_mha",
                ]
            ],
            on="statistic_key",
            how="left",
            validate="many_to_one",
        )

        if WRITE_TREND_SUPPORT_LULC_TABLE:
            figure_table.to_csv(FIGURE_OUTPUT_TABLE, index=False)
            print(f"[write] {FIGURE_OUTPUT_TABLE}")

        print("\nFigure support summary:")
        print(
            support_summary[
                [
                    "statistic_key",
                    "negative_area_mha",
                    "negative_share_of_valid_area_pct",
                ]
            ].to_string(index=False)
        )

        print("\nLULC-transition coverage of declining pixels:")
        coverage = (
            transition_summary[
                ["statistic_key", "transition_class_coverage_pct"]
            ]
            .drop_duplicates()
            .set_index("statistic_key")
            .reindex(PLOT_STATISTIC_ORDER)
        )
        print(coverage.to_string())

        make_trend_support_lulc_figure(
            support_summary,
            transition_summary,
        )


    # =============================================================================
    # 16. COMPLETION REPORT
    # =============================================================================

    print("\n" + "=" * 88)
    print("TREND NEGATIVE SHARE SUMMARY COMPLETE")
    print("=" * 88)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    print(
        "\nExcluded LULC classes:\n  - "
        + "\n  - ".join(
            sorted(EXCLUDED_LULC_NAMES)
        )
    )

    print(
        "\nPrimary denominator:"
        "\n  Total included country × LULC area from the 2022 zone raster."
    )

    print(
        "\nPrimary percentage:"
        "\n  negative_share_of_valid_area_pct = "
        "negative pixels / total included LULC pixels × 100"
    )

    print(
        "\nSecondary diagnostic:"
        "\n  retention_from_potential_pct = "
        "negative pixels under statistic / Potential-negative pixels × 100"
    )

    print("\nTotal included area by country: ")
    print(
        country_summary.loc[
            country_summary["statistic_key"]
            == "potential"
        ][
            [
                "NAM_0",
                "total_valid_area_mha",
            ]
        ]
        .sort_values(
            "total_valid_area_mha",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nAfrica-wide included LULC area and negative shares:")
    lulc_report = lulc_summary[
        [
            "lulc_name",
            "applied_statistic",
            "total_valid_area_mha",
            "negative_area_mha",
            "negative_share_of_valid_area_pct",
            "retention_from_potential_pct",
        ]
    ].copy()

    print(
        lulc_report.to_string(
            index=False
        )
    )

    print(
        f"\nDiagnostic rows above 100%: {len(diagnostics)}"
    )

    
if __name__ == "__main__":
    main()
