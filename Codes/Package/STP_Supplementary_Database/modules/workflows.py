from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from .core import (
    agreement_metrics,
    categorical_summary,
    create_combined_zone,
    crosstab,
    prepare_vector,
    rasterize_gdf,
)


# =============================================================================
# 1. LOOKUPS
# =============================================================================

TRAJECTORY = {
    11: "Persistent decline",
    12: "Decline followed by stabilisation",
    13: "Recovery/reversal to improvement",
    21: "Recent-onset decline",
    22: "No confirmed direction",
    23: "Recent-onset improvement",
    31: "Reversal to decline",
    32: "Improvement followed by stabilisation",
    33: "Persistent improvement",
}


GROUPS = {
    1: "Compounding concern",
    2: "Persistent low condition",
    3: "Recovering but still low",
    4: "Emerging concern",
    5: "Stable intermediate condition",
    6: "Improving condition",
    7: "High condition with historical loss",
    8: "Mixed evidence",
}


STATE_FORMULATION_TITLES = {
    "ecological_parametric": (
        "Ecological–parametric formulation; "
        "2022 annual mean NDVI relative to 2002–2006 "
        "land-cover × ecoregion benchmarks"
    ),
    "ecological_quantile": (
        "Ecological–quantile formulation; "
        "2022 annual mean NDVI relative to 2002–2006 "
        "land-cover × ecoregion benchmarks"
    ),
    "policy_parametric": (
        "Policy-facing parametric formulation; "
        "2022 annual mean NDVI relative to 2002–2006 "
        "country × land-cover × ecoregion benchmarks"
    ),
    "policy_quantile": (
        "Policy-facing quantile formulation; "
        "2022 annual mean NDVI relative to 2002–2006 "
        "country × land-cover × ecoregion benchmarks"
    ),
}


STATE_CLASS_ORDER = ["Poor", "Fair", "Good"]

TREND_TRAJECTORY_ORDER = [
    "Persistent decline",
    "Recent-onset decline",
    "Reversal to decline",
    "Decline followed by stabilisation",
    "No confirmed direction",
    "Improvement followed by stabilisation",
    "Recovery/reversal to improvement",
    "Recent-onset improvement",
    "Persistent improvement",
]

PERFORMANCE_DIRECTION_ORDER = [
    "Worsening",
    "Unchanged",
    "Improving",
]

JOINT_GROUP_ORDER = [
    "Compounding concern",
    "Persistent low condition",
    "Recovering but still low",
    "Emerging concern",
    "Stable intermediate condition",
    "Improving condition",
    "High condition with historical loss",
    "Mixed evidence",
]


# Fixed semantic palettes used across every figure.
# Decline/worsening = vermillion; neutral/unchanged = grey;
# improvement = bluish green.
TREND_CLASS_COLOURS = {
    "Confirmed decline": "#D55E00",
    "Neutral/unconfirmed": "#B8B8B8",
    "Confirmed improvement": "#009E73",
}

TREND_TRAJECTORY_COLOURS = {
    "Persistent decline": "#8C2D04",
    "Recent-onset decline": "#D55E00",
    "Reversal to decline": "#E69F00",
    "Decline followed by stabilisation": "#F2B6A0",
    "No confirmed direction": "#B8B8B8",
    "Improvement followed by stabilisation": "#A9D8D0",
    "Recovery/reversal to improvement": "#56B4E9",
    "Recent-onset improvement": "#0072B2",
    "Persistent improvement": "#009E73",
}

PERFORMANCE_DIRECTION_COLOURS = {
    "Worsening": "#D55E00",
    "Unchanged": "#B8B8B8",
    "Improving": "#009E73",
}


# =============================================================================
# 2. FIGURE HELPERS
# =============================================================================

def save_and_show_figure(
    fig: plt.Figure,
    output_base: Path,
    c,
) -> None:
    """
    Save a figure in all formats specified in the configuration and optionally
    display it on screen.
    """

    output_base.parent.mkdir(parents=True, exist_ok=True)

    if c.SAVE_FIGURES:
        for extension in c.FIGURE_FORMATS:
            output_path = output_base.with_suffix(f".{extension}")
            fig.savefig(
                output_path,
                dpi=c.FIGURE_DPI,
                bbox_inches="tight",
                facecolor="white",
            )

    if c.SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def labelled_stacked_bar(
    df: pd.DataFrame,
    unit_col: str,
    class_col: str,
    value_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_base: Path,
    c,
    class_order: Optional[Sequence[str]] = None,
    sort_by_class: Optional[str] = None,
    ascending: bool = True,
    max_units: Optional[int] = None,
    class_colours: Optional[dict[str, str]] = None,
) -> None:
    """
    Create a horizontal stacked bar chart with explicit analytical labels.

    Parameters
    ----------
    df
        Long-format table containing reporting units, classes and values.
    unit_col
        Column containing the reporting-unit display name.
    class_col
        Column containing class labels.
    value_col
        Numeric value used for bar widths.
    title
        Full figure title, including reporting period and analytical reference.
    x_label
        Descriptive x-axis label.
    y_label
        Reporting-unit axis label.
    output_base
        Output path without extension.
    class_order
        Desired order of stacked classes and legend entries.
    sort_by_class
        Optional class used to order reporting units.
    ascending
        Sorting direction.
    max_units
        Optional maximum number of reporting units to show.
    class_colours
        Optional fixed mapping from class labels to colours.
    """

    if df.empty:
        return

    required_columns = {unit_col, class_col, value_col}
    missing = required_columns - set(df.columns)

    if missing:
        raise KeyError(
            f"Cannot create figure. Missing columns: {sorted(missing)}"
        )

    plot_df = df[
        [unit_col, class_col, value_col]
    ].dropna().copy()

    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index=unit_col,
        columns=class_col,
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )

    if class_order is not None:
        for class_name in class_order:
            if class_name not in pivot.columns:
                pivot[class_name] = 0.0

        existing_order = [
            class_name
            for class_name in class_order
            if class_name in pivot.columns
        ]

        remaining_columns = [
            col
            for col in pivot.columns
            if col not in existing_order
        ]

        pivot = pivot[existing_order + remaining_columns]

    if sort_by_class and sort_by_class in pivot.columns:
        pivot = pivot.sort_values(
            sort_by_class,
            ascending=ascending,
        )
    else:
        pivot = pivot.sort_index()

    if max_units is not None and len(pivot) > max_units:
        if sort_by_class and sort_by_class in pivot.columns:
            # Retain the units with the largest values in the sorting class.
            pivot = pivot.nlargest(
                max_units,
                sort_by_class,
            ).sort_values(
                sort_by_class,
                ascending=ascending,
            )
        else:
            pivot = pivot.head(max_units)

    figure_height = max(
        6.0,
        min(18.0, 0.30 * len(pivot) + 2.5),
    )

    fig, ax = plt.subplots(
        figsize=(13, figure_height),
    )

    plot_colours = None
    if class_colours is not None:
        plot_colours = [
            class_colours.get(column, "#7F7F7F")
            for column in pivot.columns
        ]

    pivot.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        width=0.82,
        color=plot_colours,
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        x_label,
        fontsize=10,
    )

    ax.set_ylabel(
        y_label,
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )

    ax.legend(
        title="",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    ax.grid(
        axis="x",
        linewidth=0.4,
        alpha=0.4,
    )

    fig.tight_layout()

    save_and_show_figure(
        fig,
        output_base,
        c,
    )


# =============================================================================
# 3. PREPARE REUSABLE REPORTING ZONES
# =============================================================================

def prepare_zones(c, logger) -> None:
    """
    Prepare reusable country, ecoregion, country × LULC and
    ecoregion × LULC zone rasters.
    """

    template = c.STATE_RASTERS[
        c.PRIMARY_STATE_FORMULATION
    ]

    country_gdf, country_lookup = prepare_vector(
        c.COUNTRY_VECTOR,
        c.COUNTRY_LAYER,
        c.COUNTRY_ID_FIELD,
        c.COUNTRY_NAME_FIELD,
        "country_seq",
        [
            "ISO_A2",
            "WB_A3",
            "WB_REGION",
            "WB_STATUS",
            "SOVEREIGN",
        ],
    )

    ecoregion_gdf, ecoregion_lookup = prepare_vector(
        c.ECOREGION_VECTOR,
        c.ECOREGION_LAYER,
        c.ECOREGION_ID_FIELD,
        c.ECOREGION_NAME_FIELD,
        "ecoregion_seq",
        [
            c.BIOME_ID_FIELD,
            c.BIOME_NAME_FIELD,
            c.REALM_FIELD,
        ],
    )

    country_lookup.to_csv(
        c.REUSABLE_ZONE_DIR / "country_lookup.csv",
        index=False,
    )

    ecoregion_lookup.to_csv(
        c.REUSABLE_ZONE_DIR / "ecoregion_lookup.csv",
        index=False,
    )

    country_raster = (
        c.REUSABLE_ZONE_DIR
        / "country_id_EA250m.tif"
    )

    ecoregion_raster = (
        c.REUSABLE_ZONE_DIR
        / "ecoregion_id_EA250m.tif"
    )

    if (
        c.REBUILD_ZONE_RASTERS
        or not country_raster.exists()
    ):
        logger.info("Rasterising countries")

        rasterize_gdf(
            country_gdf,
            "country_seq",
            template,
            country_raster,
        )

    if (
        c.REBUILD_ZONE_RASTERS
        or not ecoregion_raster.exists()
    ):
        logger.info("Rasterising ecoregions")

        rasterize_gdf(
            ecoregion_gdf,
            "ecoregion_seq",
            template,
            ecoregion_raster,
        )

    if c.RUN_COUNTRY_LULC:
        logger.info(
            "Creating country × LULC 2022 zones"
        )

        create_combined_zone(
            country_raster,
            c.LULC_RASTERS[2022],
            country_lookup,
            "country_seq",
            c.COUNTRY_ID_FIELD,
            c.COUNTRY_NAME_FIELD,
            c.LULC_GROUP_LABELS,
            (
                c.REUSABLE_ZONE_DIR
                / "country_lulc_2022_zone_id.tif"
            ),
            (
                c.REUSABLE_ZONE_DIR
                / "country_lulc_lookup.csv"
            ),
        )

    if c.RUN_ECOREGION_LULC:
        logger.info(
            "Creating ecoregion × LULC 2022 zones"
        )

        create_combined_zone(
            ecoregion_raster,
            c.LULC_RASTERS[2022],
            ecoregion_lookup,
            "ecoregion_seq",
            c.ECOREGION_ID_FIELD,
            c.ECOREGION_NAME_FIELD,
            c.LULC_GROUP_LABELS,
            (
                c.REUSABLE_ZONE_DIR
                / "ecoregion_lulc_2022_zone_id.tif"
            ),
            (
                c.REUSABLE_ZONE_DIR
                / "ecoregion_lulc_lookup.csv"
            ),
        )

    logger.info("Reusable zone preparation complete")


# =============================================================================
# 4. REPORTING-UNIT DEFINITIONS
# =============================================================================

def zone_defs(c) -> list[tuple]:
    """
    Return reporting-unit definitions used across all metric modules.
    """

    zones = []

    if c.RUN_COUNTRY:
        zones.append(
            (
                "country",
                (
                    c.REUSABLE_ZONE_DIR
                    / "country_id_EA250m.tif"
                ),
                pd.read_csv(
                    c.REUSABLE_ZONE_DIR
                    / "country_lookup.csv"
                ),
                "country_seq",
                c.COUNTRY_NAME_FIELD,
            )
        )

    if c.RUN_ECOREGION:
        zones.append(
            (
                "ecoregion",
                (
                    c.REUSABLE_ZONE_DIR
                    / "ecoregion_id_EA250m.tif"
                ),
                pd.read_csv(
                    c.REUSABLE_ZONE_DIR
                    / "ecoregion_lookup.csv"
                ),
                "ecoregion_seq",
                c.ECOREGION_NAME_FIELD,
            )
        )

    if c.RUN_LULC:
        zones.append(
            (
                "lulc",
                c.LULC_RASTERS[2022],
                pd.DataFrame(
                    {
                        "zone_id": list(
                            c.LULC_GROUP_LABELS
                        ),
                        "lulc_name": list(
                            c.LULC_GROUP_LABELS.values()
                        ),
                    }
                ),
                "zone_id",
                "lulc_name",
            )
        )

    if c.RUN_COUNTRY_LULC:
        zones.append(
            (
                "country_lulc",
                (
                    c.REUSABLE_ZONE_DIR
                    / "country_lulc_2022_zone_id.tif"
                ),
                pd.read_csv(
                    c.REUSABLE_ZONE_DIR
                    / "country_lulc_lookup.csv"
                ),
                "zone_id",
                c.COUNTRY_NAME_FIELD,
            )
        )

    if c.RUN_ECOREGION_LULC:
        zones.append(
            (
                "ecoregion_lulc",
                (
                    c.REUSABLE_ZONE_DIR
                    / "ecoregion_lulc_2022_zone_id.tif"
                ),
                pd.read_csv(
                    c.REUSABLE_ZONE_DIR
                    / "ecoregion_lulc_lookup.csv"
                ),
                "zone_id",
                c.ECOREGION_NAME_FIELD,
            )
        )

    return zones


# =============================================================================
# 5. STATE MODULE
# =============================================================================

def run_state(c, logger) -> None:
    """
    Aggregate and compare four benchmark-relative State formulations.

    The primary figures represent 2022 annual mean NDVI under the State
    formulation selected by c.PRIMARY_STATE_FORMULATION.
    """

    logger.info("Starting State module")

    for (
        name,
        zone_raster,
        lookup,
        zone_id,
        label,
    ) in zone_defs(c):

        logger.info(
            "Processing State for %s",
            name,
        )

        all_summaries: list[pd.DataFrame] = []

        # ---------------------------------------------------------------------
        # State composition under each formulation
        # ---------------------------------------------------------------------
        for formulation, state_path in (
            c.STATE_RASTERS.items()
        ):
            summary = categorical_summary(
                state_path,
                zone_raster,
                c.STATE_CLASS_LOOKUP,
                lookup,
                zone_id,
                (
                    c.STATE_OUTPUT_DIR
                    / (
                        f"state_summary_{name}_"
                        f"{formulation}.csv"
                    )
                ),
                c,
                set(c.STATE_CLASS_LOOKUP),
            )

            summary["formulation"] = formulation

            all_summaries.append(summary)

        combined_summary = pd.concat(
            all_summaries,
            ignore_index=True,
        )

        combined_summary.to_csv(
            (
                c.STATE_OUTPUT_DIR
                / f"state_summary_{name}_all.csv"
            ),
            index=False,
        )

        # ---------------------------------------------------------------------
        # Pairwise State sensitivity
        # ---------------------------------------------------------------------
        for formulation_a, formulation_b in (
            combinations(c.STATE_RASTERS, 2)
        ):
            logger.info(
                "State comparison for %s: %s versus %s",
                name,
                formulation_a,
                formulation_b,
            )

            cross = crosstab(
                c.STATE_RASTERS[formulation_a],
                c.STATE_RASTERS[formulation_b],
                zone_raster,
                c.STATE_CLASS_LOOKUP,
                c.STATE_CLASS_LOOKUP,
                lookup,
                zone_id,
                (
                    c.STATE_OUTPUT_DIR
                    / (
                        f"state_confusion_{name}_"
                        f"{formulation_a}_vs_"
                        f"{formulation_b}.csv"
                    )
                ),
                c,
            )

            metrics = agreement_metrics(
                cross,
                zone_id,
                [1, 2, 3],
                c,
            ).merge(
                lookup,
                on=zone_id,
                how="left",
            )

            metrics["formulation_a"] = (
                formulation_a
            )

            metrics["formulation_b"] = (
                formulation_b
            )

            metrics.to_csv(
                (
                    c.STATE_OUTPUT_DIR
                    / (
                        f"state_metrics_{name}_"
                        f"{formulation_a}_vs_"
                        f"{formulation_b}.csv"
                    )
                ),
                index=False,
            )

        # ---------------------------------------------------------------------
        # Primary benchmark-relative State figure
        # ---------------------------------------------------------------------
        if (
            c.CREATE_FIGURES
            and name in {"country", "ecoregion"}
        ):
            primary_candidates = [
                table
                for table in all_summaries
                if (
                    not table.empty
                    and table["formulation"].iloc[0]
                    == c.PRIMARY_STATE_FORMULATION
                )
            ]

            if not primary_candidates:
                logger.warning(
                    "No primary State summary found for %s",
                    name,
                )
                continue

            primary = primary_candidates[0]

            formulation_text = (
                STATE_FORMULATION_TITLES[
                    c.PRIMARY_STATE_FORMULATION
                ]
            )

            title = (
                "Benchmark-relative productivity State "
                f"by {name} in 2022\n"
                f"{formulation_text}"
            )

            y_label = (
                "Country"
                if name == "country"
                else "Ecoregion"
            )

            # Show all countries, but restrict the ecoregion figure
            # to the configured maximum to preserve readability.
            max_units = (
                None
                if name == "country"
                else c.TOP_N_UNITS_IN_FIGURES
            )

            labelled_stacked_bar(
                df=primary,
                unit_col=label,
                class_col="class_name",
                value_col="share_pct",
                title=title,
                x_label=(
                    "Share of valid vegetated area "
                    "classified in 2022 (%)"
                ),
                y_label=y_label,
                output_base=(
                    c.FIGURE_OUTPUT_DIR
                    / (
                        f"state_{name}_composition_"
                        f"{c.PRIMARY_STATE_FORMULATION}"
                    )
                ),
                c=c,
                class_order=STATE_CLASS_ORDER,
                sort_by_class="Poor",
                ascending=True,
                max_units=max_units,
            )

    logger.info("State module complete")


# =============================================================================
# 6. TREND MODULE
# =============================================================================

TREND_WINDOW_METADATA = {
    "earlier_2001_2011": {
        "window_label": "Earlier window (2001–2011)",
        "window_start": 2001,
        "window_end": 2011,
    },
    "recent_2012_2022": {
        "window_label": "Recent window (2012–2022)",
        "window_start": 2012,
        "window_end": 2022,
    },
    "full_2001_2022": {
        "window_label": "Full period (2001-2022)",
        "window_start": 2001,
        "window_end": 2022,
    },
}

TREND_CONFIRMATION_LABELS = {
    "potential": "Potential_negative",
    "ols": "OLS_P_negative",
    "mann_kendall": "MK_negative",
    "newey_west": "NW_negative",
    "strict_joint": "ALL_STAT_negative",
}

TREND_CONFIRMATION_ORDER = [
    "potential",
    "ols",
    "mann_kendall",
    "newey_west",
    "strict_joint",
]


def reporting_unit_metadata(
    name: str,
    zone_id: str,
    label: str,
) -> dict[str, str]:
    """Return explicit metadata explaining each reporting-zone identifier."""

    descriptions = {
        "country": (
            "Sequential country-zone identifier. Join to country_lookup.csv "
            "using the reported ID field to recover country attributes."
        ),
        "ecoregion": (
            "Sequential ecoregion-zone identifier. Join to "
            "ecoregion_lookup.csv using the reported ID field."
        ),
        "lulc": (
            "2022 grouped land-cover class code; values map directly to "
            "LULC_GROUP_LABELS."
        ),
        "country_lulc": (
            "Sequential identifier for a unique country × 2022 land-cover "
            "reporting zone. Join to country_lulc_lookup.csv using zone_id."
        ),
        "ecoregion_lulc": (
            "Sequential identifier for a unique ecoregion × 2022 land-cover "
            "reporting zone. Join to ecoregion_lulc_lookup.csv using zone_id."
        ),
    }

    return {
        "reporting_unit_type": name,
        "reporting_unit_id_field": zone_id,
        "reporting_unit_label_field": label,
        "zone_id_description": descriptions.get(
            name,
            "Sequential identifier for the reporting zone; join to the "
            "corresponding lookup table using the reported ID field.",
        ),
    }


def add_reporting_metadata(
    table: pd.DataFrame,
    name: str,
    zone_id: str,
    label: str,
) -> pd.DataFrame:
    """Attach human-readable reporting-unit metadata to an output table."""

    result = table.copy()
    metadata = reporting_unit_metadata(name, zone_id, label)

    for column, value in metadata.items():
        result[column] = value

    return result


def add_trend_window_metadata(
    table: pd.DataFrame,
    window_key: str,
) -> pd.DataFrame:
    """Attach explicit period fields to every Trend output row."""

    if window_key not in TREND_WINDOW_METADATA:
        raise KeyError(
            f"Unknown Trend window key: {window_key}. "
            f"Expected one of {list(TREND_WINDOW_METADATA)}"
        )

    result = table.copy()
    metadata = TREND_WINDOW_METADATA[window_key]

    result["window_key"] = window_key
    result["window"] = (
        f"{metadata['window_start']}-{metadata['window_end']}"
    )
    result["period"] = result["window"]
    result["window_label"] = metadata["window_label"]
    result["window_start"] = metadata["window_start"]
    result["window_end"] = metadata["window_end"]
    result["window_years_inclusive"] = (
        metadata["window_end"] - metadata["window_start"] + 1
    )
    result["trend_definition"] = (
        "Final three-class productivity Trend: confirmed decline, "
        "neutral/unconfirmed, or confirmed improvement."
    )
    result["share_denominator"] = (
        "Valid vegetated pixels within the reporting unit and Trend window."
    )

    return result


def _normalise_confirmation_key(value: str) -> str:
    text = (
        str(value)
        .strip()
        .lower()
        .replace("–", "_")
        .replace("—", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "potential": "potential",
        "ols": "ols",
        "ols_p": "ols",
        "mann_kendall": "mann_kendall",
        "mannkendall": "mann_kendall",
        "mk": "mann_kendall",
        "newey_west": "newey_west",
        "neweywest": "newey_west",
        "nw": "newey_west",
        "strict": "strict_joint",
        "strict_joint": "strict_joint",
        "joint": "strict_joint",
    }

    if text not in aliases:
        raise KeyError(
            f"Unrecognised Trend confirmation level: {value}"
        )

    return aliases[text]


def _window_key_from_text(value: str) -> Optional[str]:
    text = (
        str(value)
        .lower()
        .replace("–", "_")
        .replace("—", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    for window_key in TREND_WINDOW_METADATA:
        if window_key in text:
            return window_key

    if "2001_2011" in text:
        return "earlier_2001_2011"
    if "2012_2022" in text:
        return "recent_2012_2022"
    if "2001_2022" in text:
        return "full_2001_2022"

    return None


def trend_funnel_rasters_by_window(c) -> dict[str, dict[str, Path]]:
    """
    Resolve Trend funnel rasters into window → confirmation-level mappings.

    Supported configuration forms
    -----------------------------
    1. Preferred nested mapping::

           TREND_FUNNEL_RASTERS = {
               "earlier_2001_2011": {"potential": Path(...), ...},
               "recent_2012_2022": {"potential": Path(...), ...},
               "full_2001_2022": {"potential": Path(...), ...},
           }

    2. Flat keys containing both window and level::

           "earlier_2001_2011_potential": Path(...)

    3. Legacy flat level → path mapping. This is retained and explicitly
       assigned to the full 2001–2022 period.
    """

    source = c.TREND_FUNNEL_RASTERS

    if not source:
        return {}

    first_value = next(iter(source.values()))

    # Preferred nested structure.
    if isinstance(first_value, dict):
        resolved: dict[str, dict[str, Path]] = {}

        for window_key, stage_mapping in source.items():
            resolved_window = _window_key_from_text(window_key)

            if resolved_window is None:
                raise KeyError(
                    f"Cannot infer Trend window from key: {window_key}"
                )

            resolved[resolved_window] = {
                _normalise_confirmation_key(stage): Path(path)
                for stage, path in stage_mapping.items()
            }

        return resolved

    # Flat mapping with window embedded in each key.
    embedded: dict[str, dict[str, Path]] = {}
    unresolved = []

    for raw_key, path in source.items():
        window_key = _window_key_from_text(raw_key)

        if window_key is None:
            unresolved.append((raw_key, path))
            continue

        stage_text = str(raw_key)
        for window_token in [
            window_key,
            "2001_2011",
            "2012_2022",
            "2001_2022",
            "earlier",
            "recent",
            "full",
        ]:
            stage_text = stage_text.replace(window_token, "")

        stage_text = stage_text.strip("_ -")
        stage = _normalise_confirmation_key(stage_text)

        embedded.setdefault(window_key, {})[stage] = Path(path)

    if embedded and not unresolved:
        return embedded

    # Existing flat stage-only configuration.
    #
    # The configured source paths and filenames explicitly identify these
    # products as full-period 2001-2022 rasters. Preserve that valid existing
    # configuration and assign it only to the full-period funnel.
    if unresolved and not embedded:
        recognised_stages = set(TREND_CONFIRMATION_ORDER)

        normalised = {
            _normalise_confirmation_key(raw_key): Path(path)
            for raw_key, path in unresolved
        }

        unexpected = sorted(
            set(normalised) - recognised_stages
        )

        if unexpected:
            raise ValueError(
                "TREND_FUNNEL_RASTERS contains unrecognised statistical "
                f"levels: {unexpected}. Expected: "
                f"{list(TREND_CONFIRMATION_ORDER)}"
            )

        return {
            "full_2001_2022": normalised
        }

    if embedded and unresolved:
        unresolved_keys = [
            str(key)
            for key, _ in unresolved
        ]

        raise ValueError(
            "TREND_FUNNEL_RASTERS mixes window-specific and window-ambiguous "
            f"keys. Unresolved keys: {unresolved_keys}"
        )

    raise ValueError(
        "TREND_FUNNEL_RASTERS could not be interpreted."
    )



def validate_trend_funnel_windows(
    funnel_rasters: dict[str, dict[str, Path]],
) -> None:
    """
    Validate the Trend funnel windows and statistics actually supplied.

    A three-window funnel is not required when the configured OLS-P,
    Mann-Kendall, Newey-West and joint-statistic rasters exist only for the
    full 2001-2022 period.
    """

    allowed_windows = set(TREND_WINDOW_METADATA)
    observed_windows = set(funnel_rasters)

    if not observed_windows:
        raise ValueError(
            "No Trend funnel rasters were supplied."
        )

    unexpected_windows = sorted(
        observed_windows - allowed_windows
    )

    if unexpected_windows:
        raise ValueError(
            "Unexpected Trend funnel windows: "
            f"{unexpected_windows}. Allowed windows are "
            f"{sorted(allowed_windows)}."
        )

    required_stages = set(TREND_CONFIRMATION_ORDER)
    problems = []

    for window_key, stage_mapping in funnel_rasters.items():
        observed_stages = set(stage_mapping)

        missing_stages = sorted(
            required_stages - observed_stages
        )

        unexpected_stages = sorted(
            observed_stages - required_stages
        )

        if missing_stages:
            problems.append(
                f"{window_key}: missing statistical levels "
                f"{missing_stages}"
            )

        if unexpected_stages:
            problems.append(
                f"{window_key}: unexpected statistical levels "
                f"{unexpected_stages}"
            )

        for stage, raster_path in stage_mapping.items():
            raster_path = Path(raster_path)

            if not raster_path.exists():
                problems.append(
                    f"{window_key}/{stage}: file not found: "
                    f"{raster_path}"
                )

    if problems:
        raise FileNotFoundError(
            "Trend funnel inputs are incomplete:\n"
            + "\n".join(problems)
        )

def clean_trend_funnel_table(
    table: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    """
    Return a compact, analysis-ready funnel table.
    """

    result = table.copy()

    result["applied_statistic"] = result[
        "confirmation_level"
    ].map(TREND_CONFIRMATION_LABELS)

    result["class_name"] = result["applied_statistic"]
    result["trend_direction"] = "Negative"
    result["trend_direction_code"] = 1

    result["negative_share_of_valid_area_pct"] = np.where(
        result["valid_pixel_count"] > 0,
        100.0
        * result["pixel_count"]
        / result["valid_pixel_count"],
        np.nan,
    )

    result["negative_retention_from_potential_pct"] = result[
        "retention_pct"
    ]

    preferred = [
        zone_id,
        "country_seq",
        "ISO_A3",
        "NAM_0",
        "ecoregion_seq",
        "ECO_NAME",
        "lulc_code",
        "lulc_name",
        "window_key",
        "window",
        "period",
        "window_start",
        "window_end",
        "confirmation_level",
        "applied_statistic",
        "trend_direction_code",
        "trend_direction",
        "pixel_count",
        "area_ha",
        "area_mha",
        "valid_pixel_count",
        "valid_area_ha",
        "negative_share_of_valid_area_pct",
        "retention_denominator_pixel_count",
        "retention_denominator_area_ha",
        "negative_retention_from_potential_pct",
        "quality_class",
        "reporting_unit_type",
    ]

    columns = [
        column
        for column in preferred
        if column in result.columns
    ]

    return result[columns].copy()


def build_trend_funnel_plot_ready(
    combined_funnel: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    """
    Create one row per reporting unit, window and statistical filter.
    """

    clean = clean_trend_funnel_table(
        combined_funnel,
        zone_id,
    )

    confirmation_rank = {
        level: position
        for position, level in enumerate(
            TREND_CONFIRMATION_ORDER,
            start=1,
        )
    }

    clean["statistic_order"] = clean[
        "confirmation_level"
    ].map(confirmation_rank)

    sort_columns = [
        column
        for column in [
            "NAM_0",
            "lulc_code",
            "window_start",
            "statistic_order",
        ]
        if column in clean.columns
    ]

    if sort_columns:
        clean = clean.sort_values(
            sort_columns,
            kind="stable",
        )

    return clean


def build_trend_funnel_wide(
    combined_funnel: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    """Create one comprehensive row per reporting unit and Trend window."""

    index_columns = [
        zone_id,
        "reporting_unit_type",
        "reporting_unit_id_field",
        "reporting_unit_label_field",
        "zone_id_description",
        "window_key",
        "window",
        "period",
        "window_label",
        "window_start",
        "window_end",
        "window_years_inclusive",
    ]

    lookup_columns = [
        column
        for column in combined_funnel.columns
        if column not in {
            "class_code",
            "class_name",
            "pixel_count",
            "area_ha",
            "area_mha",
            "share_pct",
            "valid_pixel_count",
            "valid_area_ha",
            "quality_class",
            "confirmation_level",
            "confirmation_label",
            "retention_pct",
            "retention_denominator_pixel_count",
            "retention_denominator_area_ha",
            "retention_denominator_definition",
            "confirmation_definition",
            "trend_direction_scope",
            "trend_definition",
            "share_denominator",
        }
        and column not in index_columns
    ]

    # Retain descriptive lookup columns without allowing duplicated names.
    for column in lookup_columns:
        if column not in index_columns:
            index_columns.append(column)

    pixel_wide = combined_funnel.pivot_table(
        index=index_columns,
        columns="confirmation_level",
        values="pixel_count",
        aggfunc="sum",
        fill_value=0,
    )

    pixel_wide.columns = [
        f"{column}_pixel_count"
        for column in pixel_wide.columns
    ]

    area_wide = combined_funnel.pivot_table(
        index=index_columns,
        columns="confirmation_level",
        values="area_ha",
        aggfunc="sum",
        fill_value=0,
    )

    area_wide.columns = [
        f"{column}_area_ha"
        for column in area_wide.columns
    ]

    retention_wide = combined_funnel.pivot_table(
        index=index_columns,
        columns="confirmation_level",
        values="retention_pct",
        aggfunc="first",
    )

    retention_wide.columns = [
        f"{column}_retention_pct"
        for column in retention_wide.columns
    ]

    wide = (
        pixel_wide.join(area_wide, how="outer")
        .join(retention_wide, how="outer")
        .reset_index()
    )

    for stage in TREND_CONFIRMATION_ORDER:
        for suffix in [
            "pixel_count",
            "area_ha",
            "retention_pct",
        ]:
            column = f"{stage}_{suffix}"
            if column not in wide.columns:
                wide[column] = np.nan

    wide["strict_to_potential_ratio_pct"] = (
        wide["strict_joint_retention_pct"]
    )
    wide["ols_to_potential_ratio_pct"] = wide["ols_retention_pct"]
    wide["mann_kendall_to_potential_ratio_pct"] = (
        wide["mann_kendall_retention_pct"]
    )
    wide["newey_west_to_potential_ratio_pct"] = (
        wide["newey_west_retention_pct"]
    )
    wide["funnel_definition"] = (
        "Declining-area confirmation funnel. Potential is the denominator; "
        "OLS, Mann–Kendall and Newey–West are individual confirmation "
        "levels; strict joint is the final concordant criterion."
    )

    return wide


def summarise_trend_funnel_statistics(
    combined_funnel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise retention distributions by reporting unit, window and level."""

    return (
        combined_funnel.groupby(
            [
                "reporting_unit_type",
                "window_key",
                "window_label",
                "window_start",
                "window_end",
                "confirmation_level",
                "confirmation_label",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            reporting_units=("retention_pct", "size"),
            reporting_units_with_ratio=("retention_pct", "count"),
            total_confirmed_pixel_count=("pixel_count", "sum"),
            total_confirmed_area_ha=("area_ha", "sum"),
            mean_retention_pct=("retention_pct", "mean"),
            median_retention_pct=("retention_pct", "median"),
            minimum_retention_pct=("retention_pct", "min"),
            maximum_retention_pct=("retention_pct", "max"),
            lower_quartile_retention_pct=(
                "retention_pct",
                lambda x: x.quantile(0.25),
            ),
            upper_quartile_retention_pct=(
                "retention_pct",
                lambda x: x.quantile(0.75),
            ),
        )
    )


def run_trend(c, logger) -> None:
    """
    Aggregate strict Trend directions, multi-window trajectories and complete
    statistical-confirmation funnels with explicit periods and zone metadata.
    """

    logger.info("Starting Trend module")

    all_window_outputs: list[pd.DataFrame] = []
    all_trajectory_outputs: list[pd.DataFrame] = []
    all_funnel_outputs: list[pd.DataFrame] = []
    all_funnel_wide_outputs: list[pd.DataFrame] = []

    funnel_rasters = trend_funnel_rasters_by_window(c)
    validate_trend_funnel_windows(funnel_rasters)

    for (
        name,
        zone_raster,
        lookup,
        zone_id,
        label,
    ) in zone_defs(c):

        logger.info("Processing Trend for %s", name)

        trend_summaries: list[pd.DataFrame] = []

        # ---------------------------------------------------------------------
        # Strict Trend direction in each explicitly named window
        # ---------------------------------------------------------------------
        for window_key, trend_path in c.TREND_DIRECTION_RASTERS.items():
            summary = categorical_summary(
                trend_path,
                zone_raster,
                c.TREND_CLASS_LOOKUP,
                lookup,
                zone_id,
                (
                    c.TREND_OUTPUT_DIR
                    / f"trend_{name}_{window_key}.csv"
                ),
                c,
                set(c.TREND_CLASS_LOOKUP),
            )

            summary = add_reporting_metadata(
                summary,
                name,
                zone_id,
                label,
            )
            summary = add_trend_window_metadata(
                summary,
                window_key,
            )

            # Rewrite the individual file after adding interpretable metadata.
            summary.to_csv(
                c.TREND_OUTPUT_DIR
                / f"trend_{name}_{window_key}.csv",
                index=False,
            )

            trend_summaries.append(summary)
            all_window_outputs.append(summary)

        if trend_summaries:
            combined_windows = pd.concat(
                trend_summaries,
                ignore_index=True,
            )

            combined_windows.to_csv(
                c.TREND_OUTPUT_DIR
                / f"trend_{name}_all_windows.csv",
                index=False,
            )

        # ---------------------------------------------------------------------
        # Earlier × recent trajectory matrix
        # ---------------------------------------------------------------------
        trajectories = crosstab(
            c.TREND_DIRECTION_RASTERS["earlier_2001_2011"],
            c.TREND_DIRECTION_RASTERS["recent_2012_2022"],
            zone_raster,
            c.TREND_CLASS_LOOKUP,
            c.TREND_CLASS_LOOKUP,
            lookup,
            zone_id,
            c.TREND_OUTPUT_DIR
            / f"trend_trajectory_matrix_{name}.csv",
            c,
        )

        trajectories["trajectory_code"] = (
            trajectories["class_a_code"] * 10
            + trajectories["class_b_code"]
        )
        trajectories["trajectory_name"] = (
            trajectories["trajectory_code"].map(TRAJECTORY)
        )
        trajectories["earlier_window_key"] = "earlier_2001_2011"
        trajectories["earlier_window_label"] = (
            TREND_WINDOW_METADATA["earlier_2001_2011"]["window_label"]
        )
        trajectories["recent_window_key"] = "recent_2012_2022"
        trajectories["recent_window_label"] = (
            TREND_WINDOW_METADATA["recent_2012_2022"]["window_label"]
        )
        trajectories["trajectory_definition"] = (
            "Pixel-level transition between final strict Trend classes in "
            "2001–2011 and 2012–2022."
        )
        trajectories["share_denominator"] = (
            "Pixels valid in both the earlier and recent Trend windows "
            "within the reporting unit."
        )
        trajectories = add_reporting_metadata(
            trajectories,
            name,
            zone_id,
            label,
        )

        trajectories.to_csv(
            c.TREND_OUTPUT_DIR
            / f"trend_trajectories_{name}_2001_2011_to_2012_2022.csv",
            index=False,
        )

        # Retain the legacy filename for compatibility, but with full metadata.
        trajectories.to_csv(
            c.TREND_OUTPUT_DIR
            / f"trend_trajectories_{name}.csv",
            index=False,
        )

        all_trajectory_outputs.append(trajectories)

        if c.CREATE_FIGURES and name in {"country", "ecoregion"}:
            max_units = (
                None
                if name == "country"
                else c.TOP_N_UNITS_IN_FIGURES
            )

            labelled_stacked_bar(
                df=trajectories,
                unit_col=label,
                class_col="trajectory_name",
                value_col="share_pct",
                title=(
                    "Strict earlier-to-recent productivity Trend "
                    f"trajectories by {name}\n"
                    "Earlier window: 2001–2011; recent window: 2012–2022"
                ),
                x_label=(
                    "Share of pixels valid in both Trend windows (%)"
                ),
                y_label=(
                    "Country" if name == "country" else "Ecoregion"
                ),
                output_base=(
                    c.FIGURE_OUTPUT_DIR
                    / f"trend_{name}_trajectories_2001_2011_to_2012_2022"
                ),
                c=c,
                class_order=TREND_TRAJECTORY_ORDER,
                sort_by_class="Persistent decline",
                ascending=True,
                max_units=max_units,
                class_colours=TREND_TRAJECTORY_COLOURS,
            )

        # ---------------------------------------------------------------------
        # Statistical confirmation funnel for every configured Trend window
        # ---------------------------------------------------------------------
        zone_funnel_outputs: list[pd.DataFrame] = []

        for window_key, stage_paths in funnel_rasters.items():
            missing_stages = [
                stage
                for stage in TREND_CONFIRMATION_ORDER
                if stage not in stage_paths
            ]

            if missing_stages:
                logger.warning(
                    "Trend funnel for %s, %s is missing stages: %s",
                    name,
                    window_key,
                    ", ".join(missing_stages),
                )

            window_funnel_parts: list[pd.DataFrame] = []

            for confirmation_level in TREND_CONFIRMATION_ORDER:
                if confirmation_level not in stage_paths:
                    continue

                trend_path = stage_paths[confirmation_level]

                funnel = categorical_summary(
                    trend_path,
                    zone_raster,
                    {1: "Declining"},
                    lookup,
                    zone_id,
                    (
                        c.TREND_OUTPUT_DIR
                        / (
                            f"trend_funnel_{name}_{window_key}_"
                            f"{confirmation_level}.csv"
                        )
                    ),
                    c,
                    {1},
                )

                funnel = add_reporting_metadata(
                    funnel,
                    name,
                    zone_id,
                    label,
                )
                funnel = add_trend_window_metadata(
                    funnel,
                    window_key,
                )
                funnel["confirmation_level"] = confirmation_level
                funnel["confirmation_label"] = (
                    TREND_CONFIRMATION_LABELS[confirmation_level]
                )
                funnel["applied_statistic"] = (
                    TREND_CONFIRMATION_LABELS[confirmation_level]
                )
                funnel["class_name"] = funnel["applied_statistic"]
                funnel["trend_direction"] = "Negative"
                funnel["trend_direction_code"] = 1
                funnel["trend_direction_scope"] = (
                    "Negative Trend pixels only"
                )
                funnel["confirmation_definition"] = (
                    "Area classified as declining at the stated statistical "
                    "confirmation level."
                )

                funnel.to_csv(
                    c.TREND_OUTPUT_DIR
                    / (
                        f"trend_funnel_{name}_{window_key}_"
                        f"{confirmation_level}.csv"
                    ),
                    index=False,
                )

                window_funnel_parts.append(funnel)

            if not window_funnel_parts:
                continue

            window_funnel = pd.concat(
                window_funnel_parts,
                ignore_index=True,
            )

            potential = window_funnel.loc[
                window_funnel["confirmation_level"] == "potential",
                [zone_id, "pixel_count", "area_ha"],
            ].rename(
                columns={
                    "pixel_count": "retention_denominator_pixel_count",
                    "area_ha": "retention_denominator_area_ha",
                }
            )

            window_funnel = window_funnel.merge(
                potential,
                on=zone_id,
                how="left",
            )

            denominator_ok = (
                window_funnel["retention_denominator_pixel_count"]
                >= c.MIN_TREND_FUNNEL_DENOMINATOR_PIXELS
            )

            window_funnel["retention_pct"] = np.where(
                denominator_ok,
                100.0
                * window_funnel["pixel_count"]
                / window_funnel["retention_denominator_pixel_count"],
                np.nan,
            )
            window_funnel["retention_denominator_definition"] = (
                "Potential negative-Trend pixels in the same reporting unit "
                "and Trend window. Ratios are suppressed below the configured "
                "minimum denominator."
            )
            window_funnel["negative_share_of_valid_area_pct"] = np.where(
                window_funnel["valid_pixel_count"] > 0,
                100.0
                * window_funnel["pixel_count"]
                / window_funnel["valid_pixel_count"],
                np.nan,
            )
            window_funnel[
                "negative_retention_from_potential_pct"
            ] = window_funnel["retention_pct"]

            for confirmation_level in TREND_CONFIRMATION_ORDER:
                stage_table = window_funnel.loc[
                    window_funnel["confirmation_level"]
                    == confirmation_level
                ].copy()

                if stage_table.empty:
                    continue

                clean_trend_funnel_table(
                    stage_table,
                    zone_id,
                ).to_csv(
                    c.TREND_OUTPUT_DIR
                    / (
                        f"trend_funnel_{name}_{window_key}_"
                        f"{confirmation_level}.csv"
                    ),
                    index=False,
                )

            build_trend_funnel_plot_ready(
                window_funnel,
                zone_id,
            ).to_csv(
                c.TREND_OUTPUT_DIR
                / (
                    f"trend_funnel_{name}_{window_key}_"
                    "all_statistics_long.csv"
                ),
                index=False,
            )

            build_trend_funnel_plot_ready(
                window_funnel,
                zone_id,
            ).to_csv(
                c.TREND_OUTPUT_DIR
                / f"trend_funnel_{name}_{window_key}_all_levels.csv",
                index=False,
            )

            zone_funnel_outputs.append(window_funnel)
            all_funnel_outputs.append(window_funnel)

        if zone_funnel_outputs:
            zone_available_windows = pd.concat(
                zone_funnel_outputs,
                ignore_index=True,
            )

            plot_ready = build_trend_funnel_plot_ready(
                zone_available_windows,
                zone_id,
            )

            available_window_keys = (
                zone_available_windows["window_key"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            available_window_keys = [
                window_key
                for window_key in TREND_WINDOW_METADATA
                if window_key in available_window_keys
            ]

            if not available_window_keys:
                raise ValueError(
                    "No Trend funnel window was found for "
                    f"reporting unit type: {name}"
                )

            if len(available_window_keys) == 1:
                output_scope = available_window_keys[0]
            else:
                output_scope = "all_available_windows"

            plot_ready.to_csv(
                c.TREND_OUTPUT_DIR
                / (
                    f"trend_funnel_{name}_{output_scope}_"
                    "all_statistics_long.csv"
                ),
                index=False,
            )

            wide = build_trend_funnel_wide(
                zone_available_windows,
                zone_id,
            )

            wide.to_csv(
                c.TREND_OUTPUT_DIR
                / (
                    f"trend_funnel_{name}_{output_scope}_"
                    "all_statistics_wide.csv"
                ),
                index=False,
            )

            all_funnel_wide_outputs.append(wide)

    # -------------------------------------------------------------------------
    # Cross-reporting-unit master outputs
    # -------------------------------------------------------------------------
    if all_window_outputs:
        pd.concat(
            all_window_outputs,
            ignore_index=True,
        ).to_csv(
            c.TREND_OUTPUT_DIR
            / "trend_all_reporting_units_all_windows.csv",
            index=False,
        )

    if all_trajectory_outputs:
        pd.concat(
            all_trajectory_outputs,
            ignore_index=True,
        ).to_csv(
            c.TREND_OUTPUT_DIR
            / "trend_trajectories_all_reporting_units.csv",
            index=False,
        )

    if all_funnel_outputs:
        master_funnel = pd.concat(
            all_funnel_outputs,
            ignore_index=True,
        )

        master_available_windows = (
            master_funnel["window_key"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        master_available_windows = [
            window_key
            for window_key in TREND_WINDOW_METADATA
            if window_key in master_available_windows
        ]

        if len(master_available_windows) == 1:
            master_output_scope = master_available_windows[0]
        else:
            master_output_scope = "all_available_windows"

        master_funnel.to_csv(
            c.TREND_OUTPUT_DIR
            / (
                "trend_funnel_all_reporting_units_"
                f"{master_output_scope}_raw.csv"
            ),
            index=False,
        )

        compact_master_parts = []
        for reporting_unit_type, subset in master_funnel.groupby(
            "reporting_unit_type",
            observed=True,
        ):
            zone_candidates = [
                column
                for column in [
                    "zone_id",
                    "country_seq",
                    "ecoregion_seq",
                    "lulc_code",
                ]
                if column in subset.columns
            ]
            selected_zone_id = (
                zone_candidates[0]
                if zone_candidates
                else "reporting_unit_type"
            )
            compact_master_parts.append(
                build_trend_funnel_plot_ready(
                    subset,
                    selected_zone_id,
                )
            )

        pd.concat(
            compact_master_parts,
            ignore_index=True,
            sort=False,
        ).to_csv(
            c.TREND_OUTPUT_DIR
            / (
                "trend_funnel_all_reporting_units_"
                f"{master_output_scope}_all_statistics_long.csv"
            ),
            index=False,
        )

        summarise_trend_funnel_statistics(master_funnel).to_csv(
            c.TREND_OUTPUT_DIR
            / "trend_funnel_statistical_summary_by_window_and_unit.csv",
            index=False,
        )

    if all_funnel_wide_outputs:
        master_wide = pd.concat(
            all_funnel_wide_outputs,
            ignore_index=True,
            sort=False,
        )

        wide_available_windows = (
            master_wide["window_key"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        wide_available_windows = [
            window_key
            for window_key in TREND_WINDOW_METADATA
            if window_key in wide_available_windows
        ]

        if len(wide_available_windows) == 1:
            wide_output_scope = wide_available_windows[0]
        else:
            wide_output_scope = "all_available_windows"

        master_wide.to_csv(
            c.TREND_OUTPUT_DIR
            / (
                "trend_funnel_all_reporting_units_"
                f"{wide_output_scope}_all_statistics_wide.csv"
            ),
            index=False,
        )

    logger.info("Trend module complete")


# =============================================================================
# 7. PERFORMANCE MODULE
# =============================================================================

PERFORMANCE_PERIOD_METADATA = {
    "performance_2011": {
        "reporting_year": 2011,
        "assessment_start": 2009,
        "assessment_end": 2011,
        "period_label": "Earlier assessment period (2009–2011)",
    },
    "performance_2022": {
        "reporting_year": 2022,
        "assessment_start": 2020,
        "assessment_end": 2022,
        "period_label": "Recent assessment period (2020–2022)",
    },
}


def add_performance_period_metadata(
    table: pd.DataFrame,
    period_key: str,
) -> pd.DataFrame:
    """Attach explicit assessment-period and fixed-baseline metadata."""

    result = table.copy()
    metadata = PERFORMANCE_PERIOD_METADATA.get(
        period_key,
        {
            "reporting_year": np.nan,
            "assessment_start": np.nan,
            "assessment_end": np.nan,
            "period_label": period_key,
        },
    )

    result["period_key"] = period_key
    result["period_label"] = metadata["period_label"]
    result["reporting_year"] = metadata["reporting_year"]
    result["assessment_start"] = metadata["assessment_start"]
    result["assessment_end"] = metadata["assessment_end"]
    result["baseline_start"] = 2001
    result["baseline_end"] = 2003
    result["baseline_label"] = "Fixed 2001–2003 same-pixel baseline"
    result["performance_definition"] = (
        "Percentage displacement of three-year mean NDVI during the "
        "assessment period relative to the fixed 2001–2003 same-pixel "
        "baseline, classified into five descriptive Performance classes."
    )
    result["share_denominator"] = (
        "Pixels with a valid Performance class and an included vegetated "
        "land-cover class within the reporting unit and assessment period."
    )

    return result


def _performance_lulc_rasters(c) -> dict[int, Path]:
    """
    Resolve LULC rasters used only by Performance.

    PERFORMANCE_LULC_RASTERS is preferred when configured; LULC_RASTERS is
    retained as a backward-compatible fallback. No State, Trend or joint-STP
    input is modified.
    """

    mapping = getattr(c, "PERFORMANCE_LULC_RASTERS", None)
    if not mapping:
        mapping = c.LULC_RASTERS

    return {
        int(year): Path(path)
        for year, path in mapping.items()
    }


def _included_performance_lulc_labels(c) -> dict[int, str]:
    """Return and validate the LULC classes included in the vegetated domain."""

    included_codes = {
        int(code)
        for code in c.INCLUDED_LULC_CODES
    }

    labels = {
        int(code): str(label)
        for code, label in c.LULC_GROUP_LABELS.items()
        if int(code) in included_codes
    }

    missing = included_codes.difference(labels)
    if missing:
        raise KeyError(
            "INCLUDED_LULC_CODES contains codes missing from "
            f"LULC_GROUP_LABELS: {sorted(missing)}"
        )

    overlap = included_codes.intersection(
        int(code)
        for code in getattr(c, "EXCLUDED_LULC_CODES", [])
    )
    if overlap:
        raise ValueError(
            "The same LULC codes occur in INCLUDED_LULC_CODES and "
            f"EXCLUDED_LULC_CODES: {sorted(overlap)}"
        )

    return labels


def _assert_performance_rasters_aligned(
    reference: rasterio.io.DatasetReader,
    other: rasterio.io.DatasetReader,
    reference_name: str,
    other_name: str,
) -> None:
    """Fail early if two Performance inputs cannot be compared pixelwise."""

    aligned = (
        reference.width == other.width
        and reference.height == other.height
        and reference.crs == other.crs
        and reference.transform.almost_equals(other.transform)
    )

    if not aligned:
        raise ValueError(
            "Performance analysis requires aligned rasters. "
            f"{reference_name} and {other_name} differ in shape, CRS or "
            "transform."
        )


def _performance_masked_zone(
    c,
    logger,
    zone_raster: Path,
    lulc_raster: Path,
    output_name: str,
) -> Path:
    """
    Mask a reporting-zone raster to the included vegetated LULC domain.

    Excluded LULC codes, background and invalid pixels are written as zero.
    The masked zones are stored inside the Performance output directory so no
    reusable zones used by State, Trend or joint STP are altered.
    """

    output_dir = c.PERFORMANCE_OUTPUT_DIR / "_vegetated_zone_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    rebuild = bool(getattr(c, "REBUILD_ZONE_RASTERS", False))
    if output_path.exists() and not rebuild:
        return output_path

    included_codes = np.asarray(
        sorted(int(code) for code in c.INCLUDED_LULC_CODES),
        dtype=np.int32,
    )

    logger.info(
        "Creating Performance vegetated-domain mask: %s",
        output_path.name,
    )

    with (
        rasterio.open(zone_raster) as zone_src,
        rasterio.open(lulc_raster) as lulc_src,
    ):
        _assert_performance_rasters_aligned(
            zone_src,
            lulc_src,
            str(zone_raster),
            str(lulc_raster),
        )

        profile = zone_src.profile.copy()
        profile.update(
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in zone_src.block_windows(1):
                zones = zone_src.read(1, window=window, masked=True)
                lulc = lulc_src.read(1, window=window, masked=True)

                valid = (
                    ~np.ma.getmaskarray(zones)
                    & ~np.ma.getmaskarray(lulc)
                    & (zones.data > 0)
                    & np.isin(lulc.data, included_codes)
                )

                out = np.zeros(zones.shape, dtype=zone_src.dtypes[0])
                out[valid] = zones.data[valid]
                dst.write(out, 1, window=window)

    return output_path


def _performance_lulc_zone(
    c,
    logger,
    land_cover_year: int,
) -> Path:
    """Return a LULC reporting-zone raster containing included classes only."""

    lulc_rasters = _performance_lulc_rasters(c)
    if land_cover_year not in lulc_rasters:
        raise KeyError(
            "Performance requires a broad-LULC raster for "
            f"{land_cover_year}, but none is configured."
        )

    lulc_raster = lulc_rasters[land_cover_year]
    return _performance_masked_zone(
        c,
        logger,
        lulc_raster,
        lulc_raster,
        f"performance_lulc_{land_cover_year}_included_zone_id.tif",
    )


def _performance_combined_zone(
    c,
    logger,
    period_key: str,
    combined_name: str,
) -> tuple[Path, pd.DataFrame, str, str]:
    """
    Build a period-specific country × LULC or ecoregion × LULC zone.

    Only included vegetated LULC codes are supplied to the combined-zone
    builder. Output names are Performance-specific, so existing reusable zones
    remain untouched.
    """

    metadata = PERFORMANCE_PERIOD_METADATA[period_key]
    land_cover_year = int(metadata["assessment_end"])
    masked_lulc = _performance_lulc_zone(c, logger, land_cover_year)
    included_labels = _included_performance_lulc_labels(c)

    if combined_name == "country_lulc":
        base_raster = c.REUSABLE_ZONE_DIR / "country_id_EA250m.tif"
        base_lookup = pd.read_csv(
            c.REUSABLE_ZONE_DIR / "country_lookup.csv"
        )
        base_id = "country_seq"
        source_id = c.COUNTRY_ID_FIELD
        source_label = c.COUNTRY_NAME_FIELD
    elif combined_name == "ecoregion_lulc":
        base_raster = c.REUSABLE_ZONE_DIR / "ecoregion_id_EA250m.tif"
        base_lookup = pd.read_csv(
            c.REUSABLE_ZONE_DIR / "ecoregion_lookup.csv"
        )
        base_id = "ecoregion_seq"
        source_id = c.ECOREGION_ID_FIELD
        source_label = c.ECOREGION_NAME_FIELD
    else:
        raise KeyError(
            f"Unsupported Performance combined zone: {combined_name}"
        )

    output_dir = c.PERFORMANCE_OUTPUT_DIR / "_vegetated_zone_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    zone_raster = (
        output_dir
        / (
            f"performance_{combined_name}_{land_cover_year}_"
            "included_zone_id.tif"
        )
    )
    lookup_path = (
        output_dir
        / (
            f"performance_{combined_name}_{land_cover_year}_"
            "included_lookup.csv"
        )
    )

    rebuild = bool(getattr(c, "REBUILD_ZONE_RASTERS", False))
    if rebuild or not zone_raster.exists() or not lookup_path.exists():
        logger.info(
            "Creating Performance-specific %s zone for %s",
            combined_name,
            land_cover_year,
        )
        create_combined_zone(
            base_raster,
            masked_lulc,
            base_lookup,
            base_id,
            source_id,
            source_label,
            included_labels,
            zone_raster,
            lookup_path,
        )

    return (
        zone_raster,
        pd.read_csv(lookup_path),
        "zone_id",
        source_label,
    )


def performance_period_zone_defs(
    c,
    logger,
    period_key: str,
) -> list[tuple]:
    """
    Return reporting zones masked to the relevant assessment-period LULC.

    The 2009–2011 assessment uses 2011 LULC; the 2020–2022 assessment uses
    2022 LULC. Country and ecoregion boundaries remain fixed, but their valid
    denominators are restricted to included vegetated classes in the relevant
    assessment year.
    """

    if period_key not in PERFORMANCE_PERIOD_METADATA:
        raise KeyError(
            f"Unknown Performance period key: {period_key}"
        )

    land_cover_year = int(
        PERFORMANCE_PERIOD_METADATA[period_key]["assessment_end"]
    )
    lulc_rasters = _performance_lulc_rasters(c)
    if land_cover_year not in lulc_rasters:
        raise KeyError(
            f"No Performance LULC raster is configured for {land_cover_year}."
        )

    assessment_lulc = lulc_rasters[land_cover_year]
    included_labels = _included_performance_lulc_labels(c)
    zones: list[tuple] = []

    if c.RUN_COUNTRY:
        country_zone = _performance_masked_zone(
            c,
            logger,
            c.REUSABLE_ZONE_DIR / "country_id_EA250m.tif",
            assessment_lulc,
            f"performance_country_{land_cover_year}_included_zone_id.tif",
        )
        zones.append(
            (
                "country",
                country_zone,
                pd.read_csv(c.REUSABLE_ZONE_DIR / "country_lookup.csv"),
                "country_seq",
                c.COUNTRY_NAME_FIELD,
                land_cover_year,
            )
        )

    if c.RUN_ECOREGION:
        ecoregion_zone = _performance_masked_zone(
            c,
            logger,
            c.REUSABLE_ZONE_DIR / "ecoregion_id_EA250m.tif",
            assessment_lulc,
            f"performance_ecoregion_{land_cover_year}_included_zone_id.tif",
        )
        zones.append(
            (
                "ecoregion",
                ecoregion_zone,
                pd.read_csv(c.REUSABLE_ZONE_DIR / "ecoregion_lookup.csv"),
                "ecoregion_seq",
                c.ECOREGION_NAME_FIELD,
                land_cover_year,
            )
        )

    if c.RUN_LULC:
        zones.append(
            (
                "lulc",
                _performance_lulc_zone(c, logger, land_cover_year),
                pd.DataFrame(
                    {
                        "zone_id": list(included_labels),
                        "lulc_name": list(included_labels.values()),
                    }
                ),
                "zone_id",
                "lulc_name",
                land_cover_year,
            )
        )

    if c.RUN_COUNTRY_LULC:
        zone_raster, lookup, zone_id, label = _performance_combined_zone(
            c,
            logger,
            period_key,
            "country_lulc",
        )
        zones.append(
            (
                "country_lulc",
                zone_raster,
                lookup,
                zone_id,
                label,
                land_cover_year,
            )
        )

    if c.RUN_ECOREGION_LULC:
        zone_raster, lookup, zone_id, label = _performance_combined_zone(
            c,
            logger,
            period_key,
            "ecoregion_lulc",
        )
        zones.append(
            (
                "ecoregion_lulc",
                zone_raster,
                lookup,
                zone_id,
                label,
                land_cover_year,
            )
        )

    return zones


def performance_transition_zone_defs(
    c,
    logger,
) -> list[tuple]:
    """
    Return transition zones masked to the recent 2022 LULC context.

    This matches the Results definition in which movement from the earlier to
    recent Performance class is reported within 2022 land-cover contexts.
    """

    return [
        (name, zone_raster, lookup, zone_id, label)
        for (
            name,
            zone_raster,
            lookup,
            zone_id,
            label,
            _land_cover_year,
        ) in performance_period_zone_defs(
            c,
            logger,
            "performance_2022",
        )
    ]


def run_performance(c, logger) -> None:
    """
    Aggregate fixed-baseline Performance within the included vegetated domain.

    This function is self-contained within the Performance module. It does not
    modify generic zones or the State, Trend and joint-STP branches.
    """

    logger.info("Starting Performance module")

    all_period_outputs: list[pd.DataFrame] = []
    all_transition_outputs: list[pd.DataFrame] = []
    all_direction_outputs: list[pd.DataFrame] = []
    period_outputs_by_unit: dict[str, list[pd.DataFrame]] = {}

    # ---------------------------------------------------------------------
    # Period-specific summaries. Each period uses its own assessment-year
    # vegetated LULC domain: 2011 for 2009–2011 and 2022 for 2020–2022.
    # ---------------------------------------------------------------------
    for period_key, performance_path in c.PERFORMANCE_RASTERS.items():
        period_zones = performance_period_zone_defs(
            c,
            logger,
            period_key,
        )

        for (
            name,
            zone_raster,
            lookup,
            zone_id,
            label,
            land_cover_reference_year,
        ) in period_zones:
            logger.info(
                "Processing Performance for %s, %s",
                name,
                period_key,
            )

            summary = categorical_summary(
                performance_path,
                zone_raster,
                c.PERFORMANCE_CLASS_LOOKUP,
                lookup,
                zone_id,
                c.PERFORMANCE_OUTPUT_DIR
                / f"performance_{name}_{period_key}.csv",
                c,
                set(c.PERFORMANCE_CLASS_LOOKUP),
            )

            summary = add_reporting_metadata(
                summary,
                name,
                zone_id,
                label,
            )
            summary = add_performance_period_metadata(
                summary,
                period_key,
            )
            summary["land_cover_reference_year"] = (
                land_cover_reference_year
            )
            summary["land_cover_domain_definition"] = (
                "Included vegetated LULC codes: "
                + ", ".join(
                    str(code)
                    for code in sorted(c.INCLUDED_LULC_CODES)
                )
            )

            output_path = (
                c.PERFORMANCE_OUTPUT_DIR
                / f"performance_{name}_{period_key}.csv"
            )
            summary.to_csv(output_path, index=False)

            # Retain the historical filename for compatibility.
            summary.to_csv(
                c.PERFORMANCE_OUTPUT_DIR
                / f"{period_key}_{name}.csv",
                index=False,
            )

            period_outputs_by_unit.setdefault(name, []).append(summary)
            all_period_outputs.append(summary)

    for name, summaries in period_outputs_by_unit.items():
        if summaries:
            pd.concat(summaries, ignore_index=True).to_csv(
                c.PERFORMANCE_OUTPUT_DIR
                / f"performance_{name}_all_periods.csv",
                index=False,
            )

    # ---------------------------------------------------------------------
    # Pixel-level transitions. These are masked to included vegetated classes
    # and attributed to the recent-period 2022 LULC context.
    # ---------------------------------------------------------------------
    for (
        name,
        zone_raster,
        lookup,
        zone_id,
        label,
    ) in performance_transition_zone_defs(c, logger):

        logger.info(
            "Processing Performance transitions for %s",
            name,
        )

        transitions = crosstab(
            c.PERFORMANCE_RASTERS["performance_2011"],
            c.PERFORMANCE_RASTERS["performance_2022"],
            zone_raster,
            c.PERFORMANCE_CLASS_LOOKUP,
            c.PERFORMANCE_CLASS_LOOKUP,
            lookup,
            zone_id,
            c.PERFORMANCE_OUTPUT_DIR
            / f"performance_transition_{name}_2011_to_2022.csv",
            c,
        )

        transitions["transition_code"] = (
            transitions["class_a_code"] * 10
            + transitions["class_b_code"]
        )

        # Retain legacy values for compatibility with existing downstream
        # tables while adding interpretation-neutral movement labels.
        transitions["transition_direction"] = "Unchanged"
        transitions.loc[
            transitions["class_b_code"] > transitions["class_a_code"],
            "transition_direction",
        ] = "Improving"
        transitions.loc[
            transitions["class_b_code"] < transitions["class_a_code"],
            "transition_direction",
        ] = "Worsening"

        transitions["transition_movement"] = "Same class"
        transitions.loc[
            transitions["class_b_code"] > transitions["class_a_code"],
            "transition_movement",
        ] = "Towards higher class"
        transitions.loc[
            transitions["class_b_code"] < transitions["class_a_code"],
            "transition_movement",
        ] = "Towards lower class"

        transitions["earlier_period_key"] = "performance_2011"
        transitions["earlier_period_label"] = (
            PERFORMANCE_PERIOD_METADATA["performance_2011"]["period_label"]
        )
        transitions["earlier_assessment_start"] = 2009
        transitions["earlier_assessment_end"] = 2011
        transitions["recent_period_key"] = "performance_2022"
        transitions["recent_period_label"] = (
            PERFORMANCE_PERIOD_METADATA["performance_2022"]["period_label"]
        )
        transitions["recent_assessment_start"] = 2020
        transitions["recent_assessment_end"] = 2022
        transitions["baseline_start"] = 2001
        transitions["baseline_end"] = 2003
        transitions["baseline_label"] = (
            "Fixed 2001–2003 same-pixel baseline"
        )
        transitions["land_cover_reference_year"] = 2022
        transitions["transition_definition"] = (
            "Pixel-level movement among five fixed-baseline Performance "
            "classes from the 2009–2011 assessment to the 2020–2022 "
            "assessment. Movement is attributed to included vegetated "
            "land-cover classes in 2022."
        )
        transitions["share_denominator"] = (
            "Pixels with valid Performance classes in both assessment periods "
            "and an included vegetated 2022 land-cover class within the "
            "reporting unit."
        )
        transitions = add_reporting_metadata(
            transitions,
            name,
            zone_id,
            label,
        )

        transitions.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / f"performance_transition_{name}_2011_to_2022.csv",
            index=False,
        )
        transitions.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / f"performance_transition_{name}.csv",
            index=False,
        )
        all_transition_outputs.append(transitions)

        direction = (
            transitions.groupby(
                [zone_id, "transition_direction", "transition_movement"],
                as_index=False,
                observed=True,
            )["pixel_count"]
            .sum()
        )

        direction["share_pct"] = (
            100.0
            * direction["pixel_count"]
            / direction.groupby(zone_id)["pixel_count"].transform("sum")
        )
        direction = direction.merge(
            lookup,
            on=zone_id,
            how="left",
        )
        direction["earlier_period_key"] = "performance_2011"
        direction["earlier_period_label"] = (
            PERFORMANCE_PERIOD_METADATA["performance_2011"]["period_label"]
        )
        direction["recent_period_key"] = "performance_2022"
        direction["recent_period_label"] = (
            PERFORMANCE_PERIOD_METADATA["performance_2022"]["period_label"]
        )
        direction["baseline_label"] = (
            "Fixed 2001–2003 same-pixel baseline"
        )
        direction["land_cover_reference_year"] = 2022
        direction["direction_definition"] = (
            "Movement towards a lower class, persistence in the same class, "
            "or movement towards a higher fixed-baseline Performance class "
            "between the two assessment periods."
        )
        direction["share_denominator"] = (
            "Pixels valid in both assessment periods and within the included "
            "vegetated 2022 land-cover domain."
        )
        direction = add_reporting_metadata(
            direction,
            name,
            zone_id,
            label,
        )

        direction.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / f"performance_direction_{name}_2011_to_2022.csv",
            index=False,
        )
        direction.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / f"performance_direction_{name}.csv",
            index=False,
        )
        all_direction_outputs.append(direction)

        # Existing workflow figures are retained for compatibility. The
        # publication-specific Performance figures will be handled separately.
        if c.CREATE_FIGURES and name in {"country", "ecoregion"}:
            max_units = (
                None
                if name == "country"
                else c.TOP_N_UNITS_IN_FIGURES
            )

            labelled_stacked_bar(
                df=direction,
                unit_col=label,
                class_col="transition_direction",
                value_col="share_pct",
                title=(
                    "Fixed-baseline Performance movement "
                    f"by {name}\n"
                    "2009–2011 to 2020–2022; both assessments relative to "
                    "the fixed 2001–2003 same-pixel baseline"
                ),
                x_label=(
                    "Share of included vegetated pixels valid in both "
                    "assessment periods (%)"
                ),
                y_label=(
                    "Country" if name == "country" else "Ecoregion"
                ),
                output_base=(
                    c.FIGURE_OUTPUT_DIR
                    / f"performance_{name}_direction_2011_to_2022"
                ),
                c=c,
                class_order=PERFORMANCE_DIRECTION_ORDER,
                sort_by_class="Worsening",
                ascending=True,
                max_units=max_units,
                class_colours=PERFORMANCE_DIRECTION_COLOURS,
            )

    if all_period_outputs:
        master_periods = pd.concat(
            all_period_outputs,
            ignore_index=True,
        )
        master_periods.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / "performance_all_reporting_units_all_periods.csv",
            index=False,
        )

        (
            master_periods.groupby(
                [
                    "reporting_unit_type",
                    "period_key",
                    "period_label",
                    "reporting_year",
                    "assessment_start",
                    "assessment_end",
                    "class_code",
                    "class_name",
                ],
                as_index=False,
                observed=True,
            )
            .agg(
                total_pixel_count=("pixel_count", "sum"),
                total_area_ha=("area_ha", "sum"),
                mean_unit_share_pct=("share_pct", "mean"),
                median_unit_share_pct=("share_pct", "median"),
            )
            .to_csv(
                c.PERFORMANCE_OUTPUT_DIR
                / "performance_summary_by_period_class_and_unit.csv",
                index=False,
            )
        )

    if all_transition_outputs:
        pd.concat(
            all_transition_outputs,
            ignore_index=True,
        ).to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / "performance_transitions_all_reporting_units_2011_to_2022.csv",
            index=False,
        )

    if all_direction_outputs:
        master_direction = pd.concat(
            all_direction_outputs,
            ignore_index=True,
        )
        master_direction.to_csv(
            c.PERFORMANCE_OUTPUT_DIR
            / "performance_direction_all_reporting_units_2011_to_2022.csv",
            index=False,
        )

        (
            master_direction.groupby(
                [
                    "reporting_unit_type",
                    "transition_direction",
                    "transition_movement",
                ],
                as_index=False,
                observed=True,
            )
            .agg(
                total_pixel_count=("pixel_count", "sum"),
                mean_unit_share_pct=("share_pct", "mean"),
                median_unit_share_pct=("share_pct", "median"),
                minimum_unit_share_pct=("share_pct", "min"),
                maximum_unit_share_pct=("share_pct", "max"),
            )
            .to_csv(
                c.PERFORMANCE_OUTPUT_DIR
                / "performance_direction_statistical_summary.csv",
                index=False,
            )
        )

    logger.info("Performance module complete")
# =============================================================================
# 8. JOINT STP RULES
# =============================================================================

def group_rule(
    state: int,
    trend: int,
    performance: int,
) -> int:
    """
    Collapse the 45 possible State × Trend × Performance combinations into
    eight diagnostic groups.
    """

    loss = performance in (1, 2)
    stable = performance == 3
    gain = performance in (4, 5)

    if (
        state == 1
        and trend == 1
        and loss
    ):
        return 1

    if (
        state == 1
        and trend == 3
        and gain
    ):
        return 3

    if (
        state == 1
        and trend == 2
        and (loss or stable)
    ):
        return 2

    if (
        state == 3
        and trend in (1, 2)
        and loss
    ):
        return 7

    if (
        state in (2, 3)
        and trend == 3
        and gain
    ):
        return 6

    if (
        state == 2
        and trend == 2
        and stable
    ):
        return 5

    if (
        state in (2, 3)
        and (
            trend == 1
            or loss
        )
    ):
        return 4

    return 8


# =============================================================================
# 9. BUILD JOINT STP RASTERS
# =============================================================================

def build_joint(c) -> tuple[Path, Path]:
    """
    Build the 45-class joint STP raster and the eight-group diagnostic raster.
    """

    state_path = c.STATE_RASTERS[
        c.PRIMARY_STATE_FORMULATION
    ]

    trend_path = c.TREND_DIRECTION_RASTERS[
        "full_2001_2022"
    ]

    performance_path = c.PERFORMANCE_RASTERS[
        "performance_2022"
    ]

    joint_output_path = (
        c.JOINT_OUTPUT_DIR
        / "joint_stp_45class_EA250m.tif"
    )

    group_output_path = (
        c.JOINT_OUTPUT_DIR
        / "joint_stp_diagnostic_group_EA250m.tif"
    )

    with (
        rasterio.open(state_path) as state_src,
        rasterio.open(trend_path) as trend_src,
        rasterio.open(performance_path) as perf_src,
    ):
        profile = state_src.profile.copy()

        profile.update(
            dtype="int16",
            nodata=0,
            compress="deflate",
        )

        with (
            rasterio.open(
                joint_output_path,
                "w",
                **profile,
            ) as joint_dst,
            rasterio.open(
                group_output_path,
                "w",
                **profile,
            ) as group_dst,
        ):
            for _, window in state_src.block_windows(1):
                state = state_src.read(
                    1,
                    window=window,
                    masked=True,
                )

                trend = trend_src.read(
                    1,
                    window=window,
                    masked=True,
                )

                performance = perf_src.read(
                    1,
                    window=window,
                    masked=True,
                )

                valid = (
                    (~state.mask)
                    & (~trend.mask)
                    & (~performance.mask)
                    & np.isin(
                        state.data,
                        [1, 2, 3],
                    )
                    & np.isin(
                        trend.data,
                        [1, 2, 3],
                    )
                    & np.isin(
                        performance.data,
                        [1, 2, 3, 4, 5],
                    )
                )

                joint = np.zeros(
                    state.shape,
                    dtype=np.int16,
                )

                groups = np.zeros(
                    state.shape,
                    dtype=np.int16,
                )

                state_values = (
                    state.data[valid].astype(int)
                )

                trend_values = (
                    trend.data[valid].astype(int)
                )

                performance_values = (
                    performance.data[valid]
                    .astype(int)
                )

                joint[valid] = (
                    state_values * 100
                    + trend_values * 10
                    + performance_values
                )

                groups[valid] = np.array(
                    [
                        group_rule(
                            state_code,
                            trend_code,
                            performance_code,
                        )
                        for (
                            state_code,
                            trend_code,
                            performance_code,
                        ) in zip(
                            state_values,
                            trend_values,
                            performance_values,
                        )
                    ],
                    dtype=np.int16,
                )

                joint_dst.write(
                    joint,
                    1,
                    window=window,
                )

                group_dst.write(
                    groups,
                    1,
                    window=window,
                )

    return (
        joint_output_path,
        group_output_path,
    )


# =============================================================================
# 10. JOINT STP MODULE
# =============================================================================

def run_joint(c, logger) -> None:
    """
    Build and aggregate joint STP combinations and diagnostic groups.
    """

    logger.info("Starting joint STP module")

    joint_path, group_path = build_joint(c)

    stp_lookup = {
        (
            state * 100
            + trend * 10
            + performance
        ): (
            f"{c.STATE_CLASS_LOOKUP[state]} | "
            f"{c.TREND_CLASS_LOOKUP[trend]} | "
            f"{c.PERFORMANCE_CLASS_LOOKUP[performance]}"
        )
        for state in [1, 2, 3]
        for trend in [1, 2, 3]
        for performance in [1, 2, 3, 4, 5]
    }

    for (
        name,
        zone_raster,
        lookup,
        zone_id,
        label,
    ) in zone_defs(c):

        logger.info(
            "Processing joint STP for %s",
            name,
        )

        categorical_summary(
            joint_path,
            zone_raster,
            stp_lookup,
            lookup,
            zone_id,
            (
                c.JOINT_OUTPUT_DIR
                / f"joint_45class_{name}.csv"
            ),
            c,
            set(stp_lookup),
        )

        group_summary = categorical_summary(
            group_path,
            zone_raster,
            GROUPS,
            lookup,
            zone_id,
            (
                c.JOINT_OUTPUT_DIR
                / f"joint_groups_{name}.csv"
            ),
            c,
            set(GROUPS),
        )

        if (
            c.CREATE_FIGURES
            and name in {"country", "ecoregion"}
        ):
            max_units = (
                None
                if name == "country"
                else c.TOP_N_UNITS_IN_FIGURES
            )

            labelled_stacked_bar(
                df=group_summary,
                unit_col=label,
                class_col="class_name",
                value_col="share_pct",
                title=(
                    "Joint benchmark-relative State, "
                    "full-period Trend and fixed-baseline "
                    f"Performance by {name}\n"
                    "State: ecological–parametric 2022; "
                    "Trend: strict 2001–2022; "
                    "Performance: 2022 relative to "
                    "the 2001–2003 baseline"
                ),
                x_label=(
                    "Share of common valid vegetated area (%)"
                ),
                y_label=(
                    "Country"
                    if name == "country"
                    else "Ecoregion"
                ),
                output_base=(
                    c.FIGURE_OUTPUT_DIR
                    / f"joint_{name}_groups"
                ),
                c=c,
                class_order=JOINT_GROUP_ORDER,
                sort_by_class="Compounding concern",
                ascending=True,
                max_units=max_units,
            )

    logger.info("Joint STP module complete")