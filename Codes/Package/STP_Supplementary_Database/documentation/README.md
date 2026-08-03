# STP Supplementary Database — Stage 1

This package aggregates existing State, Trend, Performance and joint STP rasters by country, ecoregion, LULC, country × LULC and ecoregion × LULC. It does not recalculate the original metrics.

## Important LULC decision

Only the display label of class 11 is changed:

`11 = Natural vegetation mosaic`

Its raster code and analytical treatment are unchanged. It combines the existing CCI classes 100, 110 and minor class 140, so it should be interpreted cautiously and not used as a headline example of a homogeneous ecological class.

## Files

- `00_config.py`: paths, switches, lookups and quality thresholds.
- `01_run_supplementary_STP.py`: master runner.
- `modules/core.py`: raster validation, rasterisation, aggregation, statistics and plotting.
- `modules/workflows.py`: State, Trend, Performance and joint STP modules.
- `lookups/`: class interpretation tables.
- `documentation/`: metadata and interpretation guidance.

## Installation

```bash
pip install -r requirements.txt
```

## Run sequence

### Run 1 — State

```python
RUN_PREPARE_ZONES = True
RUN_STATE = True
RUN_TREND = False
RUN_PERFORMANCE = False
RUN_JOINT_STP = False
```

### Run 2 — Trend

```python
RUN_PREPARE_ZONES = False
RUN_STATE = False
RUN_TREND = True
RUN_PERFORMANCE = False
RUN_JOINT_STP = False
```

### Run 3 — Performance

```python
RUN_PREPARE_ZONES = False
RUN_STATE = False
RUN_TREND = False
RUN_PERFORMANCE = True
RUN_JOINT_STP = False
```

### Run 4 — Joint STP

```python
RUN_PREPARE_ZONES = False
RUN_STATE = False
RUN_TREND = False
RUN_PERFORMANCE = False
RUN_JOINT_STP = True
```

Run with:

```bash
python 01_run_supplementary_STP.py
```

## Vector fields

Countries: `ISO_A3` as unique ID and `NAM_0` as display name.

Ecoregions: `ECO_NAME_CODE` as ID and `ECO_NAME` as display name. Duplicate codes are dissolved.

## Quality controls

- High: ≥1,600 valid pixels and ≥10,000 ha.
- Moderate: ≥400 valid pixels and ≥2,500 ha.
- Low: 100–399 valid pixels.
- Suppressed: <100 valid pixels.
- Jaccard is withheld where the union or class support is too small.
- Weighted kappa is withheld for small samples or single-class units.
- Trend retention is withheld where potential decline contains fewer than 100 pixels.

## Interpretation

State sensitivity shows dependence of condition attribution on reference and thresholds. Trend funneling qualifies directional evidence; it does not validate degradation. Performance describes fixed-baseline displacement, not restoration success. Joint STP groups are diagnostic, not official degradation or intervention classes.
