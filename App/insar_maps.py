
from pathlib import Path


# ============================================================
# AVAILABLE InSAR DATA
# ============================================================

INSAR_YEARS = [1395]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSAR_1395_ROOT = (
    PROJECT_ROOT
    / "SourceData"
    / "InSAR"
    / "1395"
)


INSAR_1395_TIMESERIES_DIR = (
    INSAR_1395_ROOT
    / "timeseries_Subsidence_100m_ALPRIFT"
)


# ============================================================
# AVAILABILITY
# ============================================================

def available_insar_years():
    """
    Return model years for which an InSAR observation
    package is currently available.
    """
    return list(INSAR_YEARS)


def has_insar_data(year):
    """
    Check whether InSAR data are currently available
    for the requested ALPRIFT model year.
    """
    return int(year) in INSAR_YEARS


# ============================================================
# PATHS
# ============================================================

def resolve_insar_paths(year):
    """
    Resolve the InSAR data package associated with an
    ALPRIFT model year.

    Important:
    The 1395 product is an independent observation and
    is not part of the ALPRIFT SVI equation.
    """

    year = int(year)

    if year != 1395:
        raise ValueError(
            f"InSAR data are not available for model year {year}."
        )

    return {
        "final": (
            INSAR_1395_ROOT
            / "Shabestar_1395_Subsidence_100m_FINAL.tif"
        ),

        "native_80m": (
            INSAR_1395_ROOT
            / "Shabestar_1395_Subsidence_mm.tif"
        ),

        "los": (
            INSAR_1395_ROOT
            / "Shabestar_1395_LOS_Displacement_mm.tif"
        ),

        "vertical": (
            INSAR_1395_ROOT
            / "Shabestar_1395_Vertical_Displacement_mm.tif"
        ),

        "timeseries_dir": (
            INSAR_1395_TIMESERIES_DIR
        ),

        "common_mask": (
            INSAR_1395_TIMESERIES_DIR
            / "CommonMask_ALPRIFT_InSAR_1395.tif"
        ),

        "timeseries_csv": (
            INSAR_1395_TIMESERIES_DIR
            / "Shabestar_Subsidence_TimeSeries_100m_ALPRIFT.csv"
        ),
    }


# ============================================================
# SCIENTIFIC METADATA
# ============================================================

def get_insar_metadata(year):
    """
    Scientific metadata for the InSAR product associated
    with the requested ALPRIFT model year.

    The displayed product is NOT an annual subsidence rate.

    It is an estimated cumulative vertical subsidence product
    derived from Sentinel-1 SBAS LOS displacement, under the
    assumption that horizontal displacement is negligible.
    """

    year = int(year)

    if year != 1395:
        raise ValueError(
            f"InSAR metadata are not available for model year {year}."
        )

    return {
        "year": 1395,

        "method": "SBAS-InSAR",

        "sensor": "Sentinel-1",

        "source_measurement": "LOS displacement",

        "quantity": (
            "Estimated cumulative vertical subsidence"
        ),

        "is_rate": False,

        "start_date": "2015-12-29",

        "end_date": "2016-12-23",

        "reference_date": "2015-12-29",

        "unit": "mm",

        "positive_meaning": "subsidence",

        "negative_meaning": (
            "upward movement relative to reference"
        ),

        "crs": "EPSG:32638",

        "resolution_m": 100,

        "width": 533,

        "height": 206,

        "resampling": "Bilinear",

        "common_valid_cells": 47260,

        "common_area_km2": 472.6,

        "temporal_relation_note": (
            "Associated with ALPRIFT model year 1395, "
            "but the InSAR period is not exactly identical "
            "to the Azar-to-Azar period used for layer T."
        ),

        "scientific_note": (
            "Estimated vertical displacement derived from "
            "SBAS-InSAR LOS displacement assuming negligible "
            "horizontal motion."
        ),
    }



# ============================================================
# RASTER INSPECTION
# ============================================================

def inspect_insar_raster(year):
    """
    Read the final dashboard-ready InSAR raster directly
    from disk and report its actual spatial properties
    and basic statistics.

    Statistics are calculated only over valid cells.
    NoData values are converted to NaN before calculation.
    """

    import numpy as np
    import rasterio

    paths = resolve_insar_paths(year)
    raster_path = paths["final"]

    if not raster_path.exists():
        return {
            "exists": False,
            "path": raster_path,
        }

    with rasterio.open(raster_path) as src:

        masked = src.read(
            1,
            masked=True
        )

        # Convert safely to float before filling with NaN.
        data = np.asarray(
            masked.astype(np.float64).filled(np.nan),
            dtype=np.float64
        )

        valid = np.isfinite(data)

        if np.any(valid):
            minimum = float(
                np.nanmin(data)
            )

            maximum = float(
                np.nanmax(data)
            )

            mean = float(
                np.nanmean(data)
            )

            valid_cells = int(
                np.count_nonzero(valid)
            )

        else:
            minimum = None
            maximum = None
            mean = None
            valid_cells = 0

        return {
            "exists": True,

            "path": raster_path,

            "width": int(src.width),
            "height": int(src.height),

            "crs": (
                src.crs.to_string()
                if src.crs is not None
                else None
            ),

            "resolution_x": float(
                abs(src.transform.a)
            ),

            "resolution_y": float(
                abs(src.transform.e)
            ),

            "nodata": src.nodata,

            "bounds": src.bounds,

            "dtype": src.dtypes[0],

            "valid_cells": valid_cells,

            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
        }


# ============================================================
# InSAR TIME-SERIES FILES
# ============================================================

def list_insar_timeseries_rasters(year):
    """
    Return the cumulative InSAR time-series GeoTIFFs
    sorted chronologically by their date-bearing filenames.

    Only Subsidence_Cumulative_* files are included,
    therefore CommonMask and other TIFFs are excluded.
    """

    paths = resolve_insar_paths(year)

    timeseries_dir = paths["timeseries_dir"]

    if not timeseries_dir.exists():
        return []

    rasters = sorted(
        timeseries_dir.glob(
            "Subsidence_Cumulative_*"
            "_ref_20151229_100m_ALPRIFT.tif"
        ),
        key=lambda p: p.name
    )

    return rasters



# ============================================================
# COMMON MASK INSPECTION
# ============================================================

def inspect_common_mask(year):
    """
    Inspect the common-validity mask between ALPRIFT
    and the InSAR time series.

    Cells with values > 0 are counted as common valid cells.
    """

    import numpy as np
    import rasterio

    paths = resolve_insar_paths(year)
    mask_path = paths["common_mask"]

    if not mask_path.exists():
        return {
            "exists": False,
            "path": mask_path,
        }

    with rasterio.open(mask_path) as src:

        masked = src.read(
            1,
            masked=True
        )

        data = np.asarray(
            masked.astype(np.float64).filled(np.nan),
            dtype=np.float64
        )

        common = np.isfinite(data) & (data > 0)

        common_cells = int(
            np.count_nonzero(common)
        )

        pixel_area_m2 = (
            abs(float(src.transform.a))
            *
            abs(float(src.transform.e))
        )

        common_area_km2 = (
            common_cells
            * pixel_area_m2
            / 1_000_000.0
        )

        return {
            "exists": True,

            "path": mask_path,

            "width": int(src.width),
            "height": int(src.height),

            "crs": (
                src.crs.to_string()
                if src.crs is not None
                else None
            ),

            "resolution_x": float(
                abs(src.transform.a)
            ),

            "resolution_y": float(
                abs(src.transform.e)
            ),

            "nodata": src.nodata,

            "common_cells": common_cells,

            "common_area_km2": round(
                common_area_km2,
                1
            ),
        }



# ============================================================
# InSAR TIME-SERIES CATALOG
# ============================================================

def get_insar_timeseries_catalog(year):
    """
    Build a chronological catalog of cumulative InSAR
    time-series rasters.

    Dates are extracted from the actual raster filenames,
    rather than being hard-coded.
    """

    import re
    from datetime import datetime

    rasters = list_insar_timeseries_rasters(year)

    catalog = []

    pattern = re.compile(
        r"Subsidence_Cumulative_(\d{8})_"
        r"ref_\d{8}_100m_ALPRIFT\.tif$"
    )

    for raster_path in rasters:

        match = pattern.search(
            raster_path.name
        )

        if match is None:
            continue

        date_compact = match.group(1)

        date_value = datetime.strptime(
            date_compact,
            "%Y%m%d"
        )

        catalog.append(
            {
                "date": date_value.strftime(
                    "%Y-%m-%d"
                ),

                "date_compact": date_compact,

                "path": raster_path,
            }
        )

    catalog.sort(
        key=lambda item: item["date"]
    )

    return catalog



# ============================================================
# InSAR TIME-SERIES TABLE
# ============================================================

def load_insar_timeseries_table(year):
    """
    Load the dashboard-ready InSAR time-series CSV.

    The CSV stores dates in YYYYMMDD form. This function
    converts date and reference_date to ISO YYYY-MM-DD
    strings while preserving the scientific data columns.
    """

    import pandas as pd

    paths = resolve_insar_paths(year)
    csv_path = paths["timeseries_csv"]

    if not csv_path.exists():
        raise FileNotFoundError(
            f"InSAR time-series CSV not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        dtype={
            "date": str,
            "reference_date": str,
            "filename": str,
            "unit": str,
            "quantity": str,
            "grid": str,
        },
    )

    required_columns = [
        "date",
        "reference_date",
        "filename",
        "unit",
        "quantity",
        "grid",
        "valid_common_cells",
        "mean_mm",
        "median_mm",
        "min_mm",
        "max_mm",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required InSAR time-series columns: "
            + ", ".join(missing)
        )

    # Convert compact YYYYMMDD dates to ISO YYYY-MM-DD.
    df["date"] = (
        pd.to_datetime(
            df["date"],
            format="%Y%m%d"
        )
        .dt.strftime("%Y-%m-%d")
    )

    df["reference_date"] = (
        pd.to_datetime(
            df["reference_date"],
            format="%Y%m%d"
        )
        .dt.strftime("%Y-%m-%d")
    )

    # Keep the expected scientific column order.
    df = df[required_columns].copy()

    return df



# ============================================================
# PIEZOMETER DATA
# ============================================================

def load_piezometers(year, root=None):
    """
    Load annual ALPRIFT T piezometer points.

    Expected source:
    SourceData/Piezometers/
    T_<year>_Points.shp
    """

    import geopandas as gpd

    year = int(year)

    if root is None:
        root = Path(
            str(PROJECT_ROOT)
        )
    else:
        root = Path(root)

    shp_path = (
        root
        / "SourceData"
        / "Piezometers"
        / f"T_{year}_Points.shp"
    )

    if not shp_path.exists():
        raise FileNotFoundError(
            f"Piezometer shapefile not found: {shp_path}"
        )

    gdf = gpd.read_file(
        shp_path
    )

    required_columns = [
        "Piezometer",
        "T_myr",
        "H_prev_m9",
        "H_curr_m9",
        "Hydro_Stat",
        "geometry",
    ]

    missing = [
        column
        for column in required_columns
        if column not in gdf.columns
    ]

    if missing:
        raise ValueError(
            "Missing required piezometer fields: "
            + ", ".join(missing)
        )

    # InSAR rasters use the ALPRIFT grid in EPSG:32638.
    target_crs = "EPSG:32638"

    if gdf.crs is None:
        raise ValueError(
            "Piezometer layer has no CRS."
        )

    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(
            target_crs
        )

    return gdf


# ============================================================
# InSAR TIME SERIES AT A PIEZOMETER
# ============================================================

def extract_insar_timeseries_at_piezometer(
    year,
    piezometer_id,
    root=None,
):
    """
    Extract cumulative InSAR displacement at the location
    of one piezometer for every available InSAR acquisition.

    Important:
    The returned InSAR values are satellite-derived estimated
    cumulative vertical displacement values at the piezometer
    location. They are NOT groundwater-level measurements.
    """

    import numpy as np
    import pandas as pd
    import rasterio

    year = int(year)
    piezometer_id = str(
        piezometer_id
    )

    if not has_insar_data(year):
        raise ValueError(
            f"InSAR data are not available for model year {year}."
        )

    piezometers = load_piezometers(
        year,
        root=root,
    )

    selected = piezometers.loc[
        piezometers["Piezometer"].astype(str)
        == piezometer_id
    ]

    if selected.empty:
        raise ValueError(
            f"Unknown piezometer: {piezometer_id}"
        )

    point = selected.iloc[0]

    x = float(
        point.geometry.x
    )

    y = float(
        point.geometry.y
    )

    catalog = get_insar_timeseries_catalog(
        year
    )

    rows = []

    for item in catalog:

        raster_path = item["path"]

        with rasterio.open(
            raster_path
        ) as src:

            sampled = next(
                src.sample(
                    [(x, y)],
                    masked=True
                )
            )

            value = sampled[0]

            if np.ma.is_masked(value):
                insar_mm = np.nan

            else:
                insar_mm = float(
                    value
                )

                if (
                    src.nodata is not None
                    and np.isclose(
                        insar_mm,
                        float(src.nodata)
                    )
                ):
                    insar_mm = np.nan

        rows.append(
            {
                "date": item["date"],
                "Piezometer": piezometer_id,
                "x": x,
                "y": y,
                "insar_mm": insar_mm,
                "T_myr": float(
                    point["T_myr"]
                ),
                "H_prev_m9": float(
                    point["H_prev_m9"]
                ),
                "H_curr_m9": float(
                    point["H_curr_m9"]
                ),
                "Hydro_Stat": str(
                    point["Hydro_Stat"]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )
