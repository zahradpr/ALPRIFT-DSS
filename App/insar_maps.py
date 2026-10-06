
from pathlib import Path


# ============================================================
# AVAILABLE InSAR DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Years that have been prepared and approved for the dashboard.
# Add future completed years here one by one.
INSAR_CONFIG = {

    1395: {
        "start_date": "2015-12-29",
        "end_date": "2016-12-23",
        "reference_date": "2015-12-29",
    },

    1396: {
        "start_date": "2017-03-17",
        "end_date": "2018-03-12",
        "reference_date": "2017-03-17",
    },

}


INSAR_YEARS = sorted(
    INSAR_CONFIG.keys()
)


def _first_existing(*paths):
    """
    Return the first existing path.

    If none exists, return the first candidate so that
    downstream error messages still show the expected path.
    """

    candidates = [
        Path(p)
        for p in paths
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


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
    Resolve the InSAR package associated with an ALPRIFT
    model year.

    The function supports year-specific folders while keeping
    the same dashboard interface for all available years.
    """

    year = int(year)

    if year not in INSAR_CONFIG:
        raise ValueError(
            f"InSAR data are not available for model year {year}."
        )

    year_root = (
        PROJECT_ROOT
        / "SourceData"
        / "InSAR"
        / str(year)
    )

    timeseries_dir = (
        year_root
        / "timeseries_Subsidence_100m_ALPRIFT"
    )

    common_mask = _first_existing(

        year_root
        / f"CommonMask_ALPRIFT_InSAR_{year}.tif",

        timeseries_dir
        / f"CommonMask_ALPRIFT_InSAR_{year}.tif",
    )

    timeseries_csv = _first_existing(

        year_root
        / f"Shabestar_Subsidence_TimeSeries_100m_ALPRIFT_{year}.csv",

        year_root
        / "Shabestar_Subsidence_TimeSeries_100m_ALPRIFT.csv",

        timeseries_dir
        / f"Shabestar_Subsidence_TimeSeries_100m_ALPRIFT_{year}.csv",

        timeseries_dir
        / "Shabestar_Subsidence_TimeSeries_100m_ALPRIFT.csv",
    )

    return {

        "final": (
            year_root
            / f"Shabestar_{year}_Subsidence_100m_FINAL.tif"
        ),

        "native_80m": (
            year_root
            / f"Shabestar_{year}_Subsidence_mm.tif"
        ),

        "los": (
            year_root
            / f"Shabestar_{year}_LOS_Displacement_mm.tif"
        ),

        "vertical": (
            year_root
            / f"Shabestar_{year}_Vertical_Displacement_mm.tif"
        ),

        "timeseries_dir": timeseries_dir,

        "common_mask": common_mask,

        "timeseries_csv": timeseries_csv,
    }


# ============================================================
# SCIENTIFIC METADATA
# ============================================================

def get_insar_metadata(year):
    """
    Scientific metadata for the InSAR observation associated
    with the requested ALPRIFT model year.

    The displayed product is cumulative vertical displacement,
    not an annual rate.
    """

    year = int(year)

    if year not in INSAR_CONFIG:
        raise ValueError(
            f"InSAR metadata are not available for model year {year}."
        )

    config = INSAR_CONFIG[year]

    raster_info = inspect_insar_raster(
        year
    )

    mask_info = inspect_common_mask(
        year
    )

    return {

        "year": year,

        "method": "SBAS-InSAR",

        "sensor": "Sentinel-1",

        "source_measurement": "LOS displacement",

        "quantity": (
            "Estimated cumulative vertical subsidence"
        ),

        "is_rate": False,

        "start_date": config["start_date"],

        "end_date": config["end_date"],

        "reference_date": config["reference_date"],

        "unit": "mm",

        "positive_meaning": "subsidence",

        "negative_meaning": (
            "upward movement relative to reference"
        ),

        "crs": raster_info.get("crs"),

        "resolution_m": (
            raster_info.get("resolution_x")
        ),

        "width": raster_info.get("width"),

        "height": raster_info.get("height"),

        "resampling": "Bilinear",

        "common_valid_cells": (
            mask_info.get(
                "common_cells",
                0
            )
        ),

        "common_area_km2": (
            mask_info.get(
                "common_area_km2",
                0.0
            )
        ),

        "temporal_relation_note": (
            f"Associated with ALPRIFT model year {year}, "
            "but the InSAR observation period is not necessarily "
            "identical to the Azar-to-Azar period used for layer T."
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
    Return cumulative InSAR time-series rasters for any
    configured model year.

    Acquisition and reference dates are read from filenames;
    no reference date is hard-coded.
    """

    paths = resolve_insar_paths(
        year
    )

    timeseries_dir = paths[
        "timeseries_dir"
    ]

    if not timeseries_dir.exists():
        return []

    rasters = sorted(

        timeseries_dir.glob(
            "Subsidence_Cumulative_*"
            "_ref_*_100m_ALPRIFT.tif"
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
    D:/ALPRIFT_DSS/SourceData/Piezometers/
    T_<year>_Points.shp
    """

    import geopandas as gpd

    year = int(year)

    if root is None:
        root = PROJECT_ROOT
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
