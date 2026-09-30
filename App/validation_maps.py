
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from scipy.stats import pearsonr, spearmanr

from model_maps import resolve_model_output_paths
from insar_maps import resolve_insar_paths


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High",
}


def _read_raster(path):

    path = Path(path)

    with rasterio.open(path) as src:

        arr = src.read(
            1,
            masked=True
        )

        data = np.asarray(
            arr.astype(np.float64).filled(np.nan),
            dtype=np.float64
        )

        return {
            "data": data,
            "crs": src.crs,
            "transform": src.transform,
            "shape": data.shape,
            "path": path,
        }


def condition_insar_to_csvi(
    values,
    svi_min=24.0,
    svi_max=240.0,
):

    """
    Condition InSAR observations to the ALPRIFT SVI range.

    This is algebraically equivalent to Eq. (2) used in the
    TSVI-Salmas (2021) and DSVI-Hadishahr (2022) studies:

    CSVI =
        ((S_i - S_max) / (S_min - S_max)) * 24
        +
        ((S_i - S_min) / (S_max - S_min)) * 240
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    obs_min = float(
        np.nanmin(values)
    )

    obs_max = float(
        np.nanmax(values)
    )

    if np.isclose(
        obs_min,
        obs_max
    ):
        raise ValueError(
            "InSAR minimum and maximum are identical."
        )

    csvi = (
        (
            (values - obs_max)
            /
            (obs_min - obs_max)
        )
        * svi_min
        +
        (
            (values - obs_min)
            /
            (obs_max - obs_min)
        )
        * svi_max
    )

    return csvi


def compute_alprift_insar_validation(
    year=1395,
    root=DEFAULT_ROOT,
):

    root = Path(root)

    output_paths = resolve_model_output_paths(
        year,
        root=root
    )

    insar_paths = resolve_insar_paths(
        year
    )

    svi_path = Path(
        output_paths["svi"]
    )

    rank_path = Path(
        output_paths["rank"]
    )

    insar_path = Path(
        insar_paths["final"]
    )

    mask_path = Path(
        insar_paths["common_mask"]
    )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    svi = _read_raster(
        svi_path
    )

    rank = _read_raster(
        rank_path
    )

    insar = _read_raster(
        insar_path
    )

    common = _read_raster(
        mask_path
    )


    # --------------------------------------------------------
    # GRID VALIDATION
    # --------------------------------------------------------

    rasters = [
        svi,
        rank,
        insar,
        common,
    ]

    shapes = {
        r["shape"]
        for r in rasters
    }

    crs_values = {
        str(r["crs"])
        for r in rasters
    }

    transforms = {
        tuple(r["transform"])
        for r in rasters
    }

    if len(shapes) != 1:
        raise ValueError(
            "Validation rasters do not have the same shape."
        )

    if len(crs_values) != 1:
        raise ValueError(
            "Validation rasters do not have the same CRS."
        )

    if len(transforms) != 1:
        raise ValueError(
            "Validation rasters do not have the same grid transform."
        )


    # --------------------------------------------------------
    # COMMON MASK
    # --------------------------------------------------------

    valid = (
        np.isfinite(svi["data"])
        &
        np.isfinite(rank["data"])
        &
        np.isfinite(insar["data"])
        &
        np.isfinite(common["data"])
        &
        (common["data"] > 0)
    )

    svi_values = svi["data"][
        valid
    ]

    rank_values = rank["data"][
        valid
    ].astype(int)

    insar_values = insar["data"][
        valid
    ]


    # --------------------------------------------------------
    # CSVI
    # --------------------------------------------------------

    csvi_values = condition_insar_to_csvi(
        insar_values
    )


    # --------------------------------------------------------
    # ASSOCIATION
    # --------------------------------------------------------

    pearson_r, pearson_p = pearsonr(
        svi_values,
        insar_values
    )

    pearson_csvi_r, pearson_csvi_p = pearsonr(
        svi_values,
        csvi_values
    )

    spearman_rho, spearman_p = spearmanr(
        svi_values,
        insar_values
    )

    pearson_r2 = float(
        pearson_r ** 2
    )


    # --------------------------------------------------------
    # PIXEL TABLE
    # --------------------------------------------------------

    pixel_df = pd.DataFrame(
        {
            "SVI": svi_values,
            "InSAR_mm": insar_values,
            "CSVI_24_240": csvi_values,
            "Class_ID": rank_values,
        }
    )

    pixel_df["Class"] = (
        pixel_df["Class_ID"]
        .map(CLASS_NAMES)
        .fillna("Unknown")
    )


    # --------------------------------------------------------
    # CLASS STATISTICS
    # --------------------------------------------------------

    rows = []

    for class_id in sorted(
        pixel_df["Class_ID"].unique()
    ):

        subset = pixel_df.loc[
            pixel_df["Class_ID"]
            == class_id,
            "InSAR_mm"
        ].to_numpy()

        rows.append(
            {
                "Class_ID": int(class_id),
                "Class": CLASS_NAMES.get(
                    int(class_id),
                    f"Class {class_id}"
                ),
                "Cells": int(
                    len(subset)
                ),
                "Area_km2": float(
                    len(subset) * 0.01
                ),
                "Area_percent": float(
                    len(subset)
                    / len(pixel_df)
                    * 100.0
                ),
                "Mean_InSAR_mm": float(
                    np.mean(subset)
                ),
                "Median_InSAR_mm": float(
                    np.median(subset)
                ),
                "Std_InSAR_mm": float(
                    np.std(subset)
                ),
                "Min_InSAR_mm": float(
                    np.min(subset)
                ),
                "Max_InSAR_mm": float(
                    np.max(subset)
                ),
            }
        )

    class_stats = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # MODERATE-HIGH COMPARISON
    # --------------------------------------------------------

    comparison = None

    stats_by_class = (
        class_stats
        .set_index("Class")
    )

    if (
        "Moderate" in stats_by_class.index
        and
        "High" in stats_by_class.index
    ):

        moderate = stats_by_class.loc[
            "Moderate"
        ]

        high = stats_by_class.loc[
            "High"
        ]

        ranges_overlap = not (
            moderate["Max_InSAR_mm"]
            < high["Min_InSAR_mm"]
            or
            high["Max_InSAR_mm"]
            < moderate["Min_InSAR_mm"]
        )

        comparison = {
            "mean_difference_mm": float(
                high["Mean_InSAR_mm"]
                -
                moderate["Mean_InSAR_mm"]
            ),
            "median_difference_mm": float(
                high["Median_InSAR_mm"]
                -
                moderate["Median_InSAR_mm"]
            ),
            "ranges_overlap": bool(
                ranges_overlap
            ),
        }


    return {
        "year": int(year),

        "paths": {
            "svi": svi_path,
            "rank": rank_path,
            "insar": insar_path,
            "common_mask": mask_path,
        },

        "common_cells": int(
            len(pixel_df)
        ),

        "common_area_km2": float(
            len(pixel_df) * 0.01
        ),

        "svi_min": float(
            np.min(svi_values)
        ),

        "svi_max": float(
            np.max(svi_values)
        ),

        "svi_mean": float(
            np.mean(svi_values)
        ),

        "insar_min_mm": float(
            np.min(insar_values)
        ),

        "insar_max_mm": float(
            np.max(insar_values)
        ),

        "insar_mean_mm": float(
            np.mean(insar_values)
        ),

        "csvi_min": float(
            np.min(csvi_values)
        ),

        "csvi_max": float(
            np.max(csvi_values)
        ),

        "csvi_mean": float(
            np.mean(csvi_values)
        ),

        "pearson_r": float(
            pearson_r
        ),

        "pearson_p": float(
            pearson_p
        ),

        "pearson_r2": pearson_r2,

        "pearson_csvi_r": float(
            pearson_csvi_r
        ),

        "pearson_csvi_p": float(
            pearson_csvi_p
        ),

        "spearman_rho": float(
            spearman_rho
        ),

        "spearman_p": float(
            spearman_p
        ),

        "pixel_df": pixel_df,

        "class_stats": class_stats,

        "moderate_high_comparison": comparison,
    }
