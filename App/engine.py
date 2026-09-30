
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from scipy.spatial import cKDTree
from rasterio.warp import reproject, Resampling


# ============================================================
# GLOBAL PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

T_POWER = 1.5
T_NEIGHBORS = 40
T_WEIGHT = 5.0


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================

def classify_t(arr):
    """
    Convert continuous annual groundwater decline T (m/year)
    to ALPRIFT T rate.

    Positive T = groundwater decline
    Negative T = recovery/rise
    """

    x = np.asarray(arr, dtype=float)

    result = np.full(
        x.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(x)

    result[valid & (x < 0.2)] = 1

    result[
        valid &
        (x >= 0.2) &
        (x < 0.5)
    ] = 2

    result[
        valid &
        (x >= 0.5) &
        (x < 0.9)
    ] = 3

    result[
        valid &
        (x >= 0.9) &
        (x < 1.4)
    ] = 4

    result[
        valid &
        (x >= 1.4) &
        (x < 2.0)
    ] = 5

    result[
        valid &
        (x >= 2.0) &
        (x < 2.7)
    ] = 6

    result[
        valid &
        (x >= 2.7) &
        (x < 3.5)
    ] = 7

    result[
        valid &
        (x >= 3.5) &
        (x < 4.4)
    ] = 8

    result[
        valid &
        (x >= 4.4) &
        (x < 5.4)
    ] = 9

    result[
        valid &
        (x >= 5.4)
    ] = 10

    return result


def classify_svi(arr):
    """
    Convert continuous SVI to four ALPRIFT vulnerability classes.

    1 = Low
    2 = Moderate
    3 = High
    4 = Very High
    """

    x = np.asarray(arr, dtype=float)

    result = np.zeros(
        x.shape,
        dtype=np.uint8
    )

    valid = np.isfinite(x)

    result[
        valid &
        (x >= 24) &
        (x < 78)
    ] = 1

    result[
        valid &
        (x >= 78) &
        (x < 133)
    ] = 2

    result[
        valid &
        (x >= 133) &
        (x < 186)
    ] = 3

    result[
        valid &
        (x >= 186) &
        (x <= 240)
    ] = 4

    return result


# ============================================================
# IDW
# ============================================================

def _idw_interpolation(
    xy,
    values,
    transform,
    height,
    width,
    power=T_POWER,
    neighbors=T_NEIGHBORS
):
    """
    IDW interpolation reproducing the validated ArcMap result.
    """

    rows, cols = np.indices(
        (height, width)
    )

    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center"
    )

    grid_xy = np.column_stack([
        np.asarray(xs).ravel(),
        np.asarray(ys).ravel()
    ])

    tree = cKDTree(xy)

    k = min(
        int(neighbors),
        len(values)
    )

    distances, indexes = tree.query(
        grid_xy,
        k=k
    )

    if k == 1:
        distances = distances[:, None]
        indexes = indexes[:, None]

    neighbor_values = values[indexes]

    result = np.empty(
        len(grid_xy),
        dtype=np.float64
    )

    exact = distances[:, 0] == 0

    result[exact] = (
        neighbor_values[exact, 0]
    )

    not_exact = ~exact

    d = distances[not_exact]
    v = neighbor_values[not_exact]

    weights = (
        1.0 /
        np.power(d, power)
    )

    result[not_exact] = (
        np.sum(weights * v, axis=1) /
        np.sum(weights, axis=1)
    )

    return result.reshape(
        height,
        width
    )


# ============================================================
# RASTER HELPERS
# ============================================================

def _valid_mask(arr, nodata):

    mask = np.isfinite(arr)

    if nodata is not None:
        mask &= arr != nodata

    return mask


def _save_float_raster(path, array, profile):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_profile = profile.copy()

    out_profile.update(
        dtype="float32",
        nodata=-9999.0,
        count=1,
        compress="lzw"
    )

    output = np.where(
        np.isfinite(array),
        array,
        -9999.0
    ).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        **out_profile
    ) as dst:

        dst.write(
            output,
            1
        )


# ============================================================
# MAIN T SCENARIO ENGINE
# ============================================================

def run_t_scenario(
    year,
    piezometer,
    new_t,
    save_outputs=True,
    root=ROOT
):
    """
    Run one groundwater-decline scenario.

    Parameters
    ----------
    year : int
        Persian year, e.g. 1400.

    piezometer : str
        Piezometer identifier, e.g. "P04".

    new_t : float
        Scenario annual T value in m/year.
        Positive = decline.
        Negative = recovery.

    save_outputs : bool
        Save scenario rasters/tables when True.

    root : pathlib.Path or str
        DSS root directory.

    Returns
    -------
    dict
        summary, transitions, scenario arrays, and output folder.
    """

    root = Path(root)

    year = int(year)

    # ========================================================
    # MULTI-PIEZOMETER T SUPPORT
    # ========================================================
    #
    # Backward-compatible inputs:
    #
    # Single:
    #   piezometer="P04", new_t=2.5
    #
    # Multi:
    #   piezometer={
    #       "P04": 2.5,
    #       "P06": 1.2,
    #       "P12": -0.2,
    #   },
    #   new_t=None
    # ========================================================

    multi_t_mode = isinstance(
        piezometer,
        dict
    )

    if multi_t_mode:

        if not piezometer:
            raise ValueError(
                "At least one piezometer change is required."
            )

        normalized_t_changes = {}

        for piezo_id, scenario_value in piezometer.items():

            piezo_id = str(
                piezo_id
            )

            scenario_value = float(
                scenario_value
            )

            if not np.isfinite(
                scenario_value
            ):
                raise ValueError(
                    f"Scenario T must be finite for {piezo_id}."
                )

            normalized_t_changes[
                piezo_id
            ] = scenario_value

    else:

        piezo_id = str(
            piezometer
        )

        scenario_value = float(
            new_t
        )

        if not np.isfinite(
            scenario_value
        ):
            raise ValueError(
                "new_t must be a finite numeric value."
            )

        normalized_t_changes = {
            piezo_id:
                scenario_value
        }


    # Primary point retained only for backward-compatible
    # summary fields used by the existing dashboard.
    primary_piezometer = next(
        iter(
            normalized_t_changes
        )
    )

    primary_new_t = float(
        normalized_t_changes[
            primary_piezometer
        ]
    )

    piezometer = (
        primary_piezometer
    )

    new_t = (
        primary_new_t
    )


    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    points_path = (
        root /
        "SourceData" /
        "Piezometers" /
        f"T_{year}_Points.shp"
    )

    baseline_t_rate_path = (
        root /
        "Baseline" /
        "T" /
        f"T_{year}_Rate.tif"
    )

    baseline_svi_path = (
        root /
        "Baseline" /
        "SVI" /
        f"SVI_{year}_Basic.tif"
    )

    baseline_rank_path = (
        root /
        "Baseline" /
        "Rank" /
        f"SVI_{year}_Rank.tif"
    )


    required = [
        points_path,
        baseline_t_rate_path,
        baseline_svi_path,
        baseline_rank_path
    ]

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )


    # --------------------------------------------------------
    # READ PIEZOMETERS
    # --------------------------------------------------------

    gdf = gpd.read_file(
        points_path
    )

    required_fields = {
        "Piezometer",
        "T_myr"
    }

    missing_fields = (
        required_fields -
        set(gdf.columns)
    )

    if missing_fields:

        raise ValueError(
            "Missing fields: "
            + ", ".join(
                sorted(missing_fields)
            )
        )


    available_piezometers = set(
        gdf[
            "Piezometer"
        ].astype(str)
    )

    missing_piezometers = [
        piezo_id
        for piezo_id
        in normalized_t_changes
        if piezo_id
        not in available_piezometers
    ]

    if missing_piezometers:

        raise ValueError(
            "Piezometer(s) not found: "
            + ", ".join(
                missing_piezometers
            )
        )


    if gdf.crs is None:

        raise ValueError(
            "Piezometer CRS is undefined."
        )


    if gdf.crs.to_epsg() != 32638:

        gdf = gdf.to_crs(
            EPSG=32638
        )


    gdf["Piezometer"] = (
        gdf["Piezometer"]
        .astype(str)
    )


    baseline_t_by_piezometer = {}

    for piezo_id in normalized_t_changes:

        baseline_t_by_piezometer[
            piezo_id
        ] = float(
            gdf.loc[
                gdf[
                    "Piezometer"
                ]
                ==
                piezo_id,
                "T_myr"
            ].iloc[0]
        )


    # Backward-compatible primary baseline value
    current_t = float(
        baseline_t_by_piezometer[
            primary_piezometer
        ]
    )


    xy = np.column_stack([
        gdf.geometry.x.to_numpy(),
        gdf.geometry.y.to_numpy()
    ])


    baseline_values = (
        gdf["T_myr"]
        .to_numpy(dtype=float)
    )


    scenario_gdf = (
        gdf.copy()
    )


    # IMPORTANT:
    # Apply every selected piezometer change FIRST.
    # IDW is executed only afterwards.
    for (
        piezo_id,
        scenario_value
    ) in normalized_t_changes.items():

        scenario_gdf.loc[
            scenario_gdf[
                "Piezometer"
            ]
            ==
            piezo_id,
            "T_myr"
        ] = scenario_value


    scenario_values = (
        scenario_gdf["T_myr"]
        .to_numpy(dtype=float)
    )


    # --------------------------------------------------------
    # READ T GRID    # --------------------------------------------------------
    # READ T GRID
    #
    # We use the official annual T-rate raster only as the
    # validated grid/mask. Continuous baseline T is rebuilt
    # from the observed piezometers.
    # --------------------------------------------------------

    with rasterio.open(
        baseline_t_rate_path
    ) as src:

        official_t_rate = (
            src.read(1)
            .astype(np.float64)
        )

        t_profile = src.profile.copy()

        t_transform = src.transform
        t_crs = src.crs

        t_height = src.height
        t_width = src.width

        t_mask = _valid_mask(
            official_t_rate,
            src.nodata
        )


    # --------------------------------------------------------
    # BASELINE + SCENARIO IDW
    # --------------------------------------------------------

    baseline_idw = _idw_interpolation(
        xy=xy,
        values=baseline_values,
        transform=t_transform,
        height=t_height,
        width=t_width,
        power=T_POWER,
        neighbors=T_NEIGHBORS
    )


    scenario_idw = _idw_interpolation(
        xy=xy,
        values=scenario_values,
        transform=t_transform,
        height=t_height,
        width=t_width,
        power=T_POWER,
        neighbors=T_NEIGHBORS
    )


    baseline_idw[
        ~t_mask
    ] = np.nan

    scenario_idw[
        ~t_mask
    ] = np.nan


    # --------------------------------------------------------
    # T RATE
    # --------------------------------------------------------

    baseline_t_rate = classify_t(
        baseline_idw
    )

    scenario_t_rate = classify_t(
        scenario_idw
    )

    delta_t_rate = (
        scenario_t_rate -
        baseline_t_rate
    )


    changed_t_rate_cells = int(
        np.count_nonzero(
            np.isfinite(
                delta_t_rate
            )
            &
            (
                delta_t_rate != 0
            )
        )
    )


    # --------------------------------------------------------
    # READ OFFICIAL BASELINE SVI
    # --------------------------------------------------------

    with rasterio.open(
        baseline_svi_path
    ) as src:

        baseline_svi = (
            src.read(1)
            .astype(np.float64)
        )

        svi_profile = src.profile.copy()

        svi_transform = src.transform
        svi_crs = src.crs

        svi_height = src.height
        svi_width = src.width

        svi_mask = _valid_mask(
            baseline_svi,
            src.nodata
        )


    # --------------------------------------------------------
    # ALIGN DELTA-T-RATE TO OFFICIAL SVI GRID
    # --------------------------------------------------------

    delta_aligned = np.full(
        (svi_height, svi_width),
        -9999.0,
        dtype=np.float32
    )


    delta_source = np.where(
        np.isfinite(
            delta_t_rate
        ),
        delta_t_rate,
        -9999.0
    ).astype(np.float32)


    reproject(
        source=delta_source,
        destination=delta_aligned,

        src_transform=t_transform,
        src_crs=t_crs,
        src_nodata=-9999.0,

        dst_transform=svi_transform,
        dst_crs=svi_crs,
        dst_nodata=-9999.0,

        resampling=Resampling.nearest
    )


    delta_valid = (
        delta_aligned != -9999.0
    )


    final_mask = (
        svi_mask &
        delta_valid
    )


    # --------------------------------------------------------
    # OFFICIAL BASELINE + DELTA ARCHITECTURE
    # --------------------------------------------------------

    scenario_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    scenario_svi[
        final_mask
    ] = (

        baseline_svi[
            final_mask
        ]

        +

        T_WEIGHT *
        delta_aligned[
            final_mask
        ]
    )


    delta_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    delta_svi[
        final_mask
    ] = (

        scenario_svi[
            final_mask
        ]

        -

        baseline_svi[
            final_mask
        ]
    )


    scenario_rank = classify_svi(
        scenario_svi
    )


    # --------------------------------------------------------
    # BASELINE RANK
    # --------------------------------------------------------

    with rasterio.open(
        baseline_rank_path
    ) as src:

        baseline_rank = (
            src.read(1)
        )

        baseline_rank_mask = (
            _valid_mask(
                baseline_rank,
                src.nodata
            )
            &
            (baseline_rank > 0)
        )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    cell_area_km2 = (

        abs(
            svi_transform.a
        )

        *

        abs(
            svi_transform.e
        )

        /

        1_000_000.0
    )


    changed_svi_mask = (

        final_mask

        &

        (
            np.abs(
                delta_svi
            ) > 0
        )
    )


    changed_svi_cells = int(
        np.count_nonzero(
            changed_svi_mask
        )
    )


    changed_area_km2 = (

        changed_svi_cells

        *

        cell_area_km2
    )


    baseline_mean_svi = float(
        np.mean(
            baseline_svi[
                svi_mask
            ]
        )
    )


    scenario_mean_svi = float(
        np.mean(
            scenario_svi[
                final_mask
            ]
        )
    )


    summary = {
        "year": year,

        "piezometer": piezometer,

        "baseline_t": current_t,

        "scenario_t": new_t,

        "t_change": (
            new_t -
            current_t
        ),

        "t_change_count":
            int(
                len(
                    normalized_t_changes
                )
            ),

        "t_changes":
            {
                key: float(value)
                for key, value
                in normalized_t_changes.items()
            },

        "baseline_t_by_piezometer":
            {
                key: float(value)
                for key, value
                in baseline_t_by_piezometer.items()
            },

        "t_weight": T_WEIGHT,

        "changed_t_rate_cells":
            changed_t_rate_cells,

        "changed_svi_cells":
            changed_svi_cells,

        "changed_area_km2":
            float(
                changed_area_km2
            ),

        "baseline_mean_svi":
            baseline_mean_svi,

        "scenario_mean_svi":
            scenario_mean_svi,

        "mean_delta_svi":
            float(
                np.nanmean(
                    delta_svi
                )
            ),

        "max_delta_svi":
            float(
                np.nanmax(
                    delta_svi
                )
            ),

        "min_delta_svi":
            float(
                np.nanmin(
                    delta_svi
                )
            )
    }


    # --------------------------------------------------------
    # CLASS TRANSITIONS
    # --------------------------------------------------------

    transition_mask = (

        final_mask

        &

        baseline_rank_mask

        &

        (scenario_rank > 0)
    )


    transitions = []


    for old_class in range(
        1,
        5
    ):

        for new_class in range(
            1,
            5
        ):

            count = int(
                np.count_nonzero(

                    transition_mask

                    &

                    (
                        baseline_rank
                        ==
                        old_class
                    )

                    &

                    (
                        scenario_rank
                        ==
                        new_class
                    )
                )
            )


            if count > 0:

                transitions.append({
                    "From_Class":
                        old_class,

                    "To_Class":
                        new_class,

                    "Cells":
                        count,

                    "Area_km2":
                        count *
                        cell_area_km2
                })


    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "From_Class",
            "To_Class",
            "Cells",
            "Area_km2"
        ]
    )


    # --------------------------------------------------------
    # OPTIONAL OUTPUT FILES
    # --------------------------------------------------------

    scenario_folder = None


    if save_outputs:

        if multi_t_mode:

            selected_ids_text = "_".join(
                list(
                    normalized_t_changes.keys()
                )[:5]
            )

            scenario_name = (
                f"T_{year}_MULTI"
                f"{len(normalized_t_changes)}_"
                f"{selected_ids_text}"
            )

        else:

            scenario_name = (

                f"T_{year}_"
                f"{piezometer}_to_"
                f"{new_t:.2f}"

            ).replace(
                ".",
                "p"
            )


        scenario_folder = (

            root /
            "Scenarios" /
            "T" /
            scenario_name
        )


        scenario_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        # --------------------------------------------
        # Save changed point table
        # --------------------------------------------

        point_table = (
            scenario_gdf
            .drop(
                columns="geometry"
            )
        )

        point_table.to_csv(
            scenario_folder /
            "Scenario_Piezometers.csv",

            index=False,
            encoding="utf-8-sig"
        )


        # --------------------------------------------
        # Raster outputs
        # --------------------------------------------

        _save_float_raster(
            scenario_folder /
            "T_IDW_Scenario.tif",

            scenario_idw,

            t_profile
        )


        _save_float_raster(
            scenario_folder /
            "T_Rate_Scenario.tif",

            scenario_t_rate,

            t_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_T_Rate.tif",

            delta_t_rate,

            t_profile
        )


        _save_float_raster(
            scenario_folder /
            "SVI_Scenario.tif",

            scenario_svi,

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_SVI.tif",

            delta_svi,

            svi_profile
        )


        # --------------------------------------------
        # Save scenario rank
        # --------------------------------------------

        rank_profile = (
            svi_profile.copy()
        )

        rank_profile.update(
            dtype="uint8",
            nodata=0,
            count=1,
            compress="lzw"
        )


        with rasterio.open(
            scenario_folder /
            "SVI_Scenario_Rank.tif",
            "w",
            **rank_profile
        ) as dst:

            dst.write(
                scenario_rank,
                1
            )


        # --------------------------------------------
        # Save summary
        # --------------------------------------------

        summary_df = pd.DataFrame({
            "Parameter":
                list(
                    summary.keys()
                ),

            "Value":
                list(
                    summary.values()
                )
        })


        summary_df.to_csv(
            scenario_folder /
            "Scenario_Summary.csv",

            index=False,
            encoding="utf-8-sig"
        )


        transitions_df.to_csv(
            scenario_folder /
            "Class_Transitions.csv",

            index=False,
            encoding="utf-8-sig"
        )


    # --------------------------------------------------------
    # RETURN RESULT FOR STREAMLIT / TESTS
    # --------------------------------------------------------

    return {

        "summary":
            summary,

        "transitions":
            transitions_df,

        "scenario_idw":
            scenario_idw,

        "baseline_t_rate":
            baseline_t_rate,

        "scenario_t_rate":
            scenario_t_rate,

        "delta_t_rate":
            delta_t_rate,

        "baseline_svi":
            baseline_svi,

        "scenario_svi":
            scenario_svi,

        "delta_svi":
            delta_svi,

        "baseline_rank":
            baseline_rank,

        "scenario_rank":
            scenario_rank,

        "svi_mask":
            svi_mask,

        "final_mask":
            final_mask,

        "svi_profile":
            svi_profile,

        "scenario_folder":
            scenario_folder
    }



# ============================================================
# PUMPING (P) MODULE
# ============================================================

from rasterio.features import rasterize


P_WEIGHT = 4.0


def classify_p(arr):
    """
    Convert physical pumping intensity P (cm/year)
    to ALPRIFT P rate.

    Current implementation convention:
        P < 0.0001          -> 1
        0.0001 <= P < 0.005 -> 2
        0.005 <= P < 0.01   -> 3
        0.01 <= P < 0.5     -> 4
        0.5 <= P < 1        -> 5
        1 <= P < 5          -> 6
        5 <= P < 20         -> 7
        20 <= P < 40        -> 8
        40 <= P <= 65       -> 9
        P > 65              -> 10
    """

    x = np.asarray(arr, dtype=float)

    result = np.full(
        x.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(x)

    result[
        valid &
        (x < 0.0001)
    ] = 1

    result[
        valid &
        (x >= 0.0001) &
        (x < 0.005)
    ] = 2

    result[
        valid &
        (x >= 0.005) &
        (x < 0.01)
    ] = 3

    result[
        valid &
        (x >= 0.01) &
        (x < 0.5)
    ] = 4

    result[
        valid &
        (x >= 0.5) &
        (x < 1.0)
    ] = 5

    result[
        valid &
        (x >= 1.0) &
        (x < 5.0)
    ] = 6

    result[
        valid &
        (x >= 5.0) &
        (x < 20.0)
    ] = 7

    result[
        valid &
        (x >= 20.0) &
        (x < 40.0)
    ] = 8

    result[
        valid &
        (x >= 40.0) &
        (x <= 65.0)
    ] = 9

    result[
        valid &
        (x > 65.0)
    ] = 10

    return result


def run_p_scenario(
    zone_id,
    scenario_p,
    year=1400,
    save_outputs=True,
    root=ROOT
):
    """
    Run a direct groundwater-pumping scenario.

    P is entered as the final physical pumping value
    in cm/year.

    No percentage change is used.

    Single-zone example
    -------------------
    zone_id = 6
    scenario_p = 12.5

    Multi-zone example
    ------------------
    zone_id = {
        0: 18.0,
        4: 0.8,
        14: 35.0,
    }
    scenario_p = None

    All selected zone values are applied first.
    One P-rating raster is then generated.
    """

    root = Path(root)

    year = int(year)

    # ========================================================
    # DIRECT MULTI-PUMPING-ZONE SUPPORT
    # ========================================================
    #
    # The scenario value is the final physical P value
    # in cm/year.
    #
    # Single:
    #   zone_id = 6
    #   scenario_p = 12.5
    #
    # Multi:
    #   zone_id = {
    #       0: 18.0,
    #       4: 0.8,
    #       14: 35.0,
    #   }
    #   scenario_p = None
    # ========================================================

    p_multi_mode = isinstance(
        zone_id,
        dict
    )


    if p_multi_mode:

        if not zone_id:

            raise ValueError(
                "At least one pumping zone is required."
            )


        normalized_p_scenarios = {}


        for (
            selected_zone,
            selected_scenario_p
        ) in zone_id.items():

            selected_zone = int(
                selected_zone
            )

            selected_scenario_p = float(
                selected_scenario_p
            )


            if (
                not np.isfinite(
                    selected_scenario_p
                )
                or
                selected_scenario_p < 0
            ):

                raise ValueError(
                    f"Scenario P must be finite and "
                    f"non-negative for zone "
                    f"{selected_zone}."
                )


            normalized_p_scenarios[
                selected_zone
            ] = selected_scenario_p


    else:

        selected_zone = int(
            zone_id
        )

        selected_scenario_p = float(
            scenario_p
        )


        if (
            not np.isfinite(
                selected_scenario_p
            )
            or
            selected_scenario_p < 0
        ):

            raise ValueError(
                "scenario_p must be finite "
                "and non-negative."
            )


        normalized_p_scenarios = {
            selected_zone:
                selected_scenario_p
        }


    primary_zone_id = next(
        iter(
            normalized_p_scenarios
        )
    )


    primary_scenario_p = float(
        normalized_p_scenarios[
            primary_zone_id
        ]
    )


    # Primary fields retained for compatibility
    zone_id = int(
        primary_zone_id
    )

    scenario_p = float(
        primary_scenario_p
    )


    # ========================================================
    # PATHS
    # ========================================================

    pumping_path = (
        root /
        "SourceData" /
        "Pumping" /
        "P_Thi_1398_Final3.shp"
    )

    baseline_p_path = (
        root /
        "Baseline" /
        "P" /
        "P_Baseline.tif"
    )

    baseline_svi_path = (
        root /
        "Baseline" /
        "SVI" /
        f"SVI_{year}_Basic.tif"
    )

    baseline_rank_path = (
        root /
        "Baseline" /
        "Rank" /
        f"SVI_{year}_Rank.tif"
    )


    required = [
        pumping_path,
        baseline_p_path,
        baseline_svi_path,
        baseline_rank_path
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                str(path)
            )


    # ========================================================
    # READ PUMPING POLYGONS
    # ========================================================

    gdf = gpd.read_file(
        pumping_path
    )

    required_fields = {
        "TARGET_FID",
        "takhlieh_s",
        "Area",
        "pompaj",
        "P_cm_yr",
        "P_Rate"
    }

    missing = (
        required_fields -
        set(gdf.columns)
    )

    if missing:

        raise ValueError(
            "Missing pumping fields: "
            + ", ".join(
                sorted(missing)
            )
        )


    available_zones = set(
        gdf[
            "TARGET_FID"
        ].astype(int)
    )

    missing_zones = [
        selected_zone
        for selected_zone
        in normalized_p_scenarios
        if selected_zone
        not in available_zones
    ]

    if missing_zones:

        raise ValueError(
            "Pumping zone(s) not found: "
            + ", ".join(
                str(value)
                for value
                in missing_zones
            )
        )


    # ========================================================
    # BASELINE ZONE VALUES + MULTI-ZONE SCENARIO
    # ========================================================

    scenario_gdf = gdf.copy()

    pumping_zone_details = {}


    for (
        selected_zone,
        selected_scenario_p
    ) in normalized_p_scenarios.items():


        row_mask = (
            scenario_gdf[
                "TARGET_FID"
            ].astype(int)
            ==
            int(
                selected_zone
            )
        )


        baseline_p_cm = float(
            gdf.loc[
                row_mask,
                "P_cm_yr"
            ].iloc[0]
        )


        baseline_rate = int(
            gdf.loc[
                row_mask,
                "P_Rate"
            ].iloc[0]
        )


        baseline_takhlieh_value = float(
            gdf.loc[
                row_mask,
                "takhlieh_s"
            ].iloc[0]
        )


        baseline_pompaj_value = float(
            gdf.loc[
                row_mask,
                "pompaj"
            ].iloc[0]
        )


        # ----------------------------------------------
        # DIRECT SCENARIO P
        # ----------------------------------------------

        scenario_p_cm = float(
            selected_scenario_p
        )


        scenario_rate = int(
            classify_p(
                np.array(
                    [
                        scenario_p_cm
                    ]
                )
            )[0]
        )


        # Only P_cm_yr and P_Rate are changed.
        #
        # takhlieh_s and pompaj remain the original
        # baseline/source-data attributes.

        scenario_gdf.loc[
            row_mask,
            "P_cm_yr"
        ] = scenario_p_cm


        scenario_gdf.loc[
            row_mask,
            "P_Rate"
        ] = scenario_rate


        pumping_zone_details[
            int(
                selected_zone
            )
        ] = {

            "baseline_p_cm_yr":
                baseline_p_cm,

            "scenario_p_cm_yr":
                scenario_p_cm,

            "baseline_p_rate":
                baseline_rate,

            "scenario_p_rate":
                scenario_rate,

            "p_rate_change":
                int(
                    scenario_rate
                    -
                    baseline_rate
                ),

            "baseline_takhlieh_s":
                baseline_takhlieh_value,

            "baseline_pompaj":
                baseline_pompaj_value,
        }


    # ========================================================
    # PRIMARY ZONE SUMMARY FIELDS
    # ========================================================

    primary_detail = (
        pumping_zone_details[
            int(
                primary_zone_id
            )
        ]
    )


    baseline_p_cm_yr = float(
        primary_detail[
            "baseline_p_cm_yr"
        ]
    )


    scenario_p_cm_yr = float(
        primary_detail[
            "scenario_p_cm_yr"
        ]
    )


    baseline_p_rate = int(
        primary_detail[
            "baseline_p_rate"
        ]
    )


    scenario_p_rate = int(
        primary_detail[
            "scenario_p_rate"
        ]
    )


    p_rate_change = int(
        primary_detail[
            "p_rate_change"
        ]
    )


    baseline_takhlieh = float(
        primary_detail[
            "baseline_takhlieh_s"
        ]
    )


    baseline_pompaj = float(
        primary_detail[
            "baseline_pompaj"
        ]
    )


    # ========================================================
    # READ OFFICIAL P GRID    # ========================================================
    # READ OFFICIAL P GRID
    # ========================================================

    with rasterio.open(
        baseline_p_path
    ) as src:

        baseline_p_raster = (
            src.read(1)
            .astype(np.float64)
        )

        p_profile = src.profile.copy()

        p_transform = src.transform
        p_crs = src.crs

        p_height = src.height
        p_width = src.width

        p_mask = _valid_mask(
            baseline_p_raster,
            src.nodata
        )


    if scenario_gdf.crs is None:

        raise ValueError(
            "Pumping polygon CRS is undefined."
        )


    if scenario_gdf.crs != p_crs:

        scenario_gdf = (
            scenario_gdf
            .to_crs(p_crs)
        )


    # ========================================================
    # RASTERIZE SCENARIO P RATE
    # ========================================================

    shapes = [
        (
            geom,
            float(rate)
        )

        for geom, rate in zip(
            scenario_gdf.geometry,
            scenario_gdf["P_Rate"]
        )

        if geom is not None
        and not geom.is_empty
    ]


    scenario_p_raster = rasterize(
        shapes=shapes,

        out_shape=(
            p_height,
            p_width
        ),

        transform=p_transform,

        fill=-9999.0,

        dtype="float32",

        all_touched=False
    )


    scenario_p_valid = (
        scenario_p_raster
        != -9999.0
    )


    # ========================================================
    # DELTA P RATE
    #
    # Use the OFFICIAL baseline raster rather than a rebuilt
    # baseline. Therefore zero scenario change gives zero delta.
    # ========================================================

    delta_p_rate = np.full(
        baseline_p_raster.shape,
        np.nan,
        dtype=np.float64
    )


    p_common = (
        p_mask &
        scenario_p_valid
    )


    delta_p_rate[
        p_common
    ] = (

        scenario_p_raster[
            p_common
        ]

        -

        baseline_p_raster[
            p_common
        ]
    )


    changed_p_rate_cells = int(
        np.count_nonzero(
            np.isfinite(
                delta_p_rate
            )
            &
            (
                delta_p_rate != 0
            )
        )
    )


    # ========================================================
    # READ OFFICIAL SVI BASELINE
    # ========================================================

    with rasterio.open(
        baseline_svi_path
    ) as src:

        baseline_svi = (
            src.read(1)
            .astype(np.float64)
        )

        svi_profile = src.profile.copy()

        svi_transform = src.transform
        svi_crs = src.crs

        svi_height = src.height
        svi_width = src.width

        svi_mask = _valid_mask(
            baseline_svi,
            src.nodata
        )


    # ========================================================
    # ALIGN DELTA P TO SVI GRID
    # ========================================================

    delta_aligned = np.full(
        (
            svi_height,
            svi_width
        ),
        -9999.0,
        dtype=np.float32
    )


    delta_source = np.where(
        np.isfinite(
            delta_p_rate
        ),
        delta_p_rate,
        -9999.0
    ).astype(np.float32)


    reproject(
        source=delta_source,
        destination=delta_aligned,

        src_transform=p_transform,
        src_crs=p_crs,
        src_nodata=-9999.0,

        dst_transform=svi_transform,
        dst_crs=svi_crs,
        dst_nodata=-9999.0,

        resampling=Resampling.nearest
    )


    delta_valid = (
        delta_aligned
        != -9999.0
    )


    final_mask = (
        svi_mask &
        delta_valid
    )


    # ========================================================
    # SCENARIO SVI
    # ========================================================

    scenario_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    scenario_svi[
        final_mask
    ] = (

        baseline_svi[
            final_mask
        ]

        +

        P_WEIGHT *
        delta_aligned[
            final_mask
        ]
    )


    delta_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    delta_svi[
        final_mask
    ] = (

        scenario_svi[
            final_mask
        ]

        -

        baseline_svi[
            final_mask
        ]
    )


    scenario_rank = classify_svi(
        scenario_svi
    )


    # ========================================================
    # BASELINE RANK
    # ========================================================

    with rasterio.open(
        baseline_rank_path
    ) as src:

        baseline_rank = src.read(1)

        baseline_rank_mask = (
            _valid_mask(
                baseline_rank,
                src.nodata
            )
            &
            (
                baseline_rank > 0
            )
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    cell_area_km2 = (

        abs(
            svi_transform.a
        )

        *

        abs(
            svi_transform.e
        )

        /

        1_000_000.0
    )


    changed_svi_mask = (

        final_mask

        &

        np.isfinite(
            delta_svi
        )

        &

        (
            delta_svi != 0
        )
    )


    changed_svi_cells = int(
        np.count_nonzero(
            changed_svi_mask
        )
    )


    changed_area_km2 = float(
        changed_svi_cells
        *
        cell_area_km2
    )


    summary = {

        "year":
            year,

        "zone_id":
            zone_id,

        "baseline_p_cm_yr":
            baseline_p_cm_yr,

        "scenario_p_cm_yr":
            scenario_p_cm_yr,

        "baseline_p_rate":
            baseline_p_rate,

        "scenario_p_rate":
            scenario_p_rate,

        "p_rate_change":
            p_rate_change,

        "baseline_takhlieh_s":
            baseline_takhlieh,

        "baseline_pompaj":
            baseline_pompaj,

        "zone_count":
            int(
                len(
                    normalized_p_scenarios
                )
            ),

        "zone_scenarios":
            {
                int(key):
                    float(value)

                for key, value
                in normalized_p_scenarios.items()
            },

        "zone_details":
            pumping_zone_details,

        "p_weight":
            P_WEIGHT,

        "changed_p_rate_cells":
            changed_p_rate_cells,

        "changed_svi_cells":
            changed_svi_cells,

        "changed_area_km2":
            changed_area_km2,

        "baseline_mean_svi":
            float(
                np.mean(
                    baseline_svi[
                        svi_mask
                    ]
                )
            ),

        "scenario_mean_svi":
            float(
                np.mean(
                    scenario_svi[
                        final_mask
                    ]
                )
            ),

        "mean_delta_svi":
            float(
                np.nanmean(
                    delta_svi
                )
            ),

        "max_delta_svi":
            float(
                np.nanmax(
                    delta_svi
                )
            ),

        "min_delta_svi":
            float(
                np.nanmin(
                    delta_svi
                )
            )
    }


    # ========================================================
    # CLASS TRANSITIONS
    # ========================================================

    transition_mask = (

        final_mask

        &

        baseline_rank_mask

        &

        (
            scenario_rank > 0
        )
    )


    transitions = []


    for old_class in range(
        1,
        5
    ):

        for new_class in range(
            1,
            5
        ):

            count = int(
                np.count_nonzero(

                    transition_mask

                    &

                    (
                        baseline_rank
                        ==
                        old_class
                    )

                    &

                    (
                        scenario_rank
                        ==
                        new_class
                    )
                )
            )


            if count > 0:

                transitions.append({

                    "From_Class":
                        old_class,

                    "To_Class":
                        new_class,

                    "Cells":
                        count,

                    "Area_km2":
                        count *
                        cell_area_km2
                })


    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "From_Class",
            "To_Class",
            "Cells",
            "Area_km2"
        ]
    )


    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    scenario_folder = None


    if save_outputs:
        if p_multi_mode:

            zone_text = "_".join(
                str(value)
                for value
                in list(
                    normalized_p_scenarios.keys()
                )[:5]
            )


            scenario_name = (
                f"P_Y{year}_"
                f"MULTI{len(normalized_p_scenarios)}_"
                f"{zone_text}"
            )


        else:

            scenario_text = (
                f"{scenario_p:.3f}"
                .replace(
                    ".",
                    "p"
                )
            )


            scenario_name = (
                f"P_Y{year}_"
                f"Zone{zone_id}_to_"
                f"{scenario_text}"
            )


        scenario_folder = (

            root /
            "Scenarios" /
            "P" /
            scenario_name
        )


        scenario_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        # --------------------------------------------
        # Save scenario polygon attributes
        # --------------------------------------------

        scenario_gdf.drop(
            columns="geometry"
        ).to_csv(
            scenario_folder /
            "Scenario_Pumping_Zones.csv",

            index=False,
            encoding="utf-8-sig"
        )


        # --------------------------------------------
        # Save P rasters
        # --------------------------------------------

        scenario_p_for_save = np.where(
            scenario_p_valid,
            scenario_p_raster,
            np.nan
        )


        _save_float_raster(
            scenario_folder /
            "P_Rate_Scenario.tif",

            scenario_p_for_save,

            p_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_P_Rate.tif",

            delta_p_rate,

            p_profile
        )


        # --------------------------------------------
        # Save SVI
        # --------------------------------------------

        _save_float_raster(
            scenario_folder /
            "SVI_Scenario.tif",

            scenario_svi,

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_SVI.tif",

            delta_svi,

            svi_profile
        )


        # --------------------------------------------
        # Save rank
        # --------------------------------------------

        rank_profile = (
            svi_profile.copy()
        )

        rank_profile.update(
            dtype="uint8",
            nodata=0,
            count=1,
            compress="lzw"
        )


        with rasterio.open(
            scenario_folder /
            "SVI_Scenario_Rank.tif",
            "w",
            **rank_profile
        ) as dst:

            dst.write(
                scenario_rank,
                1
            )


        # --------------------------------------------
        # Summary
        # --------------------------------------------

        pd.DataFrame({
            "Parameter":
                list(
                    summary.keys()
                ),

            "Value":
                list(
                    summary.values()
                )
        }).to_csv(
            scenario_folder /
            "Scenario_Summary.csv",

            index=False,
            encoding="utf-8-sig"
        )


        transitions_df.to_csv(
            scenario_folder /
            "Class_Transitions.csv",

            index=False,
            encoding="utf-8-sig"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "summary":
            summary,

        "transitions":
            transitions_df,

        "baseline_p_raster":
            baseline_p_raster,

        "scenario_p_raster":
            scenario_p_raster,

        "delta_p_rate":
            delta_p_rate,

        "baseline_svi":
            baseline_svi,

        "scenario_svi":
            scenario_svi,

        "delta_svi":
            delta_svi,

        "baseline_rank":
            baseline_rank,

        "scenario_rank":
            scenario_rank,

        "svi_mask":
            svi_mask,

        "final_mask":
            final_mask,

        "svi_profile":
            svi_profile,

        "scenario_folder":
            scenario_folder
    }



# ============================================================
# RECHARGE / RAINFALL (R) MODULE
# ============================================================

R_WEIGHT = 4.0


def classify_rainfall(arr):
    """
    Piscopo rainfall ranking used in the current R layer.

        Rainfall < 500 mm/year       -> 1
        500 <= Rainfall < 700        -> 2
        700 <= Rainfall <= 850       -> 3
        Rainfall > 850               -> 4
    """

    x = np.asarray(arr, dtype=float)

    result = np.full(
        x.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(x)

    result[
        valid &
        (x < 500.0)
    ] = 1

    result[
        valid &
        (x >= 500.0) &
        (x < 700.0)
    ] = 2

    result[
        valid &
        (x >= 700.0) &
        (x <= 850.0)
    ] = 3

    result[
        valid &
        (x > 850.0)
    ] = 4

    return result


def _piscopo_score_to_recharge(score):
    """
    Convert Piscopo score to recharge (cm/year)
    using the implementation already used for the
    Shabestar R layer.

        score 3-4   -> 1 cm/year
        score 5-6   -> 3 cm/year
        score 7-8   -> 5 cm/year
        score 9-10  -> 8 cm/year
        score 11-13 -> 10 cm/year

    The current baseline occupies scores 5-10,
    but the upper class is retained for rainfall scenarios.
    """

    x = np.asarray(score, dtype=float)

    result = np.full(
        x.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(x)

    result[
        valid &
        (x >= 3) &
        (x < 5)
    ] = 1

    result[
        valid &
        (x >= 5) &
        (x < 7)
    ] = 3

    result[
        valid &
        (x >= 7) &
        (x < 9)
    ] = 5

    result[
        valid &
        (x >= 9) &
        (x < 11)
    ] = 8

    result[
        valid &
        (x >= 11) &
        (x <= 13)
    ] = 10

    return result


def _recharge_to_r_rate(recharge_cm):
    """
    ALPRIFT recharge rating.

        Recharge < 4          -> 10
        4 <= Recharge < 9     -> 9
        9 <= Recharge < 14    -> 7
        14 <= Recharge < 19   -> 5
        19 <= Recharge < 24   -> 3
        Recharge >= 24        -> 1
    """

    x = np.asarray(
        recharge_cm,
        dtype=float
    )

    result = np.full(
        x.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(x)

    result[
        valid &
        (x < 4.0)
    ] = 10

    result[
        valid &
        (x >= 4.0) &
        (x < 9.0)
    ] = 9

    result[
        valid &
        (x >= 9.0) &
        (x < 14.0)
    ] = 7

    result[
        valid &
        (x >= 14.0) &
        (x < 19.0)
    ] = 5

    result[
        valid &
        (x >= 19.0) &
        (x < 24.0)
    ] = 3

    result[
        valid &
        (x >= 24.0)
    ] = 1

    return result


def run_r_scenario(
    rainfall_mm,
    year=1400,
    save_outputs=True,
    root=ROOT
):
    """
    Run an R scenario by changing only annual rainfall.

    Slope Rank and Permeability Rank remain fixed.
    The baseline R layer corresponds to the current
    1398 rainfall/recharge construction.
    """

    root = Path(root)

    rainfall_mm = float(rainfall_mm)
    year = int(year)

    if (
        not np.isfinite(rainfall_mm)
        or rainfall_mm < 0
    ):
        raise ValueError(
            "rainfall_mm must be a finite non-negative value."
        )


    # ========================================================
    # PATHS
    # ========================================================

    recharge_dir = (
        root /
        "SourceData" /
        "Recharge"
    )

    slope_path = (
        recharge_dir /
        "Slope_Rank.tif"
    )

    permeability_path = (
        recharge_dir /
        "Permeability_Rank.tif"
    )

    baseline_rainfall_path = (
        recharge_dir /
        "Rain_1398_mm.tif"
    )

    score_path = (
        recharge_dir /
        "R_Piscopo_Score_1398.tif"
    )

    baseline_r_path = (
        root /
        "Baseline" /
        "R" /
        "R_Baseline.tif"
    )

    baseline_svi_path = (
        root /
        "Baseline" /
        "SVI" /
        f"SVI_{year}_Basic.tif"
    )

    baseline_rank_path = (
        root /
        "Baseline" /
        "Rank" /
        f"SVI_{year}_Rank.tif"
    )


    required = [
        slope_path,
        permeability_path,
        baseline_rainfall_path,
        score_path,
        baseline_r_path,
        baseline_svi_path,
        baseline_rank_path
    ]

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )


    # ========================================================
    # OFFICIAL PISCOPO SCORE GRID
    # ========================================================

    with rasterio.open(
        score_path
    ) as src:

        official_score = (
            src.read(1)
            .astype(np.float64)
        )

        r_profile = src.profile.copy()

        r_transform = src.transform
        r_crs = src.crs

        r_shape = (
            src.height,
            src.width
        )

        score_valid = _valid_mask(
            official_score,
            src.nodata
        )


    # ========================================================
    # PERMEABILITY RANK
    #
    # This is intentionally read by array index.
    # Our reconstruction test proved that this reproduces
    # the official Piscopo score exactly.
    # ========================================================

    with rasterio.open(
        permeability_path
    ) as src:

        permeability = (
            src.read(1)
            .astype(np.float64)
        )

        permeability_valid = (
            _valid_mask(
                permeability,
                src.nodata
            )
        )


    if permeability.shape != r_shape:
        raise ValueError(
            "Permeability raster shape does not match the R grid."
        )


    # ========================================================
    # ALIGN SLOPE RANK TO OFFICIAL R GRID
    # ========================================================

    with rasterio.open(
        slope_path
    ) as src:

        slope_source = (
            src.read(1)
            .astype(np.float32)
        )

        slope_aligned = np.full(
            r_shape,
            -9999.0,
            dtype=np.float32
        )

        reproject(
            source=slope_source,
            destination=slope_aligned,

            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,

            dst_transform=r_transform,
            dst_crs=r_crs,
            dst_nodata=-9999.0,

            resampling=Resampling.nearest
        )


    slope_valid = (
        slope_aligned
        != -9999.0
    )


    # ========================================================
    # BASELINE RAINFALL
    # ========================================================

    with rasterio.open(
        baseline_rainfall_path
    ) as src:

        baseline_rainfall_array = (
            src.read(1)
            .astype(np.float64)
        )

        rainfall_valid = (
            _valid_mask(
                baseline_rainfall_array,
                src.nodata
            )
        )


    baseline_rainfall_mm = float(
        np.mean(
            baseline_rainfall_array[
                rainfall_valid
            ]
        )
    )


    baseline_rain_rank = int(
        classify_rainfall(
            np.array([
                baseline_rainfall_mm
            ])
        )[0]
    )


    scenario_rain_rank = int(
        classify_rainfall(
            np.array([
                rainfall_mm
            ])
        )[0]
    )


    # ========================================================
    # BUILD SCENARIO PISCOPO SCORE
    # ========================================================

    common_r = (
        score_valid
        &
        permeability_valid
        &
        slope_valid
    )


    scenario_score = np.full(
        r_shape,
        np.nan,
        dtype=np.float64
    )


    scenario_score[
        common_r
    ] = (
        slope_aligned[
            common_r
        ]
        +
        permeability[
            common_r
        ]
        +
        scenario_rain_rank
    )


    # ========================================================
    # SCORE -> RECHARGE -> R RATE
    # ========================================================

    scenario_recharge = (
        _piscopo_score_to_recharge(
            scenario_score
        )
    )


    scenario_r_rate = (
        _recharge_to_r_rate(
            scenario_recharge
        )
    )


    # ========================================================
    # OFFICIAL BASELINE R RATE
    # ========================================================

    with rasterio.open(
        baseline_r_path
    ) as src:

        baseline_r_rate = (
            src.read(1)
            .astype(np.float64)
        )

        baseline_r_valid = (
            _valid_mask(
                baseline_r_rate,
                src.nodata
            )
        )

        baseline_r_transform = (
            src.transform
        )

        baseline_r_crs = (
            src.crs
        )

        baseline_r_profile = (
            src.profile.copy()
        )


    if (
        baseline_r_rate.shape
        !=
        r_shape
    ):
        raise ValueError(
            "Official R baseline shape does not match R score grid."
        )


    # ========================================================
    # DELTA R RATE
    # ========================================================

    r_common = (
        common_r
        &
        baseline_r_valid
        &
        np.isfinite(
            scenario_r_rate
        )
    )


    delta_r_rate = np.full(
        r_shape,
        np.nan,
        dtype=np.float64
    )


    delta_r_rate[
        r_common
    ] = (

        scenario_r_rate[
            r_common
        ]

        -

        baseline_r_rate[
            r_common
        ]
    )


    changed_r_rate_cells = int(
        np.count_nonzero(
            np.isfinite(
                delta_r_rate
            )
            &
            (
                delta_r_rate != 0
            )
        )
    )


    # ========================================================
    # READ OFFICIAL SVI BASELINE
    # ========================================================

    with rasterio.open(
        baseline_svi_path
    ) as src:

        baseline_svi = (
            src.read(1)
            .astype(np.float64)
        )

        svi_profile = (
            src.profile.copy()
        )

        svi_transform = (
            src.transform
        )

        svi_crs = (
            src.crs
        )

        svi_shape = (
            src.height,
            src.width
        )

        svi_mask = (
            _valid_mask(
                baseline_svi,
                src.nodata
            )
        )


    # ========================================================
    # ALIGN DELTA R TO OFFICIAL SVI GRID
    # ========================================================

    delta_source = np.where(
        np.isfinite(
            delta_r_rate
        ),
        delta_r_rate,
        -9999.0
    ).astype(np.float32)


    delta_aligned = np.full(
        svi_shape,
        -9999.0,
        dtype=np.float32
    )


    reproject(
        source=delta_source,
        destination=delta_aligned,

        src_transform=baseline_r_transform,
        src_crs=baseline_r_crs,
        src_nodata=-9999.0,

        dst_transform=svi_transform,
        dst_crs=svi_crs,
        dst_nodata=-9999.0,

        resampling=Resampling.nearest
    )


    delta_valid = (
        delta_aligned
        != -9999.0
    )


    final_mask = (
        svi_mask
        &
        delta_valid
    )


    # ========================================================
    # SVI SCENARIO
    # ========================================================

    scenario_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    scenario_svi[
        final_mask
    ] = (

        baseline_svi[
            final_mask
        ]

        +

        R_WEIGHT
        *
        delta_aligned[
            final_mask
        ]
    )


    delta_svi = np.full(
        baseline_svi.shape,
        np.nan,
        dtype=np.float64
    )


    delta_svi[
        final_mask
    ] = (

        scenario_svi[
            final_mask
        ]

        -

        baseline_svi[
            final_mask
        ]
    )


    scenario_rank = (
        classify_svi(
            scenario_svi
        )
    )


    # ========================================================
    # BASELINE RANK
    # ========================================================

    with rasterio.open(
        baseline_rank_path
    ) as src:

        baseline_rank = (
            src.read(1)
        )

        baseline_rank_mask = (
            _valid_mask(
                baseline_rank,
                src.nodata
            )
            &
            (
                baseline_rank > 0
            )
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    cell_area_km2 = (

        abs(
            svi_transform.a
        )

        *

        abs(
            svi_transform.e
        )

        /

        1_000_000.0
    )


    changed_svi_mask = (

        final_mask

        &

        np.isfinite(
            delta_svi
        )

        &

        (
            delta_svi != 0
        )
    )


    changed_svi_cells = int(
        np.count_nonzero(
            changed_svi_mask
        )
    )


    changed_area_km2 = float(
        changed_svi_cells
        *
        cell_area_km2
    )


    summary = {

        "year":
            year,

        "baseline_rainfall_mm":
            baseline_rainfall_mm,

        "scenario_rainfall_mm":
            rainfall_mm,

        "baseline_rain_rank":
            baseline_rain_rank,

        "scenario_rain_rank":
            scenario_rain_rank,

        "r_weight":
            R_WEIGHT,

        "changed_r_rate_cells":
            changed_r_rate_cells,

        "changed_svi_cells":
            changed_svi_cells,

        "changed_area_km2":
            changed_area_km2,

        "baseline_mean_svi":
            float(
                np.mean(
                    baseline_svi[
                        svi_mask
                    ]
                )
            ),

        "scenario_mean_svi":
            float(
                np.mean(
                    scenario_svi[
                        final_mask
                    ]
                )
            ),

        "mean_delta_svi":
            float(
                np.nanmean(
                    delta_svi
                )
            ),

        "max_delta_svi":
            float(
                np.nanmax(
                    delta_svi
                )
            ),

        "min_delta_svi":
            float(
                np.nanmin(
                    delta_svi
                )
            )
    }


    # ========================================================
    # CLASS TRANSITIONS
    # ========================================================

    transition_mask = (

        final_mask

        &

        baseline_rank_mask

        &

        (
            scenario_rank > 0
        )
    )


    transitions = []


    for old_class in range(
        1,
        5
    ):

        for new_class in range(
            1,
            5
        ):

            count = int(
                np.count_nonzero(

                    transition_mask

                    &

                    (
                        baseline_rank
                        ==
                        old_class
                    )

                    &

                    (
                        scenario_rank
                        ==
                        new_class
                    )
                )
            )


            if count > 0:

                transitions.append({

                    "From_Class":
                        old_class,

                    "To_Class":
                        new_class,

                    "Cells":
                        count,

                    "Area_km2":
                        count
                        *
                        cell_area_km2
                })


    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "From_Class",
            "To_Class",
            "Cells",
            "Area_km2"
        ]
    )


    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    scenario_folder = None


    if save_outputs:

        rain_text = (
            f"{rainfall_mm:.2f}"
            .replace(".", "p")
        )


        scenario_name = (
            f"R_Y{year}_"
            f"Rain_{rain_text}mm"
        )


        scenario_folder = (

            root /
            "Scenarios" /
            "R" /
            scenario_name
        )


        scenario_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        _save_float_raster(
            scenario_folder /
            "Piscopo_Score_Scenario.tif",

            scenario_score,

            r_profile
        )


        _save_float_raster(
            scenario_folder /
            "Recharge_cm_Scenario.tif",

            scenario_recharge,

            r_profile
        )


        _save_float_raster(
            scenario_folder /
            "R_Rate_Scenario.tif",

            scenario_r_rate,

            baseline_r_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_R_Rate.tif",

            delta_r_rate,

            baseline_r_profile
        )


        _save_float_raster(
            scenario_folder /
            "SVI_Scenario.tif",

            scenario_svi,

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_SVI.tif",

            delta_svi,

            svi_profile
        )


        rank_profile = (
            svi_profile.copy()
        )

        rank_profile.update(
            dtype="uint8",
            nodata=0,
            count=1,
            compress="lzw"
        )


        with rasterio.open(
            scenario_folder /
            "SVI_Scenario_Rank.tif",
            "w",
            **rank_profile
        ) as dst:

            dst.write(
                scenario_rank,
                1
            )


        pd.DataFrame({
            "Parameter":
                list(
                    summary.keys()
                ),

            "Value":
                list(
                    summary.values()
                )
        }).to_csv(
            scenario_folder /
            "Scenario_Summary.csv",

            index=False,
            encoding="utf-8-sig"
        )


        transitions_df.to_csv(
            scenario_folder /
            "Class_Transitions.csv",

            index=False,
            encoding="utf-8-sig"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "summary":
            summary,

        "transitions":
            transitions_df,

        "baseline_r_rate":
            baseline_r_rate,

        "scenario_r_rate":
            scenario_r_rate,

        "delta_r_rate":
            delta_r_rate,

        "scenario_score":
            scenario_score,

        "scenario_recharge":
            scenario_recharge,

        "baseline_svi":
            baseline_svi,

        "scenario_svi":
            scenario_svi,

        "delta_svi":
            delta_svi,

        "baseline_rank":
            baseline_rank,

        "scenario_rank":
            scenario_rank,

        "svi_mask":
            svi_mask,

        "final_mask":
            final_mask,

        "svi_profile":
            svi_profile,

        "scenario_folder":
            scenario_folder
    }



# ============================================================
# COMBINED T + P + R SCENARIO
# ============================================================

def run_combined_scenario(
    year,

    t_piezometer,
    t_new,
    p_zone_id,
    p_scenario,

    rainfall_mm,

    save_outputs=True,
    root=ROOT
):
    """
    Combine the already validated T, P and R scenario modules.

    Important:
    This is an ALPRIFT index scenario combination:

        Delta_SVI_combined =
            Delta_SVI_T
            + Delta_SVI_P
            + Delta_SVI_R

    It does not model physical feedbacks or temporal lag between
    pumping, recharge, groundwater decline and land subsidence.
    """

    root = Path(root)

    year = int(year)


    t_multi_mode = isinstance(
        t_piezometer,
        dict
    )


    if t_multi_mode:

        t_input = {
            str(key): float(value)
            for key, value
            in t_piezometer.items()
        }

        if not t_input:
            raise ValueError(
                "At least one T piezometer change is required."
            )

        primary_t_piezometer = next(
            iter(
                t_input
            )
        )

        primary_t_new = float(
            t_input[
                primary_t_piezometer
            ]
        )

        t_new_input = None

    else:

        primary_t_piezometer = str(
            t_piezometer
        )

        primary_t_new = float(
            t_new
        )

        t_input = (
            primary_t_piezometer
        )

        t_new_input = (
            primary_t_new
        )


    # Backward-compatible primary fields
    t_piezometer = (
        primary_t_piezometer
    )

    t_new = (
        primary_t_new
    )


    p_multi_mode = isinstance(
        p_zone_id,
        dict
    )


    if p_multi_mode:

        p_input = {
            int(key):
                float(value)

            for key, value
            in p_zone_id.items()
        }


        if not p_input:

            raise ValueError(
                "At least one pumping-zone "
                "scenario is required."
            )


        primary_p_zone_id = next(
            iter(
                p_input
            )
        )


        primary_p_scenario = float(
            p_input[
                primary_p_zone_id
            ]
        )


        p_scenario_input = None


    else:

        primary_p_zone_id = int(
            p_zone_id
        )


        primary_p_scenario = float(
            p_scenario
        )


        if (
            not np.isfinite(
                primary_p_scenario
            )
            or
            primary_p_scenario < 0
        ):

            raise ValueError(
                "p_scenario must be finite "
                "and non-negative."
            )


        p_input = (
            primary_p_zone_id
        )


        p_scenario_input = (
            primary_p_scenario
        )


    # Primary fields retained for summary/output naming
    p_zone_id = int(
        primary_p_zone_id
    )


    p_scenario = float(
        primary_p_scenario
    )


    rainfall_mm = float(
        rainfall_mm
    )


    # ========================================================
    # RUN VALIDATED COMPONENT ENGINES    # ========================================================
    # RUN VALIDATED COMPONENT ENGINES
    # ========================================================

    t_result = run_t_scenario(
        year=year,
        piezometer=t_input,
        new_t=t_new_input,
        save_outputs=False,
        root=root
    )


    p_result = run_p_scenario(
        year=year,
        zone_id=p_input,
        scenario_p=p_scenario_input,
        save_outputs=False,
        root=root
    )


    r_result = run_r_scenario(
        year=year,
        rainfall_mm=rainfall_mm,
        save_outputs=False,
        root=root
    )


    # ========================================================
    # OFFICIAL BASELINE
    # ========================================================

    baseline_svi = (
        t_result[
            "baseline_svi"
        ].copy()
    )

    baseline_rank = (
        t_result[
            "baseline_rank"
        ].copy()
    )

    svi_profile = (
        t_result[
            "svi_profile"
        ].copy()
    )


    # ========================================================
    # CHECK SHAPES
    # ========================================================

    reference_shape = (
        baseline_svi.shape
    )


    arrays_to_check = [
        t_result["delta_svi"],
        p_result["delta_svi"],
        r_result["delta_svi"],
        p_result["baseline_svi"],
        r_result["baseline_svi"]
    ]


    for arr in arrays_to_check:

        if arr.shape != reference_shape:

            raise ValueError(
                "Component scenario grids do not have identical SVI shapes."
            )


    # ========================================================
    # COMMON VALID MASK
    # ========================================================

    final_mask = (

        t_result[
            "final_mask"
        ]

        &

        p_result[
            "final_mask"
        ]

        &

        r_result[
            "final_mask"
        ]

        &

        np.isfinite(
            t_result[
                "delta_svi"
            ]
        )

        &

        np.isfinite(
            p_result[
                "delta_svi"
            ]
        )

        &

        np.isfinite(
            r_result[
                "delta_svi"
            ]
        )

        &

        np.isfinite(
            baseline_svi
        )
    )


    # ========================================================
    # COMBINE DELTAS
    # ========================================================

    delta_svi = np.full(
        reference_shape,
        np.nan,
        dtype=np.float64
    )


    delta_svi[
        final_mask
    ] = (

        t_result[
            "delta_svi"
        ][
            final_mask
        ]

        +

        p_result[
            "delta_svi"
        ][
            final_mask
        ]

        +

        r_result[
            "delta_svi"
        ][
            final_mask
        ]
    )


    # ========================================================
    # COMBINED SVI
    # ========================================================

    scenario_svi = np.full(
        reference_shape,
        np.nan,
        dtype=np.float64
    )


    scenario_svi[
        final_mask
    ] = (

        baseline_svi[
            final_mask
        ]

        +

        delta_svi[
            final_mask
        ]
    )


    scenario_rank = (
        classify_svi(
            scenario_svi
        )
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    transform = (
        svi_profile[
            "transform"
        ]
    )


    cell_area_km2 = (

        abs(
            transform.a
        )

        *

        abs(
            transform.e
        )

        /

        1_000_000.0
    )


    changed_mask = (

        final_mask

        &

        (
            np.abs(
                delta_svi
            )
            >
            1e-12
        )
    )


    changed_svi_cells = int(
        np.count_nonzero(
            changed_mask
        )
    )


    changed_area_km2 = float(
        changed_svi_cells
        *
        cell_area_km2
    )


    valid_cells = int(
        np.count_nonzero(
            final_mask
        )
    )


    valid_area_km2 = float(
        valid_cells
        *
        cell_area_km2
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "year":
            year,

        # T
        "t_piezometer":
            t_piezometer,

        "t_baseline":
            t_result[
                "summary"
            ][
                "baseline_t"
            ],

        "t_scenario":
            t_new,

        "t_change_count":
            int(
                t_result[
                    "summary"
                ].get(
                    "t_change_count",
                    1
                )
            ),

        "t_changes":
            t_result[
                "summary"
            ].get(
                "t_changes",
                {
                    t_piezometer:
                        t_new
                }
            ),

        "t_baseline_by_piezometer":
            t_result[
                "summary"
            ].get(
                "baseline_t_by_piezometer",
                {
                    t_piezometer:
                        t_result[
                            "summary"
                        ][
                            "baseline_t"
                        ]
                }
            ),

        # P
        "p_zone_id":
            p_zone_id,

        "p_scenario":
            p_scenario,

        "p_baseline_cm_yr":
            p_result[
                "summary"
            ][
                "baseline_p_cm_yr"
            ],

        "p_scenario_cm_yr":
            p_result[
                "summary"
            ][
                "scenario_p_cm_yr"
            ],

        "p_baseline_rate":
            p_result[
                "summary"
            ][
                "baseline_p_rate"
            ],

        "p_scenario_rate":
            p_result[
                "summary"
            ][
                "scenario_p_rate"
            ],

        "p_change_count":
            int(
                p_result[
                    "summary"
                ].get(
                    "zone_count",
                    1
                )
            ),

        "p_scenarios":
            p_result[
                "summary"
            ].get(
                "zone_scenarios",
                {
                    p_zone_id:
                        p_scenario
                }
            ),

        "p_zone_details":
            p_result[
                "summary"
            ].get(
                "zone_details",
                {}
            ),

        # R
        "baseline_rainfall_mm":
            r_result[
                "summary"
            ][
                "baseline_rainfall_mm"
            ],

        "rainfall_mm":
            rainfall_mm,

        "baseline_rain_rank":
            r_result[
                "summary"
            ][
                "baseline_rain_rank"
            ],

        "scenario_rain_rank":
            r_result[
                "summary"
            ][
                "scenario_rain_rank"
            ],

        # Combined
        "valid_cells":
            valid_cells,

        "valid_area_km2":
            valid_area_km2,

        "changed_svi_cells":
            changed_svi_cells,

        "changed_area_km2":
            changed_area_km2,

        "baseline_mean_svi":
            float(
                np.mean(
                    baseline_svi[
                        final_mask
                    ]
                )
            ),

        "scenario_mean_svi":
            float(
                np.mean(
                    scenario_svi[
                        final_mask
                    ]
                )
            ),

        "mean_delta_svi":
            float(
                np.mean(
                    delta_svi[
                        final_mask
                    ]
                )
            ),

        "max_delta_svi":
            float(
                np.max(
                    delta_svi[
                        final_mask
                    ]
                )
            ),

        "min_delta_svi":
            float(
                np.min(
                    delta_svi[
                        final_mask
                    ]
                )
            )
    }


    # ========================================================
    # CLASS TRANSITIONS
    # ========================================================

    baseline_rank_valid = (

        np.isfinite(
            baseline_rank
        )

        &

        (
            baseline_rank > 0
        )
    )


    scenario_rank_valid = (

        np.isfinite(
            scenario_rank
        )

        &

        (
            scenario_rank > 0
        )
    )


    transition_mask = (

        final_mask

        &

        baseline_rank_valid

        &

        scenario_rank_valid
    )


    transitions = []


    for old_class in range(
        1,
        5
    ):

        for new_class in range(
            1,
            5
        ):

            count = int(
                np.count_nonzero(

                    transition_mask

                    &

                    (
                        baseline_rank
                        ==
                        old_class
                    )

                    &

                    (
                        scenario_rank
                        ==
                        new_class
                    )
                )
            )


            if count > 0:

                transitions.append({

                    "From_Class":
                        old_class,

                    "To_Class":
                        new_class,

                    "Cells":
                        count,

                    "Area_km2":
                        count
                        *
                        cell_area_km2
                })


    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "From_Class",
            "To_Class",
            "Cells",
            "Area_km2"
        ]
    )


    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    scenario_folder = None


    if save_outputs:

        t_text = (
            f"{t_new:.2f}"
            .replace(".", "p")
        )

        p_text = (
            f"{p_scenario:.3f}"
            .replace(
                ".",
                "p"
            )
        )


        rain_text = (
            f"{rainfall_mm:.2f}"
            .replace(".", "p")
        )


        if t_multi_mode:
            t_name_text = (
                f"T_MULTI{len(t_input)}"
            )
        else:
            t_name_text = (
                f"T_{t_piezometer}_{t_text}"
            )


        if p_multi_mode:
            p_name_text = (
                f"P_MULTI{len(p_input)}"
            )
        else:
            p_name_text = (
                f"P_Zone{p_zone_id}_to_{p_text}"
            )


        scenario_name = (
            f"Combined_Y{year}_"
            f"{t_name_text}_"
            f"{p_name_text}_"
            f"R_{rain_text}mm"
        )


        scenario_folder = (

            root /
            "Scenarios" /
            "Combined" /
            scenario_name
        )


        scenario_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        # --------------------------------------------
        # Component Delta SVI rasters
        # --------------------------------------------

        _save_float_raster(
            scenario_folder /
            "Delta_SVI_T.tif",

            t_result[
                "delta_svi"
            ],

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_SVI_P.tif",

            p_result[
                "delta_svi"
            ],

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "Delta_SVI_R.tif",

            r_result[
                "delta_svi"
            ],

            svi_profile
        )


        # --------------------------------------------
        # Combined rasters
        # --------------------------------------------

        _save_float_raster(
            scenario_folder /
            "Delta_SVI_Combined.tif",

            delta_svi,

            svi_profile
        )


        _save_float_raster(
            scenario_folder /
            "SVI_Combined.tif",

            scenario_svi,

            svi_profile
        )


        # --------------------------------------------
        # Combined rank
        # --------------------------------------------

        rank_profile = (
            svi_profile.copy()
        )

        rank_profile.update(
            dtype="uint8",
            nodata=0,
            count=1,
            compress="lzw"
        )


        rank_to_save = np.where(
            final_mask,
            scenario_rank,
            0
        ).astype(np.uint8)


        with rasterio.open(
            scenario_folder /
            "SVI_Combined_Rank.tif",
            "w",
            **rank_profile
        ) as dst:

            dst.write(
                rank_to_save,
                1
            )


        # --------------------------------------------
        # Summary
        # --------------------------------------------

        pd.DataFrame({
            "Parameter":
                list(
                    summary.keys()
                ),

            "Value":
                list(
                    summary.values()
                )
        }).to_csv(
            scenario_folder /
            "Scenario_Summary.csv",

            index=False,
            encoding="utf-8-sig"
        )


        transitions_df.to_csv(
            scenario_folder /
            "Class_Transitions.csv",

            index=False,
            encoding="utf-8-sig"
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "summary":
            summary,

        "transitions":
            transitions_df,

        "baseline_svi":
            baseline_svi,

        "scenario_svi":
            scenario_svi,

        "delta_svi":
            delta_svi,

        "baseline_rank":
            baseline_rank,

        "scenario_rank":
            scenario_rank,

        "final_mask":
            final_mask,

        "svi_mask":
            final_mask,

        "svi_profile":
            svi_profile,

        "scenario_folder":
            scenario_folder,

        # Keep components available for auditing
        "t_result":
            t_result,

        "p_result":
            p_result,

        "r_result":
            r_result
    }
