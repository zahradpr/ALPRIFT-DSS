
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st

from engine import (
    run_t_scenario,
    run_p_scenario,
    run_r_scenario,
    run_combined_scenario
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="ALPRIFT Scenario Simulator",
    page_icon="🗺️",
    layout="wide"
)


st.markdown(
    """
    <style>

    .main {
        direction: rtl;
    }

    h1, h2, h3, p, label, div {
        font-family: Tahoma, Arial, sans-serif;
    }

    .stMetric {
        direction: rtl;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CACHE DATA
# ============================================================

@st.cache_data
def load_piezometers(year):

    path = (
        ROOT /
        "SourceData" /
        "Piezometers" /
        f"T_{year}_Points.shp"
    )

    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    gdf = gpd.read_file(path)

    gdf["Piezometer"] = (
        gdf["Piezometer"]
        .astype(str)
    )

    return gdf


@st.cache_data
def load_pumping_zones():

    path = (
        ROOT /
        "SourceData" /
        "Pumping" /
        "P_Thi_1398_Final3.shp"
    )

    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    gdf = gpd.read_file(path)

    gdf["TARGET_FID"] = (
        gdf["TARGET_FID"]
        .astype(int)
    )

    return gdf


# ============================================================
# MAP FUNCTION
# ============================================================

def raster_figure(
    array,
    title,
    vmin=None,
    vmax=None,
    cmap="viridis"
):

    fig, ax = plt.subplots(
        figsize=(6, 4.5)
    )

    image = ax.imshow(
        array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_axis_off()

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.02
    )

    plt.tight_layout()

    return fig


CLASS_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High"
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "ALPRIFT Dashboard"
)

model_tab, dss_tab, weight_tab = st.tabs(
    [
        "ALPRIFT Model",
        "Scenario / DSS",
        "Weight Optimization",
    ]
)


# ============================================================
# TAB 1 — REAL ALPRIFT MODEL
# ============================================================

with model_tab:

    # ========================================================
    # ALPRIFT MODEL HEADER
    # ========================================================

    header_left, header_right = st.columns(
        [4, 1.2]
    )

    with header_left:

        st.subheader(
            "مدل واقعی ALPRIFT دشت شبستر"
        )

        st.caption(
            "شش لایه ثابت A, L, P, R, I, F و یک لایه زمانی T "
            "بر پایه داده‌های اندازه‌گیری‌شده سطح آب زیرزمینی"
        )

    with header_right:

        model_year = st.selectbox(
            "سال مدل ALPRIFT",
            options=list(range(1389, 1404)),
            index=list(range(1389, 1404)).index(1400),
            key="alprift_model_year"
        )


    st.info(
        f"سال انتخابی: {model_year} | "
        "لایه T از تغییرات اندازه‌گیری‌شده سطح آب "
        "در آذر سال قبل تا آذر این سال ساخته شده است."
    )


    # ========================================================
    # THREE-PANEL MODEL LAYOUT
    # ========================================================

    layers_panel, map_panel = st.columns(
        [1.35, 4.65],
        gap="large"
    )


    # --------------------------------------------------------
    # LEFT — ALPRIFT LAYERS
    # --------------------------------------------------------

    with layers_panel:

        st.subheader(
            "ALPRIFT Layers"
        )

        st.caption(
            "Model components and baseline weights"
        )

        model_active_layer = st.radio(
            "ALPRIFT Model Layer",
            options=[
                "A | Aquifer Media",
                "L | Land Use",
                "P | Pumping",
                "R | Recharge",
                "I | Aquifer Thickness",
                "F | Distance from Fault",
                "T | Groundwater-level Decline",
            ],
            index=0,
            key="alprift_model_layer"
        )

        st.caption(
            "ΣW = 24"
        )


    # --------------------------------------------------------
    # CENTER — MAIN MAP
    # --------------------------------------------------------

    with map_panel:

        st.subheader(
            "Main Map"
        )

        selected_code = model_active_layer.split(
            "|",
            1
        )[0].strip()

        from model_maps import (
            resolve_model_raster_path,
            load_model_raster,
        )

        import numpy as np
        import matplotlib.pyplot as plt

        model_root = ROOT

        raster_path = resolve_model_raster_path(
            selected_code,
            model_year,
            root=model_root,
        )

        raster = load_model_raster(
            raster_path
        )

        data = raster["data"]
        bounds = raster["bounds"]

        valid = data[
            np.isfinite(data)
        ]

        if valid.size == 0:

            st.error(
                "The selected raster contains no valid cells."
            )

        else:

            vmin = float(
                valid.min()
            )

            vmax = float(
                valid.max()
            )

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            image = ax.imshow(
                data,
                extent=[
                    bounds.left,
                    bounds.right,
                    bounds.bottom,
                    bounds.top,
                ],
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(
                f"ALPRIFT Layer {selected_code}"
            )

            ax.set_xlabel(
                "Easting (m)"
            )

            ax.set_ylabel(
                "Northing (m)"
            )

            colorbar = fig.colorbar(
                image,
                ax=ax,
                shrink=0.82
            )

            colorbar.set_label(
                "ALPRIFT Rating"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )

            plt.close(fig)

            st.caption(
                f"Raster: {raster_path.name}"
            )

            if selected_code == "T":

                st.caption(
                    f"Model year: {model_year} | "
                    "Azar-to-Azar groundwater-level change"
                )

            else:

                st.caption(
                    "Static layer in the current ALPRIFT time series"
                )


    # --------------------------------------------------------
    # LAYER DETAILS — FULL WIDTH
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Layer Details"
    )

    layer_details = {

        "A": {
            "name": "Aquifer Media",
            "weight": 5,
            "status": "Static",
            "data": (
                "Aquifer-media information used to construct "
                "the finalized A layer."
            ),
            "method": (
                "Continuous IDW-based surface used as the final "
                "A layer. The finalized continuous surface is "
                "retained without an additional reclassification."
            ),
            "temporal": (
                "Static in the current annual ALPRIFT model series."
            ),
            "output": "ALPRIFT A rating surface.",
        },

        "L": {
            "name": "Land Use",
            "weight": 3,
            "status": "Static",
            "data": (
                "Land-use information used in the baseline "
                "ALPRIFT model."
            ),
            "method": (
                "Baseline ALPRIFT land-use rating layer."
            ),
            "temporal": (
                "Static in the current annual ALPRIFT model series."
            ),
            "output": "ALPRIFT L rating raster.",
        },

        "P": {
            "name": "Groundwater Pumping",
            "weight": 4,
            "status": "Static",
            "data": (
                "Groundwater pumping and well-withdrawal data. "
                "Baseline source: P_Thi_1398_Final3.shp."
            ),
            "method": (
                "Thiessen-polygon representation of pumping. "
                "The P_Rate field was validated against the "
                "official baseline P raster."
            ),
            "temporal": (
                "Static in the current historical ALPRIFT series. "
                "Pumping is handled separately as a scenario "
                "variable in the DSS."
            ),
            "output": "ALPRIFT P rating raster.",
        },

        "R": {
            "name": "Recharge",
            "weight": 4,
            "status": "Static",
            "data": (
                "Rainfall, slope and permeability information."
            ),
            "method": (
                "Piscopo-based recharge scoring followed by "
                "conversion to the ALPRIFT R rating."
            ),
            "temporal": (
                "Static in the current historical ALPRIFT series. "
                "Rainfall is the variable component used in "
                "the DSS scenarios."
            ),
            "output": "ALPRIFT R rating raster.",
        },

        "I": {
            "name": "Aquifer Thickness",
            "weight": 2,
            "status": "Static",
            "data": (
                "Aquifer-thickness information used in the "
                "baseline ALPRIFT model."
            ),
            "method": (
                "Final baseline aquifer-thickness rating layer."
            ),
            "temporal": (
                "Static in the current annual ALPRIFT model series."
            ),
            "output": "ALPRIFT I rating raster.",
        },

        "F": {
            "name": "Distance from Fault",
            "weight": 1,
            "status": "Static",
            "data": (
                "Fault-network spatial data."
            ),
            "method": (
                "Distance from faults was calculated and rated as "
                "0–1 km = 10, 1–2 km = 8, 2–3 km = 6, "
                "3–4 km = 4, 4–5 km = 2, and >5 km = 1."
            ),
            "temporal": (
                "Static in the current annual ALPRIFT model series."
            ),
            "output": "ALPRIFT F rating raster.",
        },

        "T": {
            "name": "Groundwater-level Decline",
            "weight": 5,
            "status": "Annual / Dynamic",
            "data": (
                "Measured groundwater-level observations "
                "from piezometers."
            ),
            "method": (
                "Annual Azar-to-Azar groundwater-level change, "
                "spatialized using IDW with Power = 1.5, "
                "40 nearest observations and a 100 m grid, "
                "then converted to the ALPRIFT T rating."
            ),
            "temporal": (
                "Annual. Positive values represent groundwater "
                "decline and negative values represent "
                "groundwater recovery."
            ),
            "output": "Annual ALPRIFT T rating raster.",
        },
    }

    details = layer_details[
        selected_code
    ]

    detail_left, detail_right = st.columns(
        2,
        gap="large"
    )

    with detail_left:

        st.markdown(
            f"### {selected_code} — {details['name']}"
        )

        st.markdown(
            f"**Weight:** {details['weight']}"
        )

        st.markdown(
            f"**Temporal status:** {details['status']}"
        )

        if selected_code == "T":

            st.markdown(
                f"**Model year:** {model_year}"
            )

        else:

            st.markdown(
                "**Year dependence:** None in the current "
                "historical ALPRIFT series"
            )

        st.markdown(
            f"**Temporal behavior:** {details['temporal']}"
        )


    with detail_right:

        st.markdown(
            f"**Input data:** {details['data']}"
        )

        st.markdown(
            f"**Method:** {details['method']}"
        )

        st.markdown(
            f"**Output:** {details['output']}"
        )


    # ============================================================
    # MODEL OUTPUT FLOW
    # ========================================================

    st.markdown("---")

    from model_maps import resolve_model_output_paths

    annual_outputs = resolve_model_output_paths(
        model_year,
        root=model_root,
    )

    t_output_path = resolve_model_raster_path(
        "T",
        model_year,
        root=model_root,
    )

    svi_output_path = annual_outputs["svi"]
    rank_output_path = annual_outputs["rank"]

    output_1, output_2, output_3 = st.columns(3)

    with output_1:

        st.markdown(
            f"### T ({model_year})"
        )

        st.caption(
            "Measured groundwater-level change | Azar → Azar"
        )

        st.caption(
            f"Raster: {t_output_path.name}"
        )

    with output_2:

        st.markdown(
            f"### SVI ({model_year})"
        )

        st.caption(
            "ALPRIFT vulnerability index"
        )

        st.caption(
            f"Raster: {svi_output_path.name}"
        )

    with output_3:

        st.markdown(
            f"### Vulnerability Class ({model_year})"
        )

        st.caption(
            "Final ALPRIFT vulnerability class"
        )

        st.caption(
            f"Raster: {rank_output_path.name}"
        )




    # ========================================================
    # ANNUAL ALPRIFT–InSAR COMPARISON
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Annual ALPRIFT–InSAR Comparison"
    )

    st.caption(
        "Official annual comparison between the Reference2018 "
        "ALPRIFT vulnerability model and independent InSAR observations."
    )

    from official_comparison import (
        has_official_comparison,
        load_official_comparison,
    )

    official_comparison = None

    if has_official_comparison(model_year):

        official_comparison = load_official_comparison(
            model_year,
            root=ROOT,
        )

        official_paths = official_comparison[
            "paths"
        ]

        comparison_svi_path = official_paths[
            "svi"
        ]

        comparison_rank_path = official_paths[
            "rank"
        ]

    else:

        comparison_svi_path = svi_output_path
        comparison_rank_path = rank_output_path


    annual_col_1, annual_col_2, annual_col_3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # OFFICIAL SVI MAP
    # --------------------------------------------------------

    with annual_col_1:

        st.subheader(
            "SVI Map"
        )

        svi_raster = load_model_raster(
            comparison_svi_path
        )

        svi_data = svi_raster["data"]
        svi_bounds = svi_raster["bounds"]

        fig_svi, ax_svi = plt.subplots(
            figsize=(5.5, 4.2)
        )

        svi_image = ax_svi.imshow(
            svi_data,
            extent=[
                svi_bounds.left,
                svi_bounds.right,
                svi_bounds.bottom,
                svi_bounds.top,
            ],
            origin="upper",
            cmap="viridis",
            vmin=24,
            vmax=240,
        )

        ax_svi.set_title(
            f"SVI — {model_year}"
        )

        ax_svi.set_xlabel(
            "Easting (m)"
        )

        ax_svi.set_ylabel(
            "Northing (m)"
        )

        svi_colorbar = fig_svi.colorbar(
            svi_image,
            ax=ax_svi,
            shrink=0.78
        )

        svi_colorbar.set_label(
            "SVI"
        )

        fig_svi.tight_layout()

        st.pyplot(
            fig_svi,
            clear_figure=True
        )

        plt.close(
            fig_svi
        )

        st.caption(
            f"Raster: {comparison_svi_path.name}"
        )

        if official_comparison is not None:

            st.caption(
                "Reference2018 ALPRIFT | Official annual comparison"
            )

        else:

            st.caption(
                "ALPRIFT model output"
            )


    # --------------------------------------------------------
    # OFFICIAL VULNERABILITY CLASS MAP
    # --------------------------------------------------------

    with annual_col_2:

        st.subheader(
            "Vulnerability Class Map"
        )

        from matplotlib.colors import (
            ListedColormap,
            BoundaryNorm,
        )

        rank_raster = load_model_raster(
            comparison_rank_path
        )

        rank_data = rank_raster["data"]
        rank_bounds = rank_raster["bounds"]

        rank_cmap = ListedColormap(
            [
                "green",
                "gold",
                "darkorange",
                "firebrick",
            ]
        )

        rank_norm = BoundaryNorm(
            [
                0.5,
                1.5,
                2.5,
                3.5,
                4.5,
            ],
            rank_cmap.N
        )

        fig_rank, ax_rank = plt.subplots(
            figsize=(5.5, 4.2)
        )

        rank_image = ax_rank.imshow(
            rank_data,
            extent=[
                rank_bounds.left,
                rank_bounds.right,
                rank_bounds.bottom,
                rank_bounds.top,
            ],
            origin="upper",
            cmap=rank_cmap,
            norm=rank_norm,
        )

        ax_rank.set_title(
            f"Vulnerability Class — {model_year}"
        )

        ax_rank.set_xlabel(
            "Easting (m)"
        )

        ax_rank.set_ylabel(
            "Northing (m)"
        )

        rank_colorbar = fig_rank.colorbar(
            rank_image,
            ax=ax_rank,
            ticks=[1, 2, 3, 4],
            shrink=0.78
        )

        rank_colorbar.ax.set_yticklabels(
            [
                "Low",
                "Moderate",
                "High",
                "Very High",
            ]
        )

        fig_rank.tight_layout()

        st.pyplot(
            fig_rank,
            clear_figure=True
        )

        plt.close(
            fig_rank
        )

        st.caption(
            f"Raster: {comparison_rank_path.name}"
        )

        if official_comparison is not None:

            st.caption(
                "Reference2018 classes | "
                "24–78 Low | 78–133 Moderate | "
                "133–186 High | 186–240 Very High"
            )


    # --------------------------------------------------------
    # InSAR MAP
    # --------------------------------------------------------

    with annual_col_3:

        st.subheader(
            "InSAR Map"
        )

        from insar_maps import (
            has_insar_data,
            resolve_insar_paths,
            get_insar_metadata,
        )

        if has_insar_data(model_year):

            from matplotlib.colors import TwoSlopeNorm

            insar_paths = resolve_insar_paths(
                model_year
            )

            insar_metadata = get_insar_metadata(
                model_year
            )

            insar_map_path = insar_paths[
                "final"
            ]

            insar_raster = load_model_raster(
                insar_map_path
            )

            insar_data = insar_raster["data"]
            insar_bounds = insar_raster["bounds"]

            valid_insar = insar_data[
                np.isfinite(insar_data)
            ]

            fig_insar, ax_insar = plt.subplots(
                figsize=(5.5, 4.2)
            )

            if valid_insar.size > 0:

                insar_min = float(
                    valid_insar.min()
                )

                insar_max = float(
                    valid_insar.max()
                )

                if insar_min < 0 < insar_max:

                    insar_norm = TwoSlopeNorm(
                        vmin=insar_min,
                        vcenter=0.0,
                        vmax=insar_max,
                    )

                    insar_image = ax_insar.imshow(
                        insar_data,
                        extent=[
                            insar_bounds.left,
                            insar_bounds.right,
                            insar_bounds.bottom,
                            insar_bounds.top,
                        ],
                        origin="upper",
                        cmap="RdBu_r",
                        norm=insar_norm,
                    )

                else:

                    insar_image = ax_insar.imshow(
                        insar_data,
                        extent=[
                            insar_bounds.left,
                            insar_bounds.right,
                            insar_bounds.bottom,
                            insar_bounds.top,
                        ],
                        origin="upper",
                        cmap="RdBu_r",
                    )

                ax_insar.set_title(
                    f"InSAR — {model_year}"
                )

                ax_insar.set_xlabel(
                    "Easting (m)"
                )

                ax_insar.set_ylabel(
                    "Northing (m)"
                )

                insar_colorbar = fig_insar.colorbar(
                    insar_image,
                    ax=ax_insar,
                    shrink=0.78
                )

                insar_colorbar.set_label(
                    "Cumulative displacement (mm)"
                )

                fig_insar.tight_layout()

                st.pyplot(
                    fig_insar,
                    clear_figure=True
                )

                plt.close(
                    fig_insar
                )

                st.caption(
                    f"Raster: {insar_map_path.name}"
                )

                st.caption(
                    f"{insar_metadata['start_date']} → "
                    f"{insar_metadata['end_date']} | "
                    "Independent observation"
                )

        else:

            st.info(
                f"InSAR data not available for model year {model_year}"
            )


    # ========================================================
    # ALPRIFT–InSAR SPATIAL SIMILARITY
    # ========================================================

    if official_comparison is not None:

        metrics = official_comparison[
            "metrics"
        ]

        st.markdown("---")

        st.subheader(
            f"ALPRIFT–InSAR Spatial Similarity — {model_year}"
        )

        st.caption(
            "Pixel-by-pixel spatial agreement between the "
            "Reference2018 ALPRIFT vulnerability index and "
            "independent InSAR observations."
        )

        sim1, sim2, sim3, sim4 = st.columns(4)

        sim1.metric(
            "Pearson r",
            f"{metrics['pearson_r']:.3f}"
        )

        sim2.metric(
            "Spearman ρ",
            f"{metrics['spearman_rho']:.3f}"
        )

        sim3.metric(
            "Pearson r²",
            f"{metrics['pearson_r2']:.3f}"
        )

        sim4.metric(
            "Common pixels",
            f"{metrics['common_pixels']:,}"
        )

        pearson_r = metrics[
            "pearson_r"
        ]

        spearman_rho = metrics[
            "spearman_rho"
        ]

        if pearson_r > 0 and spearman_rho > 0:

            interpretation = (
                "Both correlation coefficients are positive, "
                "indicating that higher ALPRIFT vulnerability "
                "generally tends to coincide with greater observed "
                "InSAR subsidence. However, the small coefficient "
                "values indicate limited pixel-by-pixel spatial similarity."
            )

        elif pearson_r < 0 and spearman_rho < 0:

            interpretation = (
                "Both correlation coefficients are negative, "
                "indicating an inverse spatial relationship between "
                "ALPRIFT vulnerability and observed InSAR subsidence."
            )

        else:

            interpretation = (
                "Pearson and Spearman correlations do not show a "
                "consistent spatial relationship between ALPRIFT "
                "vulnerability and observed InSAR subsidence."
            )

        st.info(
            interpretation
        )

        st.caption(
            f"Official common area: "
            f"{metrics['common_area_km2']:.2f} km² | "
            "InSAR is used as an independent observation."
        )


    # ========================================================
    # PIEZOMETER-BASED InSAR POINT ANALYSIS
    # ========================================================

    if has_insar_data(model_year):

        import altair as alt
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.colors import TwoSlopeNorm

        from model_maps import (
            load_model_raster,
        )

        from insar_maps import (
            load_piezometers,
            extract_insar_timeseries_at_piezometer,
            load_insar_timeseries_table,
            resolve_insar_paths,
        )

        st.markdown("---")

        st.subheader(
            "Piezometer-based InSAR Point Analysis"
        )

        st.caption(
            "Explore satellite-derived displacement at "
            "the location of a selected piezometer."
        )


        # ----------------------------------------------------
        # PIEZOMETER SELECTOR
        # ----------------------------------------------------

        piezometers = load_piezometers(
            model_year
        )

        piezometer_options = sorted(
            piezometers["Piezometer"]
            .astype(str)
            .tolist()
        )

        default_index = (
            piezometer_options.index("P04")
            if "P04" in piezometer_options
            else 0
        )

        selected_piezometer = st.selectbox(
            "Selected Piezometer",
            piezometer_options,
            index=default_index,
            key="insar_selected_piezometer"
        )

        selected_piezometer_row = piezometers.loc[
            piezometers["Piezometer"].astype(str)
            == selected_piezometer
        ].iloc[0]

        piezo_x = float(
            selected_piezometer_row.geometry.x
        )

        piezo_y = float(
            selected_piezometer_row.geometry.y
        )


        # ----------------------------------------------------
        # POINT MAP + PIEZOMETER DETAILS
        # ----------------------------------------------------

        point_map_col, point_info_col = st.columns(
            [1.45, 1],
            gap="large"
        )


        with point_map_col:

            st.markdown(
                "#### Selected Location on InSAR Map"
            )

            point_insar_path = resolve_insar_paths(
                model_year
            )["final"]

            point_raster = load_model_raster(
                point_insar_path
            )

            point_data = point_raster["data"]
            point_bounds = point_raster["bounds"]

            point_valid = point_data[
                np.isfinite(point_data)
            ]

            fig_point, ax_point = plt.subplots(
                figsize=(6.2, 3.7)
            )

            if point_valid.size > 0:

                point_min = float(
                    point_valid.min()
                )

                point_max = float(
                    point_valid.max()
                )

                if point_min < 0 < point_max:

                    point_norm = TwoSlopeNorm(
                        vmin=point_min,
                        vcenter=0.0,
                        vmax=point_max,
                    )

                    point_image = ax_point.imshow(
                        point_data,
                        extent=[
                            point_bounds.left,
                            point_bounds.right,
                            point_bounds.bottom,
                            point_bounds.top,
                        ],
                        origin="upper",
                        cmap="RdBu_r",
                        norm=point_norm,
                    )

                else:

                    point_image = ax_point.imshow(
                        point_data,
                        extent=[
                            point_bounds.left,
                            point_bounds.right,
                            point_bounds.bottom,
                            point_bounds.top,
                        ],
                        origin="upper",
                        cmap="RdBu_r",
                    )

                ax_point.scatter(
                    piezo_x,
                    piezo_y,
                    s=75,
                    marker="o",
                    edgecolors="black",
                    linewidths=1.3,
                    zorder=10,
                )

                ax_point.annotate(
                    selected_piezometer,
                    (piezo_x, piezo_y),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    zorder=11,
                )

                ax_point.set_title(
                    f"InSAR Location — {selected_piezometer}"
                )

                ax_point.set_xlabel(
                    "Easting (m)"
                )

                ax_point.set_ylabel(
                    "Northing (m)"
                )

                point_colorbar = fig_point.colorbar(
                    point_image,
                    ax=ax_point,
                    shrink=0.80
                )

                point_colorbar.set_label(
                    "Cumulative displacement (mm)"
                )

                fig_point.tight_layout()

                st.pyplot(
                    fig_point,
                    clear_figure=True
                )

                plt.close(
                    fig_point
                )


        with point_info_col:

            st.markdown(
                "#### Piezometer Details"
            )

            t_value = float(
                selected_piezometer_row["T_myr"]
            )

            h_prev = float(
                selected_piezometer_row["H_prev_m9"]
            )

            h_curr = float(
                selected_piezometer_row["H_curr_m9"]
            )

            hydro_status = str(
                selected_piezometer_row["Hydro_Stat"]
            )

            selected_year = int(
                model_year
            )

            previous_year = (
                selected_year - 1
            )


            st.markdown(
                f"### {selected_piezometer}"
            )

            st.markdown(
                f"**Annual groundwater change (T):** "
                f"{t_value:+.2f} m"
            )

            st.markdown(
                f"**Status:** {hydro_status}"
            )


            st.markdown(
                "##### Period used to compute T"
            )

            st.markdown(
                f"**Azar {previous_year} → "
                f"Azar {selected_year}**"
            )


            st.markdown(
                "##### Groundwater level"
            )

            st.markdown(
                f"**{h_prev:.2f} m → "
                f"{h_curr:.2f} m**"
            )


            if t_value > 0:

                st.caption(
                    f"Groundwater level declined by "
                    f"{t_value:.2f} m during this "
                    f"Azar-to-Azar period."
                )

            elif t_value < 0:

                st.caption(
                    f"Groundwater level recovered by "
                    f"{abs(t_value):.2f} m during this "
                    f"Azar-to-Azar period."
                )

            else:

                st.caption(
                    "No groundwater-level change was "
                    "detected during this Azar-to-Azar period."
                )


            st.caption(
                "Positive T = decline | Negative T = recovery"
            )


            with st.expander(
                "Location details"
            ):

                st.write(
                    f"X = {piezo_x:.0f} m"
                )

                st.write(
                    f"Y = {piezo_y:.0f} m"
                )

        # ----------------------------------------------------
        # POINT-BASED InSAR TIME SERIES
        # ----------------------------------------------------

        point_ts = extract_insar_timeseries_at_piezometer(
            model_year,
            selected_piezometer,
        ).copy()

        point_ts["date_plot"] = pd.to_datetime(
            point_ts["date"]
        )

        st.markdown(
            f"### InSAR Time Series at {selected_piezometer}"
        )

        base = alt.Chart(
            point_ts
        ).encode(
            x=alt.X(
                "date_plot:T",
                title="Acquisition date",
                axis=alt.Axis(
                    format="%Y-%m-%d",
                    labelAngle=-45
                )
            )
        )

        line = base.mark_line(
            strokeWidth=2.5
        ).encode(
            y=alt.Y(
                "insar_mm:Q",
                title="Cumulative InSAR displacement (mm)"
            )
        )

        points = base.mark_point(
            filled=True,
            size=90
        ).encode(
            y="insar_mm:Q",
            tooltip=[
                alt.Tooltip(
                    "date:T",
                    title="Date"
                ),
                alt.Tooltip(
                    "insar_mm:Q",
                    title="InSAR displacement",
                    format=".2f"
                ),
                alt.Tooltip(
                    "Piezometer:N",
                    title="Piezometer"
                ),
            ]
        )

        zero_line = alt.Chart(
            pd.DataFrame(
                {"zero": [0.0]}
            )
        ).mark_rule(
            strokeDash=[6, 4],
            strokeWidth=1.4
        ).encode(
            y="zero:Q"
        )

        chart = (
            line
            + points
            + zero_line
        ).properties(
            height=400,
            title=(
                "Cumulative InSAR Displacement at "
                f"{selected_piezometer}"
            )
        ).interactive()

        st.altair_chart(
            chart,
            use_container_width=True
        )

        st.caption(
            f"Selected location: {selected_piezometer} | "
            f"X = {piezo_x:.0f} m | "
            f"Y = {piezo_y:.0f} m"
        )

        st.caption(
            "Values are satellite-derived estimated cumulative "
            "vertical displacement sampled at the piezometer "
            "location; they are not groundwater-level measurements."
        )


        # ----------------------------------------------------
        # REGIONAL SUMMARY — SECONDARY
        # ----------------------------------------------------

        with st.expander(
            "Regional InSAR summary"
        ):

            regional_ts = load_insar_timeseries_table(
                model_year
            )

            st.caption(
                "Regional statistics over the common "
                "ALPRIFT–InSAR valid area."
            )

            st.dataframe(
                regional_ts[
                    [
                        "date",
                        "mean_mm",
                        "median_mm",
                        "min_mm",
                        "max_mm",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================


    # ========================================================
    # LEGACY VALIDATION BLOCK REMOVED
    # ========================================================
    #
    # The previous Basic-SVI validation block was replaced by
    # the official Reference2018 annual ALPRIFT–InSAR comparison
    # displayed above.
    #


# TAB 2 — SCENARIO / DSS
# ============================================================

with dss_tab:

    st.caption(
        "سامانه تعاملی تحلیل سناریوهای آسیب‌پذیری فرونشست ـ دشت شبستر"
    )


    # ============================================================
    # SCENARIO TYPE
    # ============================================================

    scenario_type = "Combined - سناریوی ترکیبی"

    # SCENARIO_METRIC_LTR_FIX
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            direction: ltr !important;
            unicode-bidi: isolate !important;
        }

        [data-testid="stMetricDelta"] {
            direction: ltr !important;
            unicode-bidi: isolate !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.caption(
        "Configure one integrated ALPRIFT what-if scenario "
        "using groundwater-level decline (T), pumping (P), "
        "and rainfall/recharge (R)."
    )


    # ============================================================
    # COMBINED SCENARIO INPUTS
    # ============================================================

    st.subheader(
        "Combined Scenario"
    )

    st.caption(
        "Define changes in groundwater-level decline (T), "
        "groundwater pumping (P), and rainfall/recharge (R). "
        "All other ALPRIFT components remain at baseline values."
    )

    # ============================================================
    # T SCENARIO
    # ============================================================

    if scenario_type.startswith("T"):

        st.subheader(
            "سناریوی افت سطح آب زیرزمینی (T)"
        )

        st.info(
            "T مثبت نشان‌دهنده افت سطح آب و "
            "T منفی نشان‌دهنده بازیابی یا افزایش سطح آب است."
        )


        col1, col2, col3, col4 = st.columns(4)


        # --------------------------------------------------------
        # Year
        # --------------------------------------------------------

        with col1:

            years = list(
                range(1389, 1404)
            )

            year = st.selectbox(
                "سال",
                options=years,
                index=years.index(1400),
                key="t_year"
            )


        try:

            piezo_gdf = load_piezometers(
                year
            )

        except Exception as e:

            st.error(str(e))
            st.stop()


        piezometers = sorted(
            piezo_gdf[
                "Piezometer"
            ].unique()
        )


        # --------------------------------------------------------
        # Piezometer
        # --------------------------------------------------------

        with col2:

            default_index = (
                piezometers.index("P04")
                if "P04" in piezometers
                else 0
            )

            piezometer = st.selectbox(
                "Piezometer",
                options=piezometers,
                index=default_index,
                key="t_piezometer"
            )


        selected = piezo_gdf.loc[
            piezo_gdf["Piezometer"]
            ==
            piezometer
        ].iloc[0]


        current_t = float(
            selected["T_myr"]
        )


        # --------------------------------------------------------
        # Baseline T
        # --------------------------------------------------------

        with col3:

            st.metric(
                "Baseline T (m/year)",
                f"{current_t:.3f}"
            )


        # --------------------------------------------------------
        # Scenario T
        # --------------------------------------------------------

        with col4:

            new_t = st.number_input(
                "Scenario T (m/year)",
                min_value=-20.0,
                max_value=20.0,
                value=float(current_t),
                step=0.10,
                format="%.3f",
                key=f"T_value_{year}_{piezometer}"
            )


        # --------------------------------------------------------
        # Point details
        # --------------------------------------------------------

        with st.expander(
            "Piezometer Information"
        ):

            a, b, c, d = st.columns(4)

            a.write(
                f"**پیزومتر:** {piezometer}"
            )

            b.write(
                f"**UTM X:** {selected.get('utmx', '-')}"
            )

            c.write(
                f"**UTM Y:** {selected.get('utmy', '-')}"
            )

            d.write(
                f"**وضعیت:** {selected.get('Hydro_Stat', '-')}"
            )


        save_outputs = st.checkbox(
            "Save outputs",
            value=True,
            key="t_save"
        )


        run_button = st.button(
            "▶ اجرای سناریوی T",
            type="primary",
            width="stretch"
        )


        if run_button:

            try:

                with st.spinner(
                    "در حال اجرای IDW و محاسبه سناریوی T..."
                ):

                    result = run_t_scenario(
                        year=year,
                        piezometer=piezometer,
                        new_t=new_t,
                        save_outputs=save_outputs
                    )


                st.session_state[
                    "scenario_result"
                ] = result

                st.session_state[
                    "scenario_result_type"
                ] = "T"


                st.success(
                    "سناریوی T با موفقیت اجرا شد."
                )


            except Exception as e:

                st.exception(e)


    # ============================================================
    # PUMPING SCENARIO
    # ============================================================

    elif scenario_type.startswith("P"):

        st.subheader(
            "سناریوی پمپاژ آب زیرزمینی (P)"
        )

        st.info(
            "داده پایه پمپاژ مربوط به سال ۱۳۹۸ است. "
            "سناریو مقدار فیزیکی پمپاژ یک Thiessen Zone را تغییر می‌دهد "
            "و سپس P Rate و SVI مجدداً محاسبه می‌شوند."
        )


        try:

            pumping_gdf = load_pumping_zones()

        except Exception as e:

            st.error(str(e))
            st.stop()


        zones = sorted(
            pumping_gdf[
                "TARGET_FID"
            ].unique()
        )


        c1, c2, c3, c4 = st.columns(4)


        # --------------------------------------------------------
        # SVI year
        # --------------------------------------------------------

        with c1:

            years = list(
                range(1389, 1404)
            )

            p_year = st.selectbox(
                "Baseline Year",
                options=years,
                index=years.index(1400),
                key="p_year"
            )


        # --------------------------------------------------------
        # Zone
        # --------------------------------------------------------

        with c2:

            zone_id = st.selectbox(
                "Pumping Zone",
                options=zones,
                index=0,
                key="p_zone"
            )


        zone_row = pumping_gdf.loc[
            pumping_gdf["TARGET_FID"]
            ==
            zone_id
        ].iloc[0]


        baseline_p = float(
            zone_row["P_cm_yr"]
        )

        baseline_rate = int(
            zone_row["P_Rate"]
        )


        # --------------------------------------------------------
        # Baseline P
        # --------------------------------------------------------

        with c3:

            st.metric(
                "Baseline Pumping (cm/year)",
                f"{baseline_p:.3f}"
            )

            st.caption(
                f"P Rate پایه = {baseline_rate}"
            )


        # --------------------------------------------------------
        # Direct Scenario P
        # --------------------------------------------------------

        with c4:

            scenario_p = st.number_input(
                "Scenario P (cm/year)",
                min_value=0.0,
                value=float(baseline_p),
                step=0.10,
                format="%.3f",
                key=(
                    f"single_p_direct_"
                    f"{p_year}_"
                    f"{zone_id}"
                )
            )


        st.write(
            f"""
            **P سناریو:**  
            {baseline_p:.3f} → **{scenario_p:.3f} cm/year**
            """
        )


        # --------------------------------------------------------
        # Zone details
        # --------------------------------------------------------

        with st.expander(
            "اطلاعات Zone انتخاب‌شده"
        ):

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Zone ID",
                zone_id
            )

            q2.metric(
                "تخلیه پایه",
                f"{float(zone_row['takhlieh_s']):,.0f}"
            )

            q3.metric(
                "pompaj",
                f"{float(zone_row['pompaj']):.4f}"
            )

            q4.metric(
                "P Rate",
                baseline_rate
            )


        save_outputs = st.checkbox(
            "Save outputs",
            value=True,
            key="p_save"
        )


        run_button = st.button(
            "▶ اجرای سناریوی پمپاژ",
            type="primary",
            width="stretch"
        )


        if run_button:

            try:

                with st.spinner(
                    "در حال محاسبه سناریوی پمپاژ..."
                ):

                    result = run_p_scenario(
                        zone_id=zone_id,
                        scenario_p=scenario_p,
                        year=p_year,
                        save_outputs=save_outputs
                    )


                st.session_state[
                    "scenario_result"
                ] = result

                st.session_state[
                    "scenario_result_type"
                ] = "P"


                st.success(
                    "سناریوی پمپاژ با موفقیت اجرا شد."
                )


            except Exception as e:

                st.exception(e)



    # ============================================================
    # RAINFALL / RECHARGE SCENARIO
    # ============================================================

    elif scenario_type.startswith("R"):

        st.subheader(
            "سناریوی بارش و تغذیه آبخوان (R)"
        )

        st.info(
            "در این ماژول فقط بارش سالانه تغییر می‌کند. "
            "Slope Rank و Permeability Rank ثابت می‌مانند. "
            "تغییر R مطابق همان طبقات Piscopo و ALPRIFT محاسبه می‌شود."
        )


        r1, r2, r3, r4 = st.columns(4)


        # --------------------------------------------------------
        # SVI year
        # --------------------------------------------------------

        with r1:

            years = list(
                range(1389, 1404)
            )

            r_year = st.selectbox(
                "Baseline Year",
                options=years,
                index=years.index(1400),
                key="r_year"
            )


        # --------------------------------------------------------
        # Baseline rainfall
        # --------------------------------------------------------

        baseline_rainfall = 304.4

        with r2:

            st.metric(
                "Baseline Rainfall (mm/year)",
                f"{baseline_rainfall:.1f}"
            )


        # --------------------------------------------------------
        # Scenario rainfall
        # --------------------------------------------------------

        with r3:

            scenario_rainfall = st.number_input(
                "Scenario Rainfall (mm/year)",
                min_value=0.0,
                max_value=2000.0,
                value=550.0,
                step=10.0,
                format="%.1f",
                key="r_rainfall"
            )


        # --------------------------------------------------------
        # Rain rank preview
        # --------------------------------------------------------

        def rain_rank_preview(x):

            if x < 500:
                return 1

            elif x < 700:
                return 2

            elif x <= 850:
                return 3

            return 4


        baseline_rain_rank = rain_rank_preview(
            baseline_rainfall
        )

        scenario_rain_rank = rain_rank_preview(
            scenario_rainfall
        )


        with r4:

            st.metric(
                "Rain Rank",
                f"{baseline_rain_rank} → {scenario_rain_rank}"
            )


        # --------------------------------------------------------
        # Explain threshold behavior
        # --------------------------------------------------------

        if (
            baseline_rain_rank
            ==
            scenario_rain_rank
        ):

            st.warning(
                "بارش تغییر کرده است، اما هنوز در همان طبقه Piscopo قرار دارد؛ "
                "بنابراین انتظار نمی‌رود R Rate تغییر کند."
            )

        else:

            st.write(
                f"""
                **سناریوی بارش:**  
                {baseline_rainfall:.1f} → **{scenario_rainfall:.1f} mm/year**

                **Rain Rank:**  
                {baseline_rain_rank} → **{scenario_rain_rank}**
                """
            )


        with st.expander(
            "طبقات Rain Rank"
        ):

            st.write(
                """
                - کمتر از 500 mm/year → Rank 1
                - 500 تا کمتر از 700 → Rank 2
                - 700 تا 850 → Rank 3
                - بیشتر از 850 → Rank 4
                """
            )


        save_outputs = st.checkbox(
            "Save outputs",
            value=True,
            key="r_save"
        )


        run_button = st.button(
            "▶ اجرای سناریوی بارش",
            type="primary",
            width="stretch"
        )


        if run_button:

            try:

                with st.spinner(
                    "در حال محاسبه سناریوی بارش و Recharge..."
                ):

                    result = run_r_scenario(
                        rainfall_mm=scenario_rainfall,
                        year=r_year,
                        save_outputs=save_outputs
                    )


                st.session_state[
                    "scenario_result"
                ] = result

                st.session_state[
                    "scenario_result_type"
                ] = "R"


                st.success(
                    "سناریوی R با موفقیت اجرا شد."
                )


            except Exception as e:

                st.exception(e)




    # ============================================================
    # COMBINED T + P + R SCENARIO
    # ============================================================

    else:

        st.subheader(
            "Integrated T + P + R Scenario"
        )

        st.info(
            "در این بخش می‌توان افت سطح آب، پمپاژ و بارش "
            "را به‌صورت هم‌زمان تعریف کرد."
        )


        # ========================================================
        # SHARED YEAR
        # ========================================================

        years = list(
            range(1389, 1404)
        )

        combined_year = st.selectbox(
            "Baseline Year",
            options=years,
            index=years.index(1400),
            key="combined_year"
        )


        # ========================================================
        # LOAD T AND P SOURCE DATA
        # ========================================================

        try:

            combined_piezo_gdf = load_piezometers(
                combined_year
            )

            combined_pumping_gdf = load_pumping_zones()

        except Exception as e:

            st.error(str(e))
            st.stop()


        # ========================================================
        # T INPUTS
        # ========================================================
        # INLINE T AND P LOCATION MAPS

        st.markdown(
            "### T — Annual Groundwater-Level Change"
        )

        st.info(
            """
**T is the annual change in groundwater level, not the groundwater level itself.**

**+ Positive value:** groundwater-level decline  
**− Negative value:** groundwater-level recovery / rise  
**0:** no annual groundwater-level change  

**Unit:** m/year
            """
        )

        combined_piezometers = sorted(
            combined_piezo_gdf[
                "Piezometer"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        default_piezometers = (
            ["P04"]
            if "P04" in combined_piezometers
            else [combined_piezometers[0]]
        )

        combined_selected_piezometers = st.multiselect(
            "Selected Piezometers",
            options=combined_piezometers,
            default=default_piezometers,
            key="combined_multi_piezometers"
        )

        combined_selected_piezometers = [
            str(value)
            for value
            in combined_selected_piezometers
        ]

        st.caption(
            "Selected piezometer annual groundwater-level changes "
            "are modified first; the T surface is then reconstructed "
            "using IDW."
        )

        t_left, t_right = st.columns(
            [1.25, 0.75],
            gap="large"
        )


        # ----------------------------------------------------
        # LEFT — EDITABLE T TABLE
        # ----------------------------------------------------

        with t_left:

            t_editor_rows = []

            for piezo_id in combined_selected_piezometers:

                row = combined_piezo_gdf.loc[
                    combined_piezo_gdf[
                        "Piezometer"
                    ].astype(str)
                    ==
                    piezo_id
                ].iloc[0]

                baseline_value = float(
                    row["T_myr"]
                )

                t_editor_rows.append(
                    {
                        "Piezometer":
                            piezo_id,

                        "Baseline T (m/year)":
                            baseline_value,

                        "Scenario T (m/year)":
                            baseline_value,
                    }
                )


            combined_t_input_df = pd.DataFrame(
                t_editor_rows,
                columns=[
                    "Piezometer",
                    "Baseline T (m/year)",
                    "Scenario T (m/year)",
                ]
            )


            if combined_selected_piezometers:

                selection_key = "_".join(
                    combined_selected_piezometers
                )

                combined_t_editor = st.data_editor(
                    combined_t_input_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(
                        245,
                        48
                        +
                        len(
                            combined_t_input_df
                        )
                        * 36
                    ),
                    disabled=[
                        "Piezometer",
                        "Baseline T (m/year)",
                    ],
                    column_config={

                        "Piezometer":
                            st.column_config.TextColumn(
                                "Piezometer",
                                width="small"
                            ),

                        "Baseline T (m/year)":
                            st.column_config.NumberColumn(
                                "Baseline Annual Change (m/year)",
                                format="%.3f",
                                help=(
                                    "Positive = groundwater decline; "
                                    "negative = groundwater recovery/rise."
                                )
                            ),

                        "Scenario T (m/year)":
                            st.column_config.NumberColumn(
                                "Scenario Annual Change (m/year)",
                                min_value=-20.0,
                                max_value=20.0,
                                step=0.10,
                                format="%.3f",
                                required=True,
                                help=(
                                    "Enter + for groundwater decline, "
                                    "− for groundwater recovery/rise."
                                )
                            ),
                    },
                    key=(
                        f"combined_t_multi_editor_"
                        f"{combined_year}_"
                        f"{selection_key}"
                    )
                )

            else:

                combined_t_editor = (
                    combined_t_input_df.copy()
                )

                st.warning(
                    "Select at least one piezometer."
                )


        # ----------------------------------------------------
        # RIGHT — SELECTED PIEZOMETER MAP
        # ----------------------------------------------------

        with t_right:

            t_map_gdf = (
                combined_piezo_gdf.copy()
            )

            if (
                t_map_gdf.crs is not None
                and
                combined_pumping_gdf.crs is not None
                and
                t_map_gdf.crs
                !=
                combined_pumping_gdf.crs
            ):

                t_map_gdf = (
                    t_map_gdf.to_crs(
                        combined_pumping_gdf.crs
                    )
                )


            t_map_selected = (
                t_map_gdf.loc[
                    t_map_gdf[
                        "Piezometer"
                    ].astype(str)
                    .isin(
                        combined_selected_piezometers
                    )
                ]
            )


            fig_t_inline, ax_t_inline = plt.subplots(
                figsize=(5.0, 3.0)
            )


            combined_pumping_gdf.boundary.plot(
                ax=ax_t_inline,
                color="0.78",
                linewidth=0.55
            )


            t_map_gdf.plot(
                ax=ax_t_inline,
                color="0.60",
                markersize=12,
                alpha=0.65
            )


            if not t_map_selected.empty:

                t_map_selected.plot(
                    ax=ax_t_inline,
                    color="red",
                    edgecolor="black",
                    linewidth=0.6,
                    markersize=55,
                    zorder=5
                )


                for _, row in (
                    t_map_selected.iterrows()
                ):

                    ax_t_inline.annotate(
                        str(
                            row[
                                "Piezometer"
                            ]
                        ),
                        (
                            row.geometry.x,
                            row.geometry.y
                        ),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold"
                    )


            ax_t_inline.set_axis_off()

            fig_t_inline.tight_layout()

            st.pyplot(
                fig_t_inline,
                use_container_width=True
            )

            st.caption(
                f"T — Selected Piezometers "
                f"({len(combined_selected_piezometers)})"
            )

            plt.close(
                fig_t_inline
            )


        # ----------------------------------------------------
        # Convert T table to engine dictionary
        # ----------------------------------------------------

        combined_t_changes = {}

        t_editor_valid = True

        for _, row in combined_t_editor.iterrows():

            piezo_id = str(
                row["Piezometer"]
            )

            scenario_value = pd.to_numeric(
                row[
                    "Scenario T (m/year)"
                ],
                errors="coerce"
            )

            if pd.isna(
                scenario_value
            ):

                t_editor_valid = False

            else:

                combined_t_changes[
                    piezo_id
                ] = float(
                    scenario_value
                )


        combined_t_selection_valid = (
            len(
                combined_selected_piezometers
            ) > 0
            and
            t_editor_valid
            and
            len(
                combined_t_changes
            )
            ==
            len(
                combined_selected_piezometers
            )
        )


        if combined_selected_piezometers:

            combined_piezometer = (
                combined_selected_piezometers[0]
            )

        else:

            combined_piezometer = (
                combined_piezometers[0]
            )


        combined_piezo_row = (
            combined_piezo_gdf.loc[
                combined_piezo_gdf[
                    "Piezometer"
                ].astype(str)
                ==
                combined_piezometer
            ]
            .iloc[0]
        )


        combined_baseline_t = float(
            combined_piezo_row[
                "T_myr"
            ]
        )


        combined_t_new = float(
            combined_t_changes.get(
                combined_piezometer,
                combined_baseline_t
            )
        )


        mt1, mt2, mt3 = st.columns(3)

        mt1.metric(
            "Selected Piezometers",
            len(
                combined_selected_piezometers
            )
        )

        mt2.metric(
            "Modified T Inputs",
            int(
                sum(
                    abs(
                        float(row[
                            "Scenario T (m/year)"
                        ])
                        -
                        float(row[
                            "Baseline T (m/year)"
                        ])
                    )
                    > 1e-12

                    for _, row
                    in combined_t_editor.iterrows()
                )
            )
        )

        mt3.metric(
            "IDW Runs",
            "1"
        )


        # ========================================================
        # P INPUTS
        # ========================================================
        # DIRECT PHYSICAL P SCENARIO

        st.markdown(
            "### P — Groundwater Pumping"
        )


        combined_zones = sorted(
            combined_pumping_gdf[
                "TARGET_FID"
            ]
            .astype(int)
            .unique()
            .tolist()
        )


        combined_selected_zones = st.multiselect(
            "Selected Pumping Zones",
            options=combined_zones,
            default=[
                combined_zones[0]
            ],
            key="combined_multi_pumping_zones"
        )


        combined_selected_zones = [
            int(value)
            for value
            in combined_selected_zones
        ]


        st.caption(
            "Enter the final Scenario P value directly "
            "for each selected pumping zone."
        )


        p_left, p_right = st.columns(
            [1.25, 0.75],
            gap="large"
        )


        # ----------------------------------------------------
        # LEFT — DIRECT P TABLE
        # ----------------------------------------------------

        with p_left:

            p_editor_rows = []


            for zone_id in combined_selected_zones:

                zone_row = (
                    combined_pumping_gdf.loc[
                        combined_pumping_gdf[
                            "TARGET_FID"
                        ].astype(int)
                        ==
                        zone_id
                    ]
                    .iloc[0]
                )


                baseline_value = float(
                    zone_row[
                        "P_cm_yr"
                    ]
                )


                p_editor_rows.append(
                    {
                        "Zone":
                            zone_id,

                        "Baseline P (cm/year)":
                            baseline_value,

                        "Scenario P (cm/year)":
                            baseline_value,
                    }
                )


            combined_p_input_df = pd.DataFrame(
                p_editor_rows,
                columns=[
                    "Zone",
                    "Baseline P (cm/year)",
                    "Scenario P (cm/year)",
                ]
            )


            if combined_selected_zones:

                zone_key = "_".join(
                    str(value)
                    for value
                    in combined_selected_zones
                )


                combined_p_editor = st.data_editor(
                    combined_p_input_df,
                    hide_index=True,
                    use_container_width=True,
                    height=min(
                        245,
                        48
                        +
                        len(
                            combined_p_input_df
                        )
                        * 36
                    ),
                    disabled=[
                        "Zone",
                        "Baseline P (cm/year)",
                    ],
                    column_config={

                        "Zone":
                            st.column_config.NumberColumn(
                                "Zone",
                                format="%d",
                                width="small"
                            ),

                        "Baseline P (cm/year)":
                            st.column_config.NumberColumn(
                                "Baseline P",
                                format="%.3f"
                            ),

                        "Scenario P (cm/year)":
                            st.column_config.NumberColumn(
                                "Scenario P",
                                min_value=0.0,
                                step=0.10,
                                format="%.3f",
                                required=True
                            ),
                    },
                    key=(
                        f"combined_p_direct_editor_"
                        f"{combined_year}_"
                        f"{zone_key}"
                    )
                )


            else:

                combined_p_editor = (
                    combined_p_input_df.copy()
                )

                st.warning(
                    "Select at least one pumping zone."
                )


        # ----------------------------------------------------
        # RIGHT — SELECTED ZONE MAP
        # ----------------------------------------------------

        with p_right:

            p_map_selected = (
                combined_pumping_gdf.loc[
                    combined_pumping_gdf[
                        "TARGET_FID"
                    ].astype(int)
                    .isin(
                        combined_selected_zones
                    )
                ]
            )


            fig_p_inline, ax_p_inline = plt.subplots(
                figsize=(5.0, 3.0)
            )


            combined_pumping_gdf.boundary.plot(
                ax=ax_p_inline,
                color="0.65",
                linewidth=0.65
            )


            if not p_map_selected.empty:

                p_map_selected.plot(
                    ax=ax_p_inline,
                    color="red",
                    alpha=0.50,
                    edgecolor="black",
                    linewidth=0.9
                )


                for _, row in p_map_selected.iterrows():

                    rp = (
                        row.geometry
                        .representative_point()
                    )


                    ax_p_inline.annotate(
                        f"Z{int(row['TARGET_FID'])}",
                        (
                            rp.x,
                            rp.y
                        ),
                        fontsize=8,
                        fontweight="bold",
                        ha="center",
                        va="center"
                    )


            ax_p_inline.set_axis_off()

            fig_p_inline.tight_layout()


            st.pyplot(
                fig_p_inline,
                use_container_width=True
            )


            st.caption(
                f"P — Selected Pumping Zones "
                f"({len(combined_selected_zones)})"
            )


            plt.close(
                fig_p_inline
            )


        # ----------------------------------------------------
        # DIRECT P DICTIONARY SENT TO ENGINE
        # ----------------------------------------------------

        combined_p_scenarios = {}

        p_editor_valid = True


        for _, row in combined_p_editor.iterrows():

            zone_id = int(
                row[
                    "Zone"
                ]
            )


            scenario_value = pd.to_numeric(
                row[
                    "Scenario P (cm/year)"
                ],
                errors="coerce"
            )


            if pd.isna(
                scenario_value
            ):

                p_editor_valid = False
                continue


            scenario_value = float(
                scenario_value
            )


            if (
                not np.isfinite(
                    scenario_value
                )
                or
                scenario_value < 0
            ):

                p_editor_valid = False
                continue


            combined_p_scenarios[
                zone_id
            ] = scenario_value


        combined_p_selection_valid = (
            len(
                combined_selected_zones
            ) > 0
            and
            p_editor_valid
            and
            len(
                combined_p_scenarios
            )
            ==
            len(
                combined_selected_zones
            )
        )


        # ----------------------------------------------------
        # PRIMARY ZONE — DISPLAY ONLY
        # ----------------------------------------------------

        if combined_selected_zones:

            combined_zone = int(
                combined_selected_zones[0]
            )

        else:

            combined_zone = int(
                combined_zones[0]
            )


        combined_zone_row = (
            combined_pumping_gdf.loc[
                combined_pumping_gdf[
                    "TARGET_FID"
                ].astype(int)
                ==
                combined_zone
            ]
            .iloc[0]
        )


        combined_baseline_p = float(
            combined_zone_row[
                "P_cm_yr"
            ]
        )


        combined_scenario_p = float(
            combined_p_scenarios.get(
                combined_zone,
                combined_baseline_p
            )
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        pm1, pm2, pm3 = st.columns(3)


        pm1.metric(
            "Selected Zones",
            len(
                combined_selected_zones
            )
        )


        modified_zone_count = 0


        for _, row in combined_p_editor.iterrows():

            baseline_value = float(
                row[
                    "Baseline P (cm/year)"
                ]
            )


            scenario_value = pd.to_numeric(
                row[
                    "Scenario P (cm/year)"
                ],
                errors="coerce"
            )


            if (
                not pd.isna(
                    scenario_value
                )
                and
                abs(
                    float(
                        scenario_value
                    )
                    -
                    baseline_value
                )
                > 1e-12
            ):

                modified_zone_count += 1


        pm2.metric(
            "Modified Zones",
            modified_zone_count
        )


        pm3.metric(
            "P Rasterizations",
            "1"
        )


        # ========================================================
        # R / RAINFALL INPUT
        # ========================================================
        # RESTORED COMBINED RAINFALL INPUT

        st.markdown(
            "### R — Rainfall / Recharge"
        )


        # ----------------------------------------------------
        # Obtain official baseline rainfall from R engine
        # ----------------------------------------------------

        @st.cache_data(
            show_spinner=False
        )
        def _get_combined_baseline_rainfall(
            year
        ):

            probe = run_r_scenario(
                year=int(year),
                rainfall_mm=0.0,
                save_outputs=False
            )

            return float(
                probe[
                    "summary"
                ][
                    "baseline_rainfall_mm"
                ]
            )


        combined_baseline_rainfall = (
            _get_combined_baseline_rainfall(
                combined_year
            )
        )


        r1, r2 = st.columns(
            2
        )


        with r1:

            st.metric(
                "Baseline Rainfall (mm/year)",
                f"{combined_baseline_rainfall:.1f}"
            )


        with r2:

            combined_rainfall = st.number_input(
                "Scenario Rainfall (mm/year)",
                min_value=0.0,
                max_value=2000.0,
                value=float(
                    combined_baseline_rainfall
                ),
                step=10.0,
                format="%.1f",
                key=(
                    f"combined_rainfall_"
                    f"{combined_year}"
                )
            )


        # ----------------------------------------------------
        # Rain Rank preview
        # ----------------------------------------------------

        def _combined_rain_rank(
            rainfall_value
        ):

            rainfall_value = float(
                rainfall_value
            )

            if rainfall_value < 500:
                return 1

            elif rainfall_value < 700:
                return 2

            elif rainfall_value <= 850:
                return 3

            else:
                return 4


        combined_baseline_rain_rank = (
            _combined_rain_rank(
                combined_baseline_rainfall
            )
        )

        combined_scenario_rain_rank = (
            _combined_rain_rank(
                combined_rainfall
            )
        )


        st.caption(
            f"Rain Rank: "
            f"{combined_baseline_rain_rank} → "
            f"{combined_scenario_rain_rank}. "
            "The final R rating is spatial because Rain Rank is "
            "combined with fixed slope and permeability ranks."
        )


        st.divider()


        # ----------------------------------------------------
        # CURRENT COMBINED INPUT SIGNATURE
        # ----------------------------------------------------
        # Prevent stale scenario results from being shown after
        # T / P / R inputs are changed.

        combined_input_signature = (
            int(combined_year),

            tuple(
                sorted(
                    (
                        str(key),
                        round(float(value), 9)
                    )
                    for key, value
                    in combined_t_changes.items()
                )
            ),

            tuple(
                sorted(
                    (
                        int(key),
                        round(float(value), 9)
                    )
                    for key, value
                    in combined_p_scenarios.items()
                )
            ),

            round(
                float(combined_rainfall),
                9
            ),
        )

        last_executed_signature = (
            st.session_state.get(
                "combined_last_executed_signature"
            )
        )

        if (
            last_executed_signature is not None
            and
            combined_input_signature
            !=
            last_executed_signature
            and
            st.session_state.get(
                "scenario_result_type"
            )
            ==
            "Combined"
        ):

            st.session_state.pop(
                "scenario_result",
                None
            )

            st.session_state.pop(
                "scenario_result_type",
                None
            )

            st.session_state[
                "combined_result_invalidated"
            ] = True


        if st.session_state.pop(
            "combined_result_invalidated",
            False
        ):

            st.warning(
                "Scenario inputs have changed since the last run. "
                "The previous results were cleared. "
                "Run the combined scenario again to calculate "
                "results for the current inputs."
            )


        combined_run_button = st.button(
            "▶ Run Combined Scenario",
            type="primary",
            width="stretch",
            disabled=(
                not (
                    combined_t_selection_valid
                    and
                    combined_p_selection_valid
                )
            ),
            key="combined_run"
        )


        if combined_run_button:

            try:

                with st.spinner(
                    "در حال Run Combined Scenario T + P + R..."
                ):

                    result = run_combined_scenario(
                        year=combined_year,

                        t_piezometer=combined_t_changes,
                        t_new=None,

                        p_zone_id=combined_p_scenarios,
                        p_scenario=None,

                        rainfall_mm=combined_rainfall,

                        # During UI execution tests we do not need
                        # to write duplicate scenario files.
                        save_outputs=False
                    )


                st.session_state[
                    "scenario_result"
                ] = result

                st.session_state[
                    "scenario_result_type"
                ] = "Combined"

                st.session_state[
                    "combined_last_executed_signature"
                ] = combined_input_signature


                st.success(
                    "Combined scenario completed successfully."
                )


            except Exception as e:

                st.exception(e)




    # ============================================================
    # SHOW RESULTS ONLY FOR CURRENT SCENARIO TYPE
    # ============================================================

    if scenario_type.startswith("T"):

        current_code = "T"

    elif scenario_type.startswith("P"):

        current_code = "P"

    elif scenario_type.startswith("R"):

        current_code = "R"

    else:

        current_code = "Combined"


    if (
        "scenario_result" in st.session_state
        and
        st.session_state.get(
            "scenario_result_type"
        ) == current_code
    ):

        result = st.session_state[
            "scenario_result"
        ]

        summary = result[
            "summary"
        ]

        transitions = (
            result[
                "transitions"
            ].copy()
        )


        st.divider()

        st.header(
            "Scenario Results"
        )


        # ========================================================
        # METRICS
        # ========================================================

        # --------------------------------------------------------
        # Main scenario metrics
        # --------------------------------------------------------

        m1, m2, m3 = st.columns(3)


        with m1:

            st.metric(
                "Baseline Mean SVI",
                f"{summary['baseline_mean_svi']:.3f}"
            )


        with m2:

            st.metric(
                "Scenario Mean SVI",
                f"{summary['scenario_mean_svi']:.3f}"
            )


        with m3:

            st.metric(
                "Affected Area",
                f"{summary['changed_area_km2']:.2f} km²"
            )


        d1, d2, d3 = st.columns(3)


        with d1:

            st.metric(
                "Mean ΔSVI",
                f"{summary['mean_delta_svi']:+.3f}"
            )


        with d2:

            st.metric(
                "Maximum SVI Increase",
                f"{summary['max_delta_svi']:+.2f}"
            )


        with d3:

            st.metric(
                "Maximum SVI Decrease",
                f"{summary['min_delta_svi']:+.2f}"
            )


        # ========================================================
        # DESCRIPTION
        # ========================================================

        if current_code == "T":

            st.write(
                f"""
                **سناریو:** پیزومتر **{summary['piezometer']}**،
                سال **{summary['year']}**،
                T از **{summary['baseline_t']:.3f}**
                به **{summary['scenario_t']:.3f} m/year**
                تغییر داده شد.
                """
            )

        elif current_code == "P":

            st.write(
                f"""
                **سناریو:** Pumping Zone **{summary['zone_id']}**،
                P از **{summary['baseline_p_cm_yr']:.3f}**
                به **{summary['scenario_p_cm_yr']:.3f} cm/year**
                تغییر یافت.

                P Rate:
                **{summary['baseline_p_rate']} → {summary['scenario_p_rate']}**
                """
            )

        elif current_code == "R":

            st.write(
                f"""
                **سناریو:** بارش سالانه از
                **{summary['baseline_rainfall_mm']:.1f}**
                به **{summary['scenario_rainfall_mm']:.1f} mm/year**
                تغییر یافت.

                Rain Rank:
                **{summary['baseline_rain_rank']} → {summary['scenario_rain_rank']}**
                """
            )


        else:

            t_executed_changes = (
                summary.get(
                    "t_changes",
                    {
                        summary["t_piezometer"]:
                            summary["t_scenario"]
                    }
                )
            )

            t_executed_baselines = (
                summary.get(
                    "t_baseline_by_piezometer",
                    {
                        summary["t_piezometer"]:
                            summary["t_baseline"]
                    }
                )
            )

            t_executed_text = "\n\n".join(
                [
                    (
                        f"- **{piezo_id}**: "
                        f"{float(t_executed_baselines.get(piezo_id, float('nan'))):.3f}"
                        f" → "
                        f"{float(scenario_value):.3f} m/year"
                    )
                    for piezo_id, scenario_value
                    in t_executed_changes.items()
                ]
            )

            p_executed_details = (
                summary.get(
                    "p_zone_details",
                    {}
                )
            )


            if p_executed_details:

                p_executed_text = "\n\n".join(
                    [
                        (
                            f"- **Zone {zone_id}**: "
                            f"{float(detail['baseline_p_cm_yr']):.3f}"
                            f" → "
                            f"{float(detail['scenario_p_cm_yr']):.3f} cm/year "
                        )
                        for zone_id, detail
                        in p_executed_details.items()
                    ]
                )

            else:

                p_executed_text = (
                    f"- **Zone {summary['p_zone_id']}**: "
                    f"{summary['p_baseline_cm_yr']:.3f}"
                    f" → "
                    f"{summary['p_scenario_cm_yr']:.3f} cm/year "
                )


            st.markdown(
                "### Executed Combined Scenario"
            )

            result_t, result_p, result_r = st.columns(
                3,
                gap="large"
            )

            with result_t:

                st.markdown(
                    "#### T — Annual Groundwater-Level Change"
                )

                st.caption(
                    "+ = groundwater-level decline | "
                    "− = groundwater-level recovery / rise"
                )

                st.markdown(
                    f"**Selected piezometers:** "
                    f"{summary.get('t_change_count', 1)}"
                )

                st.markdown(
                    t_executed_text
                )


            with result_p:

                st.markdown(
                    "#### P — Groundwater Pumping"
                )

                st.markdown(
                    f"**Selected pumping zones:** "
                    f"{summary.get('p_change_count', 1)}"
                )

                st.markdown(
                    p_executed_text
                )


            with result_r:

                st.markdown(
                    "#### R — Rainfall / Recharge"
                )

                st.markdown(
                    f"**Rainfall:**  "
                    f"{summary['baseline_rainfall_mm']:.1f} → "
                    f"{summary['rainfall_mm']:.1f} mm/year"
                )

                st.markdown(
                    f"**Rain Rank:**  "
                    f"{summary['baseline_rain_rank']} → "
                    f"{summary['scenario_rain_rank']}"
                )


        # ========================================================
        # MAP ARRAYS
        # ========================================================


        # COMBINED ALPRIFT RATING TRACEABILITY
        # ============================================================

        def _ui_t_rate(value):

            value = float(value)

            if value < 0.2:
                return 1
            if value < 0.5:
                return 2
            if value < 0.9:
                return 3
            if value < 1.4:
                return 4
            if value < 2.0:
                return 5
            if value < 2.7:
                return 6
            if value < 3.5:
                return 7
            if value < 4.4:
                return 8
            if value < 5.4:
                return 9

            return 10


        def _ui_p_rate(value):

            value = float(value)

            if value < 0.0001:
                return 1
            if value < 0.005:
                return 2
            if value < 0.01:
                return 3
            if value < 0.5:
                return 4
            if value < 1.0:
                return 5
            if value < 5.0:
                return 6
            if value < 20.0:
                return 7
            if value < 40.0:
                return 8
            if value <= 65.0:
                return 9

            return 10


        t_base_value = float(
            summary["t_baseline"]
        )

        t_scenario_value = float(
            summary["t_scenario"]
        )

        t_base_rate = _ui_t_rate(
            t_base_value
        )

        t_scenario_rate = _ui_t_rate(
            t_scenario_value
        )

        t_rate_delta = (
            t_scenario_rate
            - t_base_rate
        )

        t_weighted_source_effect = (
            5 * t_rate_delta
        )


        p_base_value = float(
            summary["p_baseline_cm_yr"]
        )

        p_scenario_value = float(
            summary["p_scenario_cm_yr"]
        )

        p_base_rate = _ui_p_rate(
            p_base_value
        )

        p_scenario_rate = _ui_p_rate(
            p_scenario_value
        )

        p_rate_delta = (
            p_scenario_rate
            - p_base_rate
        )

        p_weighted_zone_effect = (
            4 * p_rate_delta
        )


        rain_base_rank = int(
            summary["baseline_rain_rank"]
        )

        rain_scenario_rank = int(
            summary["scenario_rain_rank"]
        )


        st.markdown("---")

        st.markdown(
            "### Input → ALPRIFT Rating → Weighted Effect"
        )

        st.caption(
            "This panel traces how the physical scenario inputs "
            "enter the ALPRIFT scoring system."
        )


        trace_t, trace_p, trace_r = st.columns(
            3,
            gap="large"
        )


        # ------------------------------------------------------------
        # T
        # ------------------------------------------------------------

        with trace_t:

            st.markdown(
                "#### T — Annual Groundwater-Level Change"
            )


            def _multi_t_rating(
                value
            ):

                value = float(
                    value
                )

                if value < 0.2:
                    return 1
                elif value < 0.5:
                    return 2
                elif value < 0.9:
                    return 3
                elif value < 1.4:
                    return 4
                elif value < 2.0:
                    return 5
                elif value < 2.7:
                    return 6
                elif value < 3.5:
                    return 7
                elif value < 4.4:
                    return 8
                elif value < 5.4:
                    return 9
                else:
                    return 10


            t_trace_changes = (
                summary.get(
                    "t_changes",
                    {
                        summary["t_piezometer"]:
                            summary["t_scenario"]
                    }
                )
            )


            t_trace_baselines = (
                summary.get(
                    "t_baseline_by_piezometer",
                    {
                        summary["t_piezometer"]:
                            summary["t_baseline"]
                    }
                )
            )


            t_trace_rows = []

            for (
                piezo_id,
                scenario_value
            ) in t_trace_changes.items():

                baseline_value = float(
                    t_trace_baselines[
                        piezo_id
                    ]
                )

                scenario_value = float(
                    scenario_value
                )

                baseline_rating = (
                    _multi_t_rating(
                        baseline_value
                    )
                )

                scenario_rating = (
                    _multi_t_rating(
                        scenario_value
                    )
                )

                t_trace_rows.append(
                    {
                        "Piezometer":
                            piezo_id,

                        "Baseline T":
                            baseline_value,

                        "Scenario T":
                            scenario_value,

                        "Baseline Rating":
                            baseline_rating,

                        "Scenario Rating":
                            scenario_rating,

                        "Point Rating Δ":
                            (
                                scenario_rating
                                -
                                baseline_rating
                            ),

                        "W × Point Rating Δ":
                            5
                            *
                            (
                                scenario_rating
                                -
                                baseline_rating
                            ),
                    }
                )


            st.dataframe(
                pd.DataFrame(
                    t_trace_rows
                ),
                hide_index=True,
                use_container_width=True
            )


            st.caption(
                "These ratings describe the selected source "
                "piezometers. The actual ΔSVI_T raster is obtained "
                "after all selected T changes are applied together "
                "and one IDW surface is reconstructed."
            )


        with trace_p:

            st.markdown(
                "#### P — Groundwater Pumping"
            )


            p_trace_details = summary.get(
                "p_zone_details",
                {}
            )


            p_trace_rows = []


            for (
                zone_id,
                detail
            ) in p_trace_details.items():

                baseline_rate = int(
                    detail[
                        "baseline_p_rate"
                    ]
                )


                scenario_rate = int(
                    detail[
                        "scenario_p_rate"
                    ]
                )


                rating_delta = (
                    scenario_rate
                    -
                    baseline_rate
                )


                p_trace_rows.append(
                    {
                        "Zone":
                            int(
                                zone_id
                            ),

                        "Baseline P (cm/year)":
                            float(
                                detail[
                                    "baseline_p_cm_yr"
                                ]
                            ),

                        "Scenario P (cm/year)":
                            float(
                                detail[
                                    "scenario_p_cm_yr"
                                ]
                            ),

                        "Baseline Rating":
                            baseline_rate,

                        "Scenario Rating":
                            scenario_rate,

                        "Rating Δ":
                            rating_delta,

                        "4 × Rating Δ":
                            4
                            *
                            rating_delta,
                    }
                )


            if p_trace_rows:

                st.dataframe(
                    pd.DataFrame(
                        p_trace_rows
                    ),
                    hide_index=True,
                    use_container_width=True
                )


            st.caption(
                "Scenario P is entered directly in cm/year. "
                "Each physical P value is converted to its "
                "ALPRIFT P rating. All selected zones are "
                "rasterized together once."
            )


        with trace_r:

            st.markdown(
                "#### R — Rainfall / Recharge"
            )

            st.write(
                f"Rainfall: "
                f"**{float(summary['baseline_rainfall_mm']):.1f} → "
                f"{float(summary['rainfall_mm']):.1f} mm/year**"
            )

            st.write(
                f"Rain Rank: "
                f"**{rain_base_rank} → "
                f"{rain_scenario_rank}**"
            )

            st.write(
                "R Weight: **4**"
            )

            st.write(
                "**Final R Rating: spatially derived**"
            )

            st.caption(
                "Rain Rank is an intermediate input to the "
                "Piscopo-based recharge calculation. Slope Rank "
                "and Permeability Rank remain fixed, while rainfall "
                "changes. Therefore a single R-rating change is not "
                "reported here; the final spatial R contribution is "
                "included in the scenario SVI and ΔSVI maps."
            )


        st.info(
            "Interpretation: physical changes affect SVI only when "
            "they cross the relevant ALPRIFT rating thresholds. "
            "Therefore a large physical change can sometimes produce "
            "no SVI change, while crossing one or more rating classes "
            "can produce a discrete change in the index."
        )

        st.markdown("---")

        baseline_map = np.where(
            result["svi_mask"],
            result["baseline_svi"],
            np.nan
        )


        scenario_map = np.where(
            result["final_mask"],
            result["scenario_svi"],
            np.nan
        )


        delta_map = np.where(
            result["final_mask"],
            result["delta_svi"],
            np.nan
        )


        combined = np.concatenate([
            baseline_map[
                np.isfinite(
                    baseline_map
                )
            ],

            scenario_map[
                np.isfinite(
                    scenario_map
                )
            ]
        ])


        svi_min = float(
            np.min(
                combined
            )
        )

        svi_max = float(
            np.max(
                combined
            )
        )


        delta_values = (
            delta_map[
                np.isfinite(
                    delta_map
                )
            ]
        )


        delta_abs = float(
            np.max(
                np.abs(
                    delta_values
                )
            )
        )


        if delta_abs == 0:
            delta_abs = 1.0


        # ========================================================
        # MAP DISPLAY
        # ========================================================

        st.subheader(
            "Spatial Comparison"
        )


        map1, map2, map3 = st.columns(3)


        with map1:

            fig = raster_figure(
                baseline_map,
                "Baseline SVI",
                vmin=svi_min,
                vmax=svi_max,
                cmap="viridis"
            )

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)


        with map2:

            fig = raster_figure(
                scenario_map,
                "Scenario SVI",
                vmin=svi_min,
                vmax=svi_max,
                cmap="viridis"
            )

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)


        with map3:

            fig = raster_figure(
                delta_map,
                "ΔSVI",
                vmin=-delta_abs,
                vmax=delta_abs,
                cmap="RdBu_r"
            )

            st.pyplot(
                fig,
                width="stretch"
            )

            plt.close(fig)


        # ========================================================
        # CLASS TRANSITIONS
        # ========================================================



        # FACTOR-WISE SVI CHANGE DECOMPOSITION
        # ============================================================

        st.subheader(
            "Factor-wise SVI Change Decomposition"
        )

        st.caption(
            "This diagnostic panel separates the actual combined "
            "ALPRIFT ΔSVI into the T, P, and R components calculated "
            "by the same scenario engine."
        )


        # ------------------------------------------------------------
        # Extract EXACT component results from Combined engine
        # ------------------------------------------------------------

        t_component_result = result.get(
            "t_result"
        )

        p_component_result = result.get(
            "p_result"
        )

        r_component_result = result.get(
            "r_result"
        )


        if (
            t_component_result is not None
            and
            p_component_result is not None
            and
            r_component_result is not None
        ):

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt


            delta_t = np.asarray(
                t_component_result["delta_svi"],
                dtype=float
            )

            delta_p = np.asarray(
                p_component_result["delta_svi"],
                dtype=float
            )

            delta_r = np.asarray(
                r_component_result["delta_svi"],
                dtype=float
            )

            delta_combined = np.asarray(
                result["delta_svi"],
                dtype=float
            )


            # ========================================================
            # COMMON VALID MASK
            # ========================================================

            decomposition_valid = (
                np.isfinite(delta_t)
                &
                np.isfinite(delta_p)
                &
                np.isfinite(delta_r)
                &
                np.isfinite(delta_combined)
            )


            reconstructed_delta = np.full(
                delta_combined.shape,
                np.nan,
                dtype=float
            )

            reconstructed_delta[
                decomposition_valid
            ] = (
                delta_t[
                    decomposition_valid
                ]
                +
                delta_p[
                    decomposition_valid
                ]
                +
                delta_r[
                    decomposition_valid
                ]
            )


            residual_delta = np.full(
                delta_combined.shape,
                np.nan,
                dtype=float
            )

            residual_delta[
                decomposition_valid
            ] = (
                delta_combined[
                    decomposition_valid
                ]
                -
                reconstructed_delta[
                    decomposition_valid
                ]
            )


            # ========================================================
            # ADDITIVITY AUDIT
            # ========================================================

            residual_values = residual_delta[
                decomposition_valid
            ]


            if residual_values.size > 0:

                max_abs_residual = float(
                    np.max(
                        np.abs(
                            residual_values
                        )
                    )
                )

                mean_abs_residual = float(
                    np.mean(
                        np.abs(
                            residual_values
                        )
                    )
                )

                residual_rmse = float(
                    np.sqrt(
                        np.mean(
                            residual_values ** 2
                        )
                    )
                )

            else:

                max_abs_residual = np.nan
                mean_abs_residual = np.nan
                residual_rmse = np.nan


            audit1, audit2, audit3, audit4 = st.columns(
                4
            )

            audit1.metric(
                "Common Cells",
                f"{int(np.sum(decomposition_valid)):,}"
            )

            audit2.metric(
                "Max |Residual|",
                f"{max_abs_residual:.3e}"
            )

            audit3.metric(
                "Mean |Residual|",
                f"{mean_abs_residual:.3e}"
            )

            audit4.metric(
                "Residual RMSE",
                f"{residual_rmse:.3e}"
            )


            if (
                np.isfinite(max_abs_residual)
                and
                max_abs_residual <= 1e-6
            ):

                st.success(
                    "Additivity check PASSED: "
                    "ΔSVI_T + ΔSVI_P + ΔSVI_R reproduces "
                    "the Combined ΔSVI within numerical precision."
                )

            else:

                st.error(
                    "Additivity check FAILED. "
                    "Do not interpret the component maps until "
                    "the residual has been investigated."
                )


            # ========================================================
            # COMPONENT MAPS
            # Same symmetric scale for direct visual comparison
            # ========================================================

            component_arrays = [
                delta_t,
                delta_p,
                delta_r,
                delta_combined,
            ]

            component_titles = [
                "ΔSVI — T",
                "ΔSVI — P",
                "ΔSVI — R",
                "ΔSVI — Combined",
            ]


            finite_abs_values = []

            for component_array in component_arrays:

                component_values = component_array[
                    np.isfinite(component_array)
                ]

                if component_values.size > 0:

                    finite_abs_values.append(
                        np.abs(component_values)
                    )


            if finite_abs_values:

                common_scale = float(
                    np.max(
                        np.concatenate(
                            finite_abs_values
                        )
                    )
                )

            else:

                common_scale = 1.0


            if (
                not np.isfinite(common_scale)
                or common_scale == 0
            ):
                common_scale = 1.0



            # HIGH-CONTRAST ΔSVI COLORMAP
            from matplotlib.colors import LinearSegmentedColormap

            contrast_cmap = LinearSegmentedColormap.from_list(
                "enhanced_delta_svi",
                [
                    (0.00, "#08306b"),
                    (0.28, "#2171b5"),
                    (0.42, "#6baed6"),
                    (0.485, "#d9ecf5"),
                    (0.50, "#ffffff"),
                    (0.515, "#fee0d2"),
                    (0.58, "#fc9272"),
                    (0.72, "#de2d26"),
                    (1.00, "#67000d"),
                ]
            )

            fig_components, axes_components = plt.subplots(
                2,
                2,
                figsize=(12, 7.5)
            )


            component_image = None

            for (
                ax,
                component_array,
                component_title
            ) in zip(
                axes_components.ravel(),
                component_arrays,
                component_titles
            ):

                component_image = ax.imshow(
                    component_array,
                    cmap=contrast_cmap,
                    vmin=-common_scale,
                    vmax=common_scale,
                    origin="upper"
                )

                # SUBTLE OUTER BORDER — FACTOR-WISE MAPS
                from scipy.ndimage import binary_fill_holes

                plain_mask = np.isfinite(
                    delta_combined
                )

                plain_outer_mask = binary_fill_holes(
                    plain_mask
                ).astype(float)

                ax.contour(
                    plain_outer_mask,
                    levels=[0.5],
                    colors="#4a4a4a",
                    linewidths=1.0,
                    alpha=0.95,
                                    zorder=20
                )


                ax.set_title(
                    component_title
                )

                ax.set_axis_off()


            fig_components.colorbar(
                component_image,
                ax=axes_components.ravel().tolist(),
                fraction=0.025,
                pad=0.02,
                label="ΔSVI (index points)"
            )

            fig_components.subplots_adjust(
                wspace=0.05,
                hspace=0.15
            )


            st.pyplot(
                fig_components,
                use_container_width=True
            )

            plt.close(
                fig_components
            )


            st.caption(
                "All four maps use the same color scale. "
                "Red indicates increased ALPRIFT vulnerability; "
                "blue indicates decreased vulnerability; "
                "near-white indicates little or no ΔSVI."
            )


            # ========================================================
            # CELL AREA
            # ========================================================

            try:

                profile_transform = result[
                    "svi_profile"
                ]["transform"]

                cell_area_km2 = abs(
                    float(profile_transform.a)
                    *
                    float(profile_transform.e)
                ) / 1_000_000.0

            except Exception:

                # Current model grid is 100 m × 100 m
                cell_area_km2 = 0.01


            # ========================================================
            # COMPONENT SUMMARY TABLE
            # ========================================================

            def _component_statistics(
                component_name,
                component_array
            ):

                values = component_array[
                    np.isfinite(component_array)
                ]

                changed = (
                    np.abs(values)
                    > 1e-12
                )


                if values.size == 0:

                    return {
                        "Component": component_name,
                        "Min ΔSVI": np.nan,
                        "Max ΔSVI": np.nan,
                        "Mean ΔSVI": np.nan,
                        "Changed Cells": 0,
                        "Changed Area (km²)": 0.0,
                    }


                changed_cells = int(
                    np.sum(
                        changed
                    )
                )


                return {
                    "Component":
                        component_name,

                    "Min ΔSVI":
                        float(
                            np.min(values)
                        ),

                    "Max ΔSVI":
                        float(
                            np.max(values)
                        ),

                    "Mean ΔSVI":
                        float(
                            np.mean(values)
                        ),

                    "Changed Cells":
                        changed_cells,

                    "Changed Area (km²)":
                        float(
                            changed_cells
                            *
                            cell_area_km2
                        ),
                }


            decomposition_summary = pd.DataFrame(
                [
                    _component_statistics(
                        "T — Groundwater-level Decline",
                        delta_t
                    ),

                    _component_statistics(
                        "P — Groundwater Pumping",
                        delta_p
                    ),

                    _component_statistics(
                        "R — Rainfall / Recharge",
                        delta_r
                    ),

                    _component_statistics(
                        "Combined",
                        delta_combined
                    ),
                ]
            )



            # STUDY AREA PERCENTAGE — DECOMPOSITION
            # ------------------------------------------------------------

            total_study_cells = int(
                np.sum(
                    np.isfinite(
                        delta_combined
                    )
                )
            )

            total_study_area_km2 = (
                total_study_cells
                *
                cell_area_km2
            )

            if total_study_area_km2 > 0:

                decomposition_summary[
                    "Study Area (%)"
                ] = (
                    100.0
                    *
                    decomposition_summary[
                        "Changed Area (km²)"
                    ]
                    /
                    total_study_area_km2
                )

                decomposition_summary[
                    "Study Area (%)"
                ] = (
                    decomposition_summary[
                        "Study Area (%)"
                    ].round(2)
                )

            else:

                decomposition_summary[
                    "Study Area (%)"
                ] = np.nan


            st.dataframe(
                decomposition_summary,
                use_container_width=True,
                hide_index=True
            )


            # ========================================================
            # DIAGNOSTIC INTERPRETATION
            # ========================================================

            with st.expander(
                "Diagnostic interpretation guide"
            ):

                st.markdown(
                    """
        **T — Groundwater-level Decline**

        The selected piezometer values are modified together before the T surface
        is reconstructed using one scenario-engine IDW procedure.
        Therefore the T effect can extend beyond the selected
        piezometer location.

        **P — Groundwater Pumping**

        The P scenario modifies the selected pumping zones together.
        If its ALPRIFT P rating changes, the ΔSVI_P map should reveal
        the spatial footprints of the selected zones. If the physical
        pumping value changes but remains within the same P-rating
        class, ΔSVI_P can be zero.

        **R — Rainfall / Recharge**

        Rainfall changes across the recharge calculation while slope
        and permeability remain spatially fixed. Therefore ΔSVI_R may
        occur in different parts of the plain.

        **Combined**

        The Combined map is the actual scenario change:

        `ΔSVI_Combined = ΔSVI_T + ΔSVI_P + ΔSVI_R`

        These maps describe change in the **ALPRIFT vulnerability
        index**. They do not represent millimetres of predicted land
        subsidence.
                    """
                )


            # ========================================================
            # SHOW RESIDUAL MAP ONLY IF AUDIT FAILS
            # ========================================================

            if (
                np.isfinite(max_abs_residual)
                and
                max_abs_residual > 1e-6
            ):

                st.markdown(
                    "#### Residual Map"
                )

                residual_scale = float(
                    np.nanmax(
                        np.abs(
                            residual_delta
                        )
                    )
                )

                if (
                    not np.isfinite(residual_scale)
                    or residual_scale == 0
                ):
                    residual_scale = 1.0


                fig_res, ax_res = plt.subplots(
                    figsize=(8, 4.5)
                )

                res_im = ax_res.imshow(
                    residual_delta,
                    cmap="RdBu_r",
                    vmin=-residual_scale,
                    vmax=residual_scale,
                    origin="upper"
                )

                ax_res.set_title(
                    "Residual: Combined − (T + P + R)"
                )

                ax_res.set_axis_off()

                fig_res.colorbar(
                    res_im,
                    ax=ax_res,
                    label="Residual ΔSVI"
                )

                fig_res.tight_layout()

                st.pyplot(
                    fig_res,
                    use_container_width=True
                )

                plt.close(
                    fig_res
                )


        else:

            st.warning(
                "T, P, or R component results are unavailable "
                "for decomposition."
            )


        st.markdown("---")


        # FINAL SCENARIO SCIENTIFIC INTERPRETATION
        # ============================================================

        st.subheader(
            "Scientific Interpretation"
        )

        st.warning(
            "This module is an ALPRIFT vulnerability what-if analysis. "
            "It evaluates changes in the ALPRIFT vulnerability index "
            "and its classes; it does not directly predict future "
            "land-subsidence displacement."
        )


        if all(
            name in locals()
            for name in [
                "delta_t",
                "delta_p",
                "delta_r",
                "delta_combined",
            ]
        ):

            baseline_mean_svi_science = float(
                np.nanmean(
                    result["baseline_svi"]
                )
            )

            scenario_mean_svi_science = float(
                np.nanmean(
                    result["scenario_svi"]
                )
            )

            mean_delta_science = (
                scenario_mean_svi_science
                -
                baseline_mean_svi_science
            )


            component_means_science = {
                "T — Groundwater-level Decline":
                    float(
                        np.nanmean(delta_t)
                    ),

                "P — Groundwater Pumping":
                    float(
                        np.nanmean(delta_p)
                    ),

                "R — Rainfall / Recharge":
                    float(
                        np.nanmean(delta_r)
                    ),
            }


            strongest_increase = max(
                component_means_science,
                key=component_means_science.get
            )

            strongest_decrease = min(
                component_means_science,
                key=component_means_science.get
            )


            transitions_science = result.get(
                "transitions"
            )

            higher_area_science = 0.0
            lower_area_science = 0.0


            class_order_science = {
                "Low": 1,
                "Moderate": 2,
                "High": 3,
                "Very High": 4,
            }


            if transitions_science is not None:

                for _, row_science in transitions_science.iterrows():

                    from_class_science = str(
                        row_science["From"]
                    )

                    to_class_science = str(
                        row_science["To"]
                    )

                    area_science = float(
                        row_science["Area_km2"]
                    )

                    if (
                        from_class_science
                        in class_order_science
                        and
                        to_class_science
                        in class_order_science
                    ):

                        if (
                            class_order_science[
                                to_class_science
                            ]
                            >
                            class_order_science[
                                from_class_science
                            ]
                        ):

                            higher_area_science += (
                                area_science
                            )

                        elif (
                            class_order_science[
                                to_class_science
                            ]
                            <
                            class_order_science[
                                from_class_science
                            ]
                        ):

                            lower_area_science += (
                                area_science
                            )


            if mean_delta_science > 1e-9:

                net_statement = (
                    "The scenario produces a net increase "
                    "in mean ALPRIFT vulnerability."
                )

            elif mean_delta_science < -1e-9:

                net_statement = (
                    "The scenario produces a net decrease "
                    "in mean ALPRIFT vulnerability."
                )

            else:

                net_statement = (
                    "The scenario produces essentially no "
                    "change in mean ALPRIFT vulnerability."
                )


            st.markdown(
                f"""
        **Overall scenario response**

        Baseline Mean SVI: **{baseline_mean_svi_science:.3f}**

        Scenario Mean SVI: **{scenario_mean_svi_science:.3f}**

        Mean ΔSVI: **{mean_delta_science:+.3f}**

        {net_statement}

        **Factor-wise response**

        Largest positive mean contribution:
        **{strongest_increase}**
        ({component_means_science[strongest_increase]:+.3f})

        Most negative mean contribution:
        **{strongest_decrease}**
        ({component_means_science[strongest_decrease]:+.3f})

        **Vulnerability-class transitions**

        Shifted to a higher vulnerability class:
        **{higher_area_science:.2f} km²**

        Shifted to a lower vulnerability class:
        **{lower_area_science:.2f} km²**
        """
            )


            st.info(
                "Class transitions and ΔSVI magnitude are different "
                "concepts. A small ΔSVI may change a class when the "
                "baseline SVI is close to a class threshold, while "
                "a larger ΔSVI may leave the class unchanged."
            )


        # ============================================================
        # HOW WAS THIS CALCULATED?
        # ============================================================

        with st.expander(
            "How was this scenario calculated?"
        ):

            st.markdown(
                r"""
        ### Combined ALPRIFT scenario

        The official baseline SVI is retained as the reference surface.

        \[
        SVI_{scenario}
        =
        SVI_{baseline}
        +
        \Delta SVI_T
        +
        \Delta SVI_P
        +
        \Delta SVI_R
        \]

        with:

        \[
        \Delta SVI_T
        =
        5(T_{rate,scenario}-T_{rate,baseline})
        \]

        \[
        \Delta SVI_P
        =
        4(P_{rate,scenario}-P_{rate,baseline})
        \]

        \[
        \Delta SVI_R
        =
        4(R_{rate,scenario}-R_{rate,baseline})
        \]

        The remaining ALPRIFT components
        **A, L, I and F remain fixed at their baseline values.**

        ---

        ### T — Groundwater-level Decline

        **Data source:** annual piezometer observations.

        T is calculated using the annual Azar-to-Azar groundwater-level
        change.

        The selected piezometer value(s) are modified together by the scenario.

        The scenario T surface is reconstructed using the same IDW method
        used by the T engine:

        - Power = 1.5
        - 40 nearest observations
        - 100 m model grid

        **Weight: W = 5**

        Because T is spatially interpolated, changing one piezometer can
        affect multiple surrounding raster cells.

        ---

        ### P — Groundwater Pumping

        **Baseline source:** pumping Thiessen zones.

        **Zone identifier:** TARGET_FID.

        A direct Scenario P value is assigned to each selected pumping zone.

        The physical pumping value is converted to its ALPRIFT P rating.

        **Weight: W = 4**

        A pumping change affects SVI only if it changes the corresponding
        P rating.

        ---

        ### R — Rainfall / Recharge

        The user changes annual rainfall.

        Rainfall is first converted to **Rain Rank**.

        Rain Rank is combined with the fixed spatial:

        - Slope Rank
        - Permeability Rank

        using the Piscopo-based recharge calculation.

        Recharge is subsequently converted to the ALPRIFT R rating.

        **Weight: W = 4**

        Because slope and permeability vary spatially, the same rainfall
        scenario may produce different R changes across the plain.

        ---

        ### Fixed ALPRIFT components

        - A — Aquifer Media: W = 5
        - L — Land Use: W = 3
        - I — Aquifer Thickness: W = 2
        - F — Distance from Fault: W = 1

        These components are not modified in the current scenario module.

        ---

        ### Additivity check

        The diagnostic module verifies:

        \[
        Residual
        =
        \Delta SVI_{Combined}
        -
        (
        \Delta SVI_T
        +
        \Delta SVI_P
        +
        \Delta SVI_R
        )
        \]

        A residual close to zero confirms that the factor-wise maps
        reproduce the actual Combined Scenario.

        ---

        ### Interpretation

        The DSS answers:

        **How would the ALPRIFT vulnerability index change under the
        specified changes in T, P and R?**

        It does not provide a direct physical forecast of future land
        subsidence in millimetres.
        """
            )


        st.markdown("---")

        # VULNERABILITY CLASS MAP COMPARISON
        # ============================================================

        from matplotlib.colors import (
            BoundaryNorm,
            ListedColormap,
        )


        def _classify_svi_for_display(svi_array):

            svi_array = np.asarray(
                svi_array,
                dtype=float
            )

            out = np.full(
                svi_array.shape,
                np.nan,
                dtype=float
            )

            valid = np.isfinite(
                svi_array
            )

            out[
                valid
                & (svi_array >= 24)
                & (svi_array < 78)
            ] = 1

            out[
                valid
                & (svi_array >= 78)
                & (svi_array < 133)
            ] = 2

            out[
                valid
                & (svi_array >= 133)
                & (svi_array < 186)
            ] = 3

            out[
                valid
                & (svi_array >= 186)
                & (svi_array <= 240)
            ] = 4

            return out


        baseline_class_map = (
            _classify_svi_for_display(
                baseline_map
            )
        )

        scenario_class_map = (
            _classify_svi_for_display(
                scenario_map
            )
        )


        # ------------------------------------------------------------
        # Direction of class change
        #
        # -1 = lower vulnerability
        #  0 = no class change
        # +1 = higher vulnerability
        # ------------------------------------------------------------

        class_change_map = np.full(
            baseline_class_map.shape,
            np.nan,
            dtype=float
        )

        class_valid = (
            np.isfinite(baseline_class_map)
            &
            np.isfinite(scenario_class_map)
        )

        class_difference = (
            scenario_class_map
            -
            baseline_class_map
        )

        class_change_map[
            class_valid
            &
            (class_difference < 0)
        ] = -1

        class_change_map[
            class_valid
            &
            (class_difference == 0)
        ] = 0

        class_change_map[
            class_valid
            &
            (class_difference > 0)
        ] = 1


        # ============================================================
        # DISPLAY
        # ============================================================

        st.subheader(
            "Vulnerability Class Comparison"
        )

        st.caption(
            "Baseline and scenario SVI are classified using the "
            "fixed ALPRIFT vulnerability thresholds. The third map "
            "shows only the direction of class transition."
        )


        class_cmap = ListedColormap(
            [
                "#2ca25f",
                "#fee08b",
                "#f46d43",
                "#a50026",
            ]
        )

        class_cmap.set_bad(
            "white"
        )

        class_norm = BoundaryNorm(
            [
                0.5,
                1.5,
                2.5,
                3.5,
                4.5,
            ],
            class_cmap.N
        )


        change_cmap = ListedColormap(
            [
                "#2ca25f",
                "#d9d9d9",
                "#d73027",
            ]
        )

        change_cmap.set_bad(
            "white"
        )

        change_norm = BoundaryNorm(
            [
                -1.5,
                -0.5,
                0.5,
                1.5,
            ],
            change_cmap.N
        )


        fig_class, axes_class = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.8)
        )


        # ------------------------------------------------------------
        # Baseline Vulnerability Class
        # ------------------------------------------------------------

        im_base_class = axes_class[0].imshow(
            baseline_class_map,
            cmap=class_cmap,
            norm=class_norm,
            origin="upper"
        )

        axes_class[0].set_title(
            "Baseline Vulnerability Class"
        )

        axes_class[0].set_axis_off()


        cb_base = fig_class.colorbar(
            im_base_class,
            ax=axes_class[0],
            fraction=0.046,
            pad=0.03,
            ticks=[1, 2, 3, 4]
        )

        cb_base.ax.set_yticklabels(
            [
                "Low",
                "Moderate",
                "High",
                "Very High",
            ]
        )


        # ------------------------------------------------------------
        # Scenario Vulnerability Class
        # ------------------------------------------------------------

        im_scenario_class = axes_class[1].imshow(
            scenario_class_map,
            cmap=class_cmap,
            norm=class_norm,
            origin="upper"
        )

        axes_class[1].set_title(
            "Scenario Vulnerability Class"
        )

        axes_class[1].set_axis_off()


        cb_scenario = fig_class.colorbar(
            im_scenario_class,
            ax=axes_class[1],
            fraction=0.046,
            pad=0.03,
            ticks=[1, 2, 3, 4]
        )

        cb_scenario.ax.set_yticklabels(
            [
                "Low",
                "Moderate",
                "High",
                "Very High",
            ]
        )


        fig_class.tight_layout()

        st.pyplot(
            fig_class,
            use_container_width=True
        )

        plt.close(
            fig_class
        )


        st.caption(
            "Fixed ALPRIFT classes: "
            "Low = 24–<78 | "
            "Moderate = 78–<133 | "
            "High = 133–<186 | "
            "Very High = 186–240."
        )

        st.markdown("---")

        st.subheader(
            "Vulnerability Class Transitions"
        )


        if len(transitions) > 0:

            transitions[
                "From"
            ] = transitions[
                "From_Class"
            ].map(
                CLASS_NAMES
            )


            transitions[
                "To"
            ] = transitions[
                "To_Class"
            ].map(
                CLASS_NAMES
            )


            table = transitions[
                [
                    "From",
                    "To",
                    "Cells",
                    "Area_km2"
                ]
            ].copy()


            table[
                "Area_km2"
            ] = table[
                "Area_km2"
            ].round(2)


            st.dataframe(
                table,
                width="stretch",
                hide_index=True
            )


            upward = transitions[
                transitions[
                    "To_Class"
                ]
                >
                transitions[
                    "From_Class"
                ]
            ]


            downward = transitions[
                transitions[
                    "To_Class"
                ]
                <
                transitions[
                    "From_Class"
                ]
            ]


            if len(upward) > 0:

                area = float(
                    upward[
                        "Area_km2"
                    ].sum()
                )

                st.warning(
                    f"{area:.2f} km² "
                    "shifted to a higher vulnerability class."
                )


            if len(downward) > 0:

                area = float(
                    downward[
                        "Area_km2"
                    ].sum()
                )

                st.success(
                    f"{area:.2f} km² "
                    "shifted to a lower vulnerability class."
                )


            if (
                len(upward) == 0
                and
                len(downward) == 0
            ):

                st.info(
                    "مقدار SVI تغییر کرده است، "
                    "اما هیچ سلولی از مرز کلاس‌های آسیب‌پذیری عبور نکرده است."
                )


        # ========================================================
        # TECHNICAL DETAILS
        # ========================================================

        with st.expander(
            "Technical Details"
        ):

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


            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True
            )


        # ========================================================
        # OUTPUT FOLDER
        # ========================================================

        if result[
            "scenario_folder"
        ] is not None:

            st.caption(
                "مسیر ذخیره خروجی‌ها"
            )

            st.code(
                str(
                    result[
                        "scenario_folder"
                    ]
                )
            )


    else:

        st.divider()

        st.info(
            "پارامترهای سناریو را انتخاب کن و سپس دکمه اجرا را بزن."
        )


# ============================================================
# WEIGHT SENSITIVITY AND OPTIMIZATION TAB
# ============================================================

with weight_tab:

    st.header(
        "ALPRIFT Weight Optimization & Model Benchmark"
    )

    st.caption(
        "Weight calibration, spatial validation, nonlinear model "
        "benchmarking and temporal transfer are reported separately "
        "to avoid mixing different evaluation designs."
    )


    # ========================================================
    # REFERENCE MODEL
    # ========================================================

    st.subheader(
        "Reference ALPRIFT"
    )

    ref1, ref2 = st.columns(
        [2, 1]
    )

    with ref1:

        st.code(
            "A,L,P,R,I,F,T = 5,4,4,3,2,1,5"
        )

        st.latex(
            r"SVI_{ref}=5A+4L+4P+3R+2I+F+5T"
        )

    with ref2:

        st.metric(
            "Reference ΣW",
            "24"
        )

    st.caption(
        "Reference2018 is preserved unchanged throughout all analyses."
    )


    # ========================================================
    # 1 — SAME-YEAR WEIGHT CALIBRATION
    # ========================================================

    st.markdown("---")

    st.header(
        "1. Same-year Weight Calibration"
    )

    st.caption(
        "Each year's optimized weights were calibrated against "
        "InSAR observations from the same year. These are calibration "
        "results, not independent validation results."
    )


    # --------------------------------------------------------
    # 1395
    # --------------------------------------------------------

    cal95, cal96 = st.columns(
        2,
        gap="large"
    )

    with cal95:

        st.subheader(
            "1395"
        )

        st.markdown(
            "**Reference → Optimized**"
        )

        st.code(
            "5,4,4,3,2,1,5  →  5,5,5,1,1,1,5"
        )

        a1, a2 = st.columns(2)

        a1.metric(
            "Pearson r — Reference",
            "0.1320"
        )

        a2.metric(
            "Pearson r — Optimized",
            "0.1769",
            delta="+34.03%"
        )

        b1, b2 = st.columns(2)

        b1.metric(
            "Spearman ρ — Reference",
            "0.1612"
        )

        b2.metric(
            "Spearman ρ — Optimized",
            "0.2127"
        )

        st.caption(
            "Optimized weight sum = 23 | "
            "Calibration pixels = 47,257"
        )


    # --------------------------------------------------------
    # 1396
    # --------------------------------------------------------

    with cal96:

        st.subheader(
            "1396"
        )

        st.markdown(
            "**Reference → Optimized**"
        )

        st.code(
            "5,4,4,3,2,1,5  →  5,5,5,5,1,1,1"
        )

        a1, a2 = st.columns(2)

        a1.metric(
            "Pearson r — Reference",
            "0.1452"
        )

        a2.metric(
            "Pearson r — Optimized",
            "0.2701",
            delta="+85.98%"
        )

        b1, b2 = st.columns(2)

        b1.metric(
            "Spearman ρ — Reference",
            "0.1534"
        )

        b2.metric(
            "Spearman ρ — Optimized",
            "0.3003"
        )

        st.caption(
            "Optimized weight sum = 23"
        )


    st.warning(
        "Year-specific optimized weights are not temporally stable: "
        "the best 1395 and 1396 weight vectors are different. "
        "Therefore these same-year solutions are diagnostic "
        "calibrations rather than a final universal multi-year "
        "weighting system."
    )


    # ========================================================
    # 2 — BEYOND LINEAR WEIGHTS
    # ========================================================

    st.markdown("---")

    st.header(
        "2. Beyond Linear Weights — Model Performance"
    )

    st.caption(
        "The values below summarize different modeling approaches. "
        "Because their validation designs are not identical, they "
        "must not be interpreted as directly comparable accuracy scores."
    )


    # --------------------------------------------------------
    # 1395 MODEL SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "1395"
    )

    y95_1, y95_2, y95_3, y95_4 = st.columns(4)

    with y95_1:
        st.metric(
            "Reference ALPRIFT",
            "0.1320"
        )
        st.caption(
            "Pearson r | Raw annual comparison"
        )

    with y95_2:
        st.metric(
            "Weight Optimized",
            "0.1769"
        )
        st.caption(
            "Pearson r | Same-year calibration"
        )

    with y95_3:
        st.metric(
            "SVR-RBF",
            "0.6067"
        )
        st.caption(
            "Mean Pearson r | 5-fold Spatial CV"
        )

    with y95_4:
        st.metric(
            "ANN 7–9–1",
            "0.8222"
        )
        st.caption(
            "Pearson r | Random 80/20 Test"
        )


    # --------------------------------------------------------
    # 1396 MODEL SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "1396"
    )

    y96_1, y96_2, y96_3, y96_4 = st.columns(4)

    with y96_1:
        st.metric(
            "Reference ALPRIFT",
            "0.1452"
        )
        st.caption(
            "Pearson r | Raw annual comparison"
        )

    with y96_2:
        st.metric(
            "Weight Optimized",
            "0.2701"
        )
        st.caption(
            "Pearson r | Same-year calibration"
        )

    with y96_3:
        st.metric(
            "SVR-RBF",
            "0.5591"
        )
        st.caption(
            "Mean Pearson r | 5-fold Spatial CV"
        )

    with y96_4:
        st.metric(
            "ANN 7–9–1",
            "0.8408"
        )
        st.caption(
            "Pearson r | Random 80/20 Test"
        )


    st.info(
        "Important: Reference ALPRIFT, same-year optimized ALPRIFT, "
        "SVR spatial cross-validation and ANN random testing use "
        "different evaluation designs. The displayed Pearson values "
        "therefore summarize model behavior but are not four strictly "
        "equivalent accuracy estimates."
    )


    # ========================================================
    # 3 — SPATIAL VALIDATION
    # ========================================================

    st.markdown("---")

    st.header(
        "3. Spatially Independent Validation"
    )

    st.caption(
        "Five contiguous west-to-east spatial folds were used "
        "to reduce spatial leakage between training and validation."
    )


    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    st.subheader(
        "SVR-RBF — 5-fold Spatial CV"
    )

    st.caption(
        "Fixed hyperparameters: C = 3, γ = 0.03, ε = 2"
    )

    svr_df = pd.DataFrame({
        "Year": [
            1395,
            1396,
        ],

        "Mean Pearson": [
            0.606741,
            0.559103,
        ],

        "Mean Spearman": [
            0.588814,
            0.653989,
        ],

        "RMSE (mm)": [
            14.508,
            16.556,
        ],

        "MAE (mm)": [
            11.180,
            12.527,
        ],

        "Mean R²": [
            0.229569,
            -0.013281,
        ],

        "Bias (mm)": [
            None,
            -3.448,
        ],
    })

    st.dataframe(
        svr_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ANN VALIDATION DESIGN
    # --------------------------------------------------------

    st.subheader(
        "ANN 7–9–1 — Effect of Validation Design"
    )

    ann95_1, ann95_2 = st.columns(2)

    with ann95_1:

        st.metric(
            "1395 Random 80/20 Pearson",
            "0.8222"
        )

        st.caption(
            "Random pixel split"
        )

    with ann95_2:

        st.metric(
            "1395 Spatial CV Pearson",
            "0.4999",
            delta="-0.3223"
        )

        st.caption(
            "Spatially independent validation"
        )


    st.warning(
        "Random pixel splitting produced substantially higher "
        "apparent ANN performance than spatial cross-validation. "
        "This demonstrates the importance of spatially independent "
        "validation when evaluating spatial prediction models."
    )


    # --------------------------------------------------------
    # ANN ARTICLE-STYLE DETAILS
    # --------------------------------------------------------

    with st.expander(
        "ANN article-style random 80/20 results",
        expanded=False
    ):

        ann_df = pd.DataFrame({
            "Year": [
                1395,
                1396,
            ],

            "Pearson": [
                0.822156,
                0.840848,
            ],

            "Spearman": [
                0.809138,
                0.860493,
            ],

            "R²": [
                0.675390,
                0.706959,
            ],

            "RMSE (mm)": [
                10.496,
                9.894,
            ],

            "MAE (mm)": [
                7.576,
                7.240,
            ],

            "Bias (mm)": [
                None,
                0.083,
            ],
        })

        st.dataframe(
            ann_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "ANN architecture = 7–9–1 | Validation = Random 80/20"
        )


    # ========================================================
    # 4 — TEMPORAL TRANSFER
    # ========================================================

    st.markdown("---")

    st.header(
        "4. Temporal Transfer — 1395 → 1396"
    )

    st.caption(
        "SVR-RBF was trained using 1395 and transferred to 1396 "
        "without model retraining."
    )

    tt1, tt2, tt3 = st.columns(3)

    tt1.metric(
        "Pearson r",
        "0.6818"
    )

    tt2.metric(
        "Spearman ρ",
        "0.6496"
    )

    tt3.metric(
        "Bias",
        "+13.86 mm"
    )

    tt4, tt5, tt6 = st.columns(3)

    tt4.metric(
        "RMSE",
        "19.55 mm"
    )

    tt5.metric(
        "MAE",
        "16.31 mm"
    )

    tt6.metric(
        "R²",
        "-0.135"
    )

    st.info(
        "Spatial ranking transferred reasonably well to 1396, "
        "but the large positive bias indicates poor calibration "
        "of absolute deformation magnitude across years."
    )


    # ========================================================
    # TECHNICAL WEIGHT OPTIMIZATION DETAILS
    # ========================================================

    st.markdown("---")

    with st.expander(
        "Weight optimization methodology and robustness checks",
        expanded=False
    ):

        st.markdown(
            "### 1395 constrained exhaustive search"
        )

        st.write(
            "Weights: 1–5 | Constraint: ΣW = 24 | "
            "Combinations tested: 6,055"
        )

        constrained_df = pd.DataFrame({
            "Model": [
                "Reference",
                "Best constrained"
            ],

            "Weights A,L,P,R,I,F,T": [
                "5,4,4,3,2,1,5",
                "5,5,5,2,1,1,5",
            ],

            "Pearson": [
                0.132002,
                0.173665,
            ],

            "Spearman": [
                0.161192,
                0.210288,
            ],
        })

        st.dataframe(
            constrained_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "### 1395 full exhaustive search"
        )

        st.write(
            "All 78,125 integer combinations with weights 1–5 "
            "were evaluated without the ΣW = 24 constraint."
        )

        st.code(
            "Best = 5,5,5,1,1,1,5 | ΣW = 23"
        )


        st.markdown(
            "### 1395 spatial CV of optimized weights"
        )

        spatial_weight_df = pd.DataFrame({
            "Model": [
                "Reference2018",
                "Optimized",
            ],

            "Mean Pearson": [
                0.136598,
                0.155232,
            ],

            "Mean Spearman": [
                0.138678,
                0.164658,
            ],

            "RMSE (mm)": [
                19.3836,
                19.2625,
            ],

            "MAE (mm)": [
                16.2044,
                16.0797,
            ],
        })

        st.dataframe(
            spatial_weight_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown(
            "### Continuous GA — Pearson objective"
        )

        st.write(
            "10 GA runs were performed over continuous weights "
            "between 1 and 5."
        )

        st.code(
            "Best = 5,5,5,1,1,1,5 | Pearson = 0.176922817"
        )

        st.caption(
            "The exhaustive-search solution was included in the "
            "GA initial population; therefore repeated recovery of "
            "the same result is not an independent convergence proof. "
            "The valid conclusion is that GA did not find a better solution."
        )


        st.markdown(
            "### Diagnostic GA — RMSE objective"
        )

        diagnostic_df = pd.DataFrame({
            "Status": [
                "Rejected"
            ],

            "Weights": [
                "1,1,1,1,5,5,1"
            ],

            "RMSE (mm)": [
                39.7949
            ],

            "Pearson": [
                -0.3970
            ],
        })

        st.dataframe(
            diagnostic_df,
            use_container_width=True,
            hide_index=True
        )

        st.error(
            "Rejected because the spatial relationship with "
            "observations became negative."
        )


    # ========================================================
    # TECHNICAL FILES
    # ========================================================

    with st.expander(
        "Technical output files",
        expanded=False
    ):

        optimization_files = [

            ROOT
            / "Baseline"
            / "SVI"
            / "WeightOptimization_1395_All78125.csv",

            ROOT
            / "Baseline"
            / "SVI"
            / "SpatialCV_1395_5Fold.csv",

            ROOT
            / "Baseline"
            / "SVI"
            / "ML_Benchmark_1395_1396"
            / "01_Family_Summary_1395.csv",

            ROOT
            / "Baseline"
            / "SVI"
            / "ML_Benchmark_1395_1396"
            / "02_Temporal_Holdout_1396.csv",
        ]

        for path in optimization_files:

            if path.exists():

                st.caption(
                    f"✅ {path}"
                )

            else:

                st.caption(
                    f"⚠️ Not found locally: {path}"
                )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Overall Interpretation"
    )

    st.info(
        "Weight optimization produced a measurable but limited "
        "improvement in the linear ALPRIFT framework. "
        "The much larger performance gains obtained by nonlinear "
        "models indicate that limitations of the original model "
        "are not explained by weight selection alone. At the same "
        "time, spatial and temporal validation show that evaluation "
        "design strongly affects apparent model performance."
    )

    st.warning(
        "Reference ALPRIFT remains the official baseline. "
        "Year-specific optimized weights and nonlinear models are "
        "maintained as separate analytical products."
    )

