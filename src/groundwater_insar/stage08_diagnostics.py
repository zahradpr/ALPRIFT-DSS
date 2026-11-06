from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config


DETAIL_FILENAME = (
    "08_Trend_Seasonality_Autocorrelation_ByWell.csv"
)

SUMMARY_FILENAME = (
    "08_Diagnostics_Summary.csv"
)

VARIABLES = [
    "GW_Decline_m",
    "InSAR_Increment_mm",
]


def _pearson_safe(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan

    result = stats.pearsonr(x, y)

    return float(result.statistic), float(result.pvalue)


def _calendar_acf(
    g: pd.DataFrame,
    variable: str,
    lag: int,
):

    current = (
        g[
            ["MonthIndex", variable]
        ]
        .rename(
            columns={
                variable: "Current"
            }
        )
        .copy()
    )

    previous = (
        g[
            ["MonthIndex", variable]
        ]
        .rename(
            columns={
                variable: "Previous"
            }
        )
        .copy()
    )

    # Associate observation at month t
    # with observation at month t + lag.
    previous["MonthIndex"] = (
        previous["MonthIndex"] + lag
    )

    paired = (
        current
        .merge(
            previous,
            on="MonthIndex",
            how="inner",
            validate="one_to_one",
        )
        .dropna(
            subset=[
                "Current",
                "Previous",
            ]
        )
    )

    r, p = _pearson_safe(
        paired["Previous"],
        paired["Current"],
    )

    return len(paired), r, p


def reconstruct_stage08(
    cfg: dict | None = None,
):

    if cfg is None:
        cfg = load_config()

    regenerated_root = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage07_path = (
        regenerated_root
        / "stage07"
        / "07_Canonical_Monthly_Pairs_37Wells.csv"
    )

    if not stage07_path.exists():
        raise FileNotFoundError(
            stage07_path
        )

    s7 = pd.read_csv(
        stage07_path
    )

    expected_wells = int(
        cfg["analysis"][
            "expected_lag_eligible_wells"
        ]
    )

    observed_wells = (
        s7["Piezometer"]
        .nunique()
    )

    if observed_wells != expected_wells:
        raise RuntimeError(
            "Unexpected Stage-07 well count: "
            f"{observed_wells}; "
            f"expected {expected_wells}."
        )

    rows = []

    # ==========================================================
    # BY-WELL DIAGNOSTICS
    # ==========================================================

    for well, g in s7.groupby(
        "Piezometer",
        sort=True,
    ):

        g = (
            g
            .sort_values(
                "MonthIndex"
            )
            .copy()
        )

        for variable in VARIABLES:

            d = (
                g[
                    [
                        "MonthIndex",
                        "JMonth",
                        variable,
                    ]
                ]
                .dropna(
                    subset=[variable]
                )
                .copy()
            )

            x = (
                d["MonthIndex"]
                .to_numpy(
                    dtype=float
                )
            )

            y = (
                d[variable]
                .to_numpy(
                    dtype=float
                )
            )

            n = len(d)

            if n < 3:
                raise RuntimeError(
                    f"Insufficient records for "
                    f"{well} / {variable}"
                )

            # --------------------------------------------------
            # Linear trend
            # --------------------------------------------------

            linear = stats.linregress(
                x,
                y,
            )

            trend_pearson = (
                stats.pearsonr(
                    x,
                    y,
                )
            )

            trend_spearman = (
                stats.spearmanr(
                    x,
                    y,
                )
            )

            # --------------------------------------------------
            # Exact calendar-lag autocorrelation
            # --------------------------------------------------

            (
                acf1_n,
                acf1_r,
                acf1_p,
            ) = _calendar_acf(
                g,
                variable,
                lag=1,
            )

            (
                acf12_n,
                acf12_r,
                acf12_p,
            ) = _calendar_acf(
                g,
                variable,
                lag=12,
            )

            # --------------------------------------------------
            # Jalali-month seasonality
            # --------------------------------------------------

            seasonal_groups = []

            for month in sorted(
                d["JMonth"]
                .unique()
            ):

                values = (
                    d.loc[
                        d["JMonth"] == month,
                        variable,
                    ]
                    .dropna()
                    .to_numpy(
                        dtype=float
                    )
                )

                if len(values):
                    seasonal_groups.append(
                        values
                    )

            kruskal = stats.kruskal(
                *seasonal_groups
            )

            monthly_medians = (
                d.groupby(
                    "JMonth"
                )[variable]
                .median()
            )

            seasonal_amplitude = float(
                monthly_medians.max()
                - monthly_medians.min()
            )

            seasonal_max_month = int(
                monthly_medians.idxmax()
            )

            seasonal_min_month = int(
                monthly_medians.idxmin()
            )

            rows.append({
                "Piezometer":
                    well,

                "Variable":
                    variable,

                "N":
                    n,

                "Linear_slope_per_month":
                    float(linear.slope),

                "Linear_slope_per_year":
                    float(
                        linear.slope * 12.0
                    ),

                "Trend_Pearson_r":
                    float(
                        trend_pearson.statistic
                    ),

                "Trend_Pearson_p":
                    float(
                        trend_pearson.pvalue
                    ),

                "Trend_Spearman_rho":
                    float(
                        trend_spearman.statistic
                    ),

                "Trend_Spearman_p":
                    float(
                        trend_spearman.pvalue
                    ),

                "ACF1_N":
                    int(acf1_n),

                "ACF1_Pearson":
                    float(acf1_r),

                "ACF1_p":
                    float(acf1_p),

                "ACF12_N":
                    int(acf12_n),

                "ACF12_Pearson":
                    float(acf12_r),

                "ACF12_p":
                    float(acf12_p),

                "Seasonal_Kruskal_H":
                    float(
                        kruskal.statistic
                    ),

                "Seasonal_Kruskal_p":
                    float(
                        kruskal.pvalue
                    ),

                "Seasonal_Median_Amplitude":
                    seasonal_amplitude,

                "Seasonal_Max_Month":
                    seasonal_max_month,

                "Seasonal_Min_Month":
                    seasonal_min_month,
            })

    detail = pd.DataFrame(rows)

    expected_detail_rows = (
        expected_wells
        * len(VARIABLES)
    )

    if len(detail) != expected_detail_rows:
        raise RuntimeError(
            "Stage-08 detail row-count QC failed: "
            f"{len(detail)} != "
            f"{expected_detail_rows}"
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    summary_rows = []

    for variable in VARIABLES:

        g = detail[
            detail["Variable"]
            == variable
        ]

        summary_rows.append({
            "Variable":
                variable,

            "Wells":
                len(g),

            "Trend_Pearson_p_lt_0.05":
                int(
                    (
                        g[
                            "Trend_Pearson_p"
                        ]
                        < 0.05
                    ).sum()
                ),

            "Trend_Spearman_p_lt_0.05":
                int(
                    (
                        g[
                            "Trend_Spearman_p"
                        ]
                        < 0.05
                    ).sum()
                ),

            "Seasonality_p_lt_0.05":
                int(
                    (
                        g[
                            "Seasonal_Kruskal_p"
                        ]
                        < 0.05
                    ).sum()
                ),

            "ACF1_p_lt_0.05":
                int(
                    (
                        g["ACF1_p"]
                        < 0.05
                    ).sum()
                ),

            "ACF12_p_lt_0.05":
                int(
                    (
                        g["ACF12_p"]
                        < 0.05
                    ).sum()
                ),

            "Median_ACF1":
                float(
                    g[
                        "ACF1_Pearson"
                    ].median()
                ),

            "Median_ACF12":
                float(
                    g[
                        "ACF12_Pearson"
                    ].median()
                ),

            "Median_Seasonal_Amplitude":
                float(
                    g[
                        "Seasonal_Median_Amplitude"
                    ].median()
                ),

            "Median_abs_Trend_Pearson":
                float(
                    g[
                        "Trend_Pearson_r"
                    ]
                    .abs()
                    .median()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ==========================================================
    # STRUCTURAL QC
    # ==========================================================

    if (
        detail[
            ["Piezometer", "Variable"]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate Stage-08 "
            "well-variable diagnostics."
        )

    if not (
        detail[
            "Linear_slope_per_year"
        ]
        .to_numpy()
        ==
        (
            detail[
                "Linear_slope_per_month"
            ]
            * 12.0
        ).to_numpy()
    ).all():
        raise RuntimeError(
            "Annual trend-slope QC failed."
        )

    if set(
        summary["Variable"]
    ) != set(VARIABLES):
        raise RuntimeError(
            "Stage-08 summary variable QC failed."
        )

    # ==========================================================
    # OUTPUT
    # ==========================================================

    out_dir = (
        regenerated_root
        / "stage08"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_out = (
        out_dir
        / DETAIL_FILENAME
    )

    summary_out = (
        out_dir
        / SUMMARY_FILENAME
    )

    detail.to_csv(
        detail_out,
        index=False,
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    return (
        detail,
        summary,
        detail_out,
        summary_out,
    )


def main():

    cfg = load_config()

    (
        detail,
        summary,
        detail_out,
        summary_out,
    ) = reconstruct_stage08(cfg)

    print("=" * 100)
    print(
        "STAGE 08 — TREND, SEASONALITY "
        "AND AUTOCORRELATION"
    )
    print("=" * 100)

    print(
        "Detail output  :",
        detail_out
    )

    print(
        "Summary output :",
        summary_out
    )

    print(
        "Detail rows    :",
        len(detail)
    )

    print(
        "Wells          :",
        detail["Piezometer"].nunique()
    )

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "STAGE 08 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
