from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config


MONTHLY_OUTPUT = (
    "13_Cumulative_GW_InSAR_Monthly_37Wells.csv"
)

METRICS_OUTPUT = (
    "13_Cumulative_Relationship_Metrics_ByWell.csv"
)


def reconstruct_stage13(
    cfg: dict | None = None,
):

    if cfg is None:
        cfg = load_config()

    regen = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage12_path = (
        regen
        / "stage12"
        / "12_Stitched_Cumulative_Epochs_1396_1403_37Wells.csv"
    )

    stage01_path = (
        regen
        / "stage01"
        / "01_Groundwater_Monthly_41Wells.csv"
    )

    for p in [
        stage12_path,
        stage01_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    s12 = pd.read_csv(stage12_path)
    s01 = pd.read_csv(stage01_path)

    s12["AcquisitionDate"] = pd.to_datetime(
        s12["AcquisitionDate"]
    )

    # ========================================================
    # 1. Monthly stitched cumulative InSAR per stack
    # ========================================================

    monthly_stack = (
        s12.groupby(
            [
                "Piezometer",
                "StackYear",
                "JYear",
                "JMonth",
                "MonthIndex",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            Cumulative_Subsidence_mm=(
                "Stitched_Cumulative_mm",
                "mean",
            ),
            N_Epochs=(
                "Stitched_Cumulative_mm",
                "count",
            ),
            First_Date=(
                "AcquisitionDate",
                "min",
            ),
            Last_Date=(
                "AcquisitionDate",
                "max",
            ),
        )
    )

    monthly_stack["StackDistance"] = (
        monthly_stack["StackYear"]
        .sub(
            monthly_stack["JYear"]
        )
        .abs()
    )

    monthly_stack["PreferredStack"] = (
        monthly_stack["StackYear"]
        ==
        monthly_stack["JYear"]
    )

    # Special first month:
    # 1395/12 is represented by stack 1396.
    first_month = (
        monthly_stack["MonthIndex"]
        .min()
    )

    first_mask = (
        monthly_stack["MonthIndex"]
        == first_month
    )

    monthly_stack.loc[
        first_mask,
        "PreferredStack",
    ] = True

    # ========================================================
    # 2. Select one preferred stack per well-month
    #
    # Rule:
    #   1) preferred same-year stack
    #   2) otherwise nearest stack
    #   3) smaller StackYear resolves ties
    # ========================================================

    monthly = (
        monthly_stack.sort_values(
            [
                "Piezometer",
                "MonthIndex",
                "PreferredStack",
                "StackDistance",
                "StackYear",
            ],
            ascending=[
                True,
                True,
                False,
                True,
                True,
            ],
        )
        .drop_duplicates(
            [
                "Piezometer",
                "MonthIndex",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Historical output marks the selected row as preferred.
    monthly["PreferredStack"] = True

    # ========================================================
    # 3. Add monthly groundwater head
    # ========================================================

    gw = s01[
        [
            "Piezometer",
            "JYear",
            "JMonth",
            "MonthIndex",
            "Head_m",
        ]
    ].copy()

    monthly = monthly.merge(
        gw,
        on=[
            "Piezometer",
            "JYear",
            "JMonth",
            "MonthIndex",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # 4. Baselines and cumulative changes
    #
    # Baseline month = 1395/12.
    # Positive cumulative drawdown = groundwater decline.
    # ========================================================

    baseline_month = (
        monthly["MonthIndex"].min()
    )

    baseline = (
        monthly[
            monthly["MonthIndex"]
            == baseline_month
        ][
            [
                "Piezometer",
                "Head_m",
                "Cumulative_Subsidence_mm",
            ]
        ]
        .rename(
            columns={
                "Head_m":
                    "Baseline_Head_1395M12_m",
                "Cumulative_Subsidence_mm":
                    "Baseline_InSAR_mm",
            }
        )
    )

    monthly = monthly.merge(
        baseline,
        on="Piezometer",
        how="left",
        validate="many_to_one",
    )

    monthly["Cumulative_Drawdown_m"] = (
        monthly[
            "Baseline_Head_1395M12_m"
        ]
        -
        monthly["Head_m"]
    )

    monthly[
        "Cumulative_Subsidence_from_Base_mm"
    ] = (
        monthly[
            "Cumulative_Subsidence_mm"
        ]
        -
        monthly[
            "Baseline_InSAR_mm"
        ]
    )

    monthly[
        "Valid_Cumulative_Pair"
    ] = (
        monthly["Head_m"].notna()
        &
        monthly[
            "Cumulative_Subsidence_from_Base_mm"
        ].notna()
    )

    # ========================================================
    # 5. Historical output order / dates
    # ========================================================

    monthly["First_Date"] = pd.to_datetime(
        monthly["First_Date"]
    ).dt.strftime("%Y-%m-%d")

    monthly["Last_Date"] = pd.to_datetime(
        monthly["Last_Date"]
    ).dt.strftime("%Y-%m-%d")

    monthly = monthly[
        [
            "Piezometer",
            "StackYear",
            "JYear",
            "JMonth",
            "MonthIndex",
            "Cumulative_Subsidence_mm",
            "N_Epochs",
            "First_Date",
            "Last_Date",
            "PreferredStack",
            "StackDistance",
            "Head_m",
            "Baseline_Head_1395M12_m",
            "Cumulative_Drawdown_m",
            "Valid_Cumulative_Pair",
            "Baseline_InSAR_mm",
            "Cumulative_Subsidence_from_Base_mm",
        ]
    ]

    monthly = (
        monthly.sort_values(
            [
                "Piezometer",
                "MonthIndex",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 6. Relationship metrics by well
    # ========================================================

    metric_rows = []

    for well, g in monthly.groupby(
        "Piezometer",
        sort=True,
    ):

        valid = (
            g[
                g[
                    "Valid_Cumulative_Pair"
                ]
            ]
            .copy()
        )

        x = valid[
            "Cumulative_Drawdown_m"
        ].to_numpy(float)

        y = valid[
            "Cumulative_Subsidence_from_Base_mm"
        ].to_numpy(float)

        pearson = stats.pearsonr(
            x,
            y,
        )

        spearman = stats.spearmanr(
            x,
            y,
        )

        reg = stats.linregress(
            x,
            y,
        )

        final = (
            valid.sort_values(
                "MonthIndex"
            )
            .iloc[-1]
        )

        metric_rows.append({
            "Piezometer":
                well,

            "N":
                int(len(valid)),

            "Pearson_Drawdown_vs_CumulativeSubsidence":
                float(
                    pearson.statistic
                ),

            "Pearson_p_naive":
                float(
                    pearson.pvalue
                ),

            "Spearman_Drawdown_vs_CumulativeSubsidence":
                float(
                    spearman.statistic
                ),

            "Spearman_p_naive":
                float(
                    spearman.pvalue
                ),

            "Slope_mm_subsidence_per_m_drawdown":
                float(
                    reg.slope
                ),

            "Final_JYear":
                int(
                    final["JYear"]
                ),

            "Final_JMonth":
                int(
                    final["JMonth"]
                ),

            "Final_Drawdown_m":
                float(
                    final[
                        "Cumulative_Drawdown_m"
                    ]
                ),

            "Final_Cumulative_Subsidence_mm":
                float(
                    final[
                        "Cumulative_Subsidence_from_Base_mm"
                    ]
                ),
        })

    metrics = pd.DataFrame(
        metric_rows
    )

    # ========================================================
    # Structural QC
    # ========================================================

    if len(monthly) != 3589:
        raise RuntimeError(
            f"Expected 3589 monthly rows; "
            f"found {len(monthly)}."
        )

    if (
        monthly["Piezometer"].nunique()
        != 37
    ):
        raise RuntimeError(
            "Stage-13 well-count QC failed."
        )

    rows_per_well = (
        monthly.groupby(
            "Piezometer"
        ).size()
    )

    if not (
        rows_per_well == 97
    ).all():
        raise RuntimeError(
            "Expected 97 months per well."
        )

    if (
        monthly["MonthIndex"].min()
        != 16751
        or
        monthly["MonthIndex"].max()
        != 16847
    ):
        raise RuntimeError(
            "Stage-13 MonthIndex range QC failed."
        )

    if not (
        monthly["PreferredStack"]
    ).all():
        raise RuntimeError(
            "PreferredStack QC failed."
        )

    if (
        monthly[
            "Valid_Cumulative_Pair"
        ].sum()
        != 3366
    ):
        raise RuntimeError(
            "Valid cumulative pair QC failed."
        )

    if len(metrics) != 37:
        raise RuntimeError(
            "Stage-13 metric row-count QC failed."
        )

    n_dist = (
        metrics["N"]
        .value_counts()
        .to_dict()
    )

    if n_dist != {
        91: 36,
        90: 1,
    }:
        raise RuntimeError(
            f"Unexpected metrics N distribution: "
            f"{n_dist}"
        )

    # ========================================================
    # Outputs
    # ========================================================

    out_dir = (
        regen
        / "stage13"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_out = (
        out_dir
        / MONTHLY_OUTPUT
    )

    metrics_out = (
        out_dir
        / METRICS_OUTPUT
    )

    monthly.to_csv(
        monthly_out,
        index=False,
    )

    metrics.to_csv(
        metrics_out,
        index=False,
    )

    return (
        monthly,
        metrics,
        monthly_out,
        metrics_out,
    )


def main():

    cfg = load_config()

    (
        monthly,
        metrics,
        monthly_out,
        metrics_out,
    ) = reconstruct_stage13(cfg)

    print("=" * 105)
    print(
        "STAGE 13 — CUMULATIVE GW–INSAR"
    )
    print("=" * 105)

    print(
        "Monthly rows       :",
        len(monthly)
    )

    print(
        "Wells              :",
        monthly[
            "Piezometer"
        ].nunique()
    )

    print(
        "Valid pairs        :",
        int(
            monthly[
                "Valid_Cumulative_Pair"
            ].sum()
        )
    )

    print(
        "Metric rows        :",
        len(metrics)
    )

    print("\nMetrics N distribution:")
    print(
        metrics["N"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutputs:")
    print(monthly_out)
    print(metrics_out)

    print(
        "\nSTAGE 13 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
