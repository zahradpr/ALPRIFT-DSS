from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config


MONTHLY_OUTPUT = "14_Monthly_HydroMechanical_States.csv"
SUMMARY_OUTPUT = "14_Recovery_Continued_Subsidence_ByWell.csv"


def _longest_true_run(values) -> int:

    longest = 0
    current = 0

    for v in values:

        if bool(v):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


def reconstruct_stage14(
    cfg: dict | None = None,
):

    if cfg is None:
        cfg = load_config()

    regen = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage13_path = (
        regen
        / "stage13"
        / "13_Cumulative_GW_InSAR_Monthly_37Wells.csv"
    )

    if not stage13_path.exists():
        raise FileNotFoundError(stage13_path)

    monthly = pd.read_csv(stage13_path)

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
    # 1. Previous-month values within each well
    # ========================================================

    grp = monthly.groupby(
        "Piezometer",
        sort=False,
    )

    monthly["Prev_MonthIndex"] = (
        grp["MonthIndex"].shift(1)
    )

    monthly["Prev_Head_m"] = (
        grp["Head_m"].shift(1)
    )

    monthly["Prev_CumSubs_mm"] = (
        grp[
            "Cumulative_Subsidence_from_Base_mm"
        ].shift(1)
    )

    monthly["Prev_StackYear"] = (
        grp["StackYear"].shift(1)
    )

    # ========================================================
    # 2. Transition QC
    # ========================================================

    monthly[
        "Exact_Consecutive_Month"
    ] = (
        monthly["MonthIndex"]
        - monthly["Prev_MonthIndex"]
        == 1
    )

    # Historical convention:
    # first record of each well is also Cross_Stack=True
    # because it has no preceding stack.
    monthly["Cross_Stack"] = (
        monthly["Prev_StackYear"].isna()
        |
        (
            monthly["StackYear"]
            != monthly["Prev_StackYear"]
        )
    )

    # ========================================================
    # 3. Monthly changes
    #
    # Positive GW_Decline_m = groundwater decline.
    # Positive Subsidence_Increment_mm = subsidence.
    # ========================================================

    monthly["GW_Decline_m"] = (
        monthly["Prev_Head_m"]
        -
        monthly["Head_m"]
    )

    monthly[
        "Subsidence_Increment_mm"
    ] = (
        monthly[
            "Cumulative_Subsidence_from_Base_mm"
        ]
        -
        monthly["Prev_CumSubs_mm"]
    )

    # ========================================================
    # 4. Primary-valid transition
    #
    # Only exact consecutive months inside the SAME stitched
    # SBAS stack are retained for the primary state analysis.
    # ========================================================

    monthly["Primary_Valid"] = (
        monthly[
            "Exact_Consecutive_Month"
        ]
        &
        (~monthly["Cross_Stack"])
        &
        monthly["GW_Decline_m"].notna()
        &
        monthly[
            "Subsidence_Increment_mm"
        ].notna()
    )

    # ========================================================
    # 5. Hydro-mechanical state
    #
    # Strict signs are used.
    # Zeros and invalid transitions remain OTHER_OR_INVALID.
    # ========================================================

    monthly[
        "HydroMechanical_State"
    ] = "OTHER_OR_INVALID"

    valid = monthly["Primary_Valid"]

    decline = (
        monthly["GW_Decline_m"] > 0
    )

    recovery = (
        monthly["GW_Decline_m"] < 0
    )

    subsidence = (
        monthly[
            "Subsidence_Increment_mm"
        ] > 0
    )

    uplift = (
        monthly[
            "Subsidence_Increment_mm"
        ] < 0
    )

    monthly.loc[
        valid & decline & subsidence,
        "HydroMechanical_State",
    ] = "DECLINE_AND_SUBSIDENCE"

    monthly.loc[
        valid & decline & uplift,
        "HydroMechanical_State",
    ] = "DECLINE_BUT_UPLIFT"

    monthly.loc[
        valid & recovery & subsidence,
        "HydroMechanical_State",
    ] = "RECOVERY_BUT_SUBSIDENCE"

    monthly.loc[
        valid & recovery & uplift,
        "HydroMechanical_State",
    ] = "RECOVERY_AND_UPLIFT"

    # ========================================================
    # 6. Historical column order
    # ========================================================

    base_columns = [
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

    transition_columns = [
        "Prev_MonthIndex",
        "Prev_Head_m",
        "Prev_CumSubs_mm",
        "Prev_StackYear",
        "Exact_Consecutive_Month",
        "Cross_Stack",
        "GW_Decline_m",
        "Subsidence_Increment_mm",
        "Primary_Valid",
        "HydroMechanical_State",
    ]

    monthly = monthly[
        base_columns
        + transition_columns
    ]

    # ========================================================
    # 7. Recovery / continued-subsidence summary
    # ========================================================

    summary_rows = []

    for well, g in monthly.groupby(
        "Piezometer",
        sort=True,
    ):

        g = (
            g.sort_values("MonthIndex")
            .copy()
        )

        primary = g[
            g["Primary_Valid"]
        ].copy()

        recovery_mask = (
            primary["GW_Decline_m"] < 0
        )

        decline_mask = (
            primary["GW_Decline_m"] > 0
        )

        recovery_subs_mask = (
            recovery_mask
            &
            (
                primary[
                    "Subsidence_Increment_mm"
                ] > 0
            )
        )

        decline_subs_mask = (
            decline_mask
            &
            (
                primary[
                    "Subsidence_Increment_mm"
                ] > 0
            )
        )

        n_valid = int(
            len(primary)
        )

        n_recovery = int(
            recovery_mask.sum()
        )

        n_recovery_subs = int(
            recovery_subs_mask.sum()
        )

        n_decline = int(
            decline_mask.sum()
        )

        n_decline_subs = int(
            decline_subs_mask.sum()
        )

        recovery_positive_subs = (
            primary.loc[
                recovery_subs_mask,
                "Subsidence_Increment_mm",
            ]
        )

        if n_recovery > 0:

            pct_recovery_subs = (
                100.0
                * n_recovery_subs
                / n_recovery
            )

        else:
            pct_recovery_subs = np.nan

        if n_decline > 0:

            pct_decline_subs = (
                100.0
                * n_decline_subs
                / n_decline
            )

        else:
            pct_decline_subs = np.nan

        if len(
            recovery_positive_subs
        ):

            median_recovery_subs = float(
                recovery_positive_subs.median()
            )

            total_positive_recovery_subs = float(
                recovery_positive_subs.sum()
            )

        else:

            median_recovery_subs = np.nan
            total_positive_recovery_subs = 0.0

        # Run is evaluated on the complete ordered monthly
        # sequence so invalid/cross-stack months break a run.
        recovery_plus_subs_run = (
            g["HydroMechanical_State"]
            ==
            "RECOVERY_BUT_SUBSIDENCE"
        )

        longest_run = _longest_true_run(
            recovery_plus_subs_run
        )

        summary_rows.append({
            "Piezometer":
                well,

            "Valid_Monthly_Transitions":
                n_valid,

            "GW_Recovery_Months":
                n_recovery,

            "Recovery_But_Subsidence_Months":
                n_recovery_subs,

            "Pct_Recovery_Months_Still_Subsiding":
                float(
                    pct_recovery_subs
                ),

            "Median_Subsidence_During_Recovery_mm":
                median_recovery_subs,

            "Total_Positive_Subsidence_During_Recovery_mm":
                total_positive_recovery_subs,

            "Longest_RecoveryPlusSubsidence_Run_months":
                longest_run,

            "GW_Decline_Months":
                n_decline,

            "Decline_And_Subsidence_Months":
                n_decline_subs,

            "Pct_Decline_Months_Subsiding":
                float(
                    pct_decline_subs
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # 8. Structural QC
    # ========================================================

    if len(monthly) != 3589:
        raise RuntimeError(
            "Stage-14 monthly row-count QC failed."
        )

    if (
        monthly["Piezometer"].nunique()
        != 37
    ):
        raise RuntimeError(
            "Stage-14 well-count QC failed."
        )

    if int(
        monthly[
            "Exact_Consecutive_Month"
        ].sum()
    ) != 3552:
        raise RuntimeError(
            "Exact-consecutive-month QC failed."
        )

    if int(
        monthly["Cross_Stack"].sum()
    ) != 296:
        raise RuntimeError(
            "Cross-stack QC failed."
        )

    if int(
        monthly["Primary_Valid"].sum()
    ) != 3033:
        raise RuntimeError(
            "Primary-valid QC failed."
        )

    expected_states = {
        "DECLINE_AND_SUBSIDENCE":
            940,
        "RECOVERY_BUT_SUBSIDENCE":
            757,
        "RECOVERY_AND_UPLIFT":
            664,
        "DECLINE_BUT_UPLIFT":
            623,
        "OTHER_OR_INVALID":
            605,
    }

    observed_states = (
        monthly[
            "HydroMechanical_State"
        ]
        .value_counts()
        .to_dict()
    )

    if observed_states != expected_states:
        raise RuntimeError(
            "State-count QC failed: "
            f"{observed_states}"
        )

    if len(summary) != 37:
        raise RuntimeError(
            "Stage-14 summary row-count QC failed."
        )

    valid_dist = (
        summary[
            "Valid_Monthly_Transitions"
        ]
        .value_counts()
        .to_dict()
    )

    if valid_dist != {
        82: 36,
        81: 1,
    }:
        raise RuntimeError(
            "Valid-transition distribution failed: "
            f"{valid_dist}"
        )

    # ========================================================
    # 9. Outputs
    # ========================================================

    out_dir = (
        regen
        / "stage14"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly_out = (
        out_dir
        / MONTHLY_OUTPUT
    )

    summary_out = (
        out_dir
        / SUMMARY_OUTPUT
    )

    monthly.to_csv(
        monthly_out,
        index=False,
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    return (
        monthly,
        summary,
        monthly_out,
        summary_out,
    )


def main():

    cfg = load_config()

    (
        monthly,
        summary,
        monthly_out,
        summary_out,
    ) = reconstruct_stage14(cfg)

    print("=" * 105)
    print(
        "STAGE 14 — HYDROMECHANICAL STATES"
    )
    print("=" * 105)

    print(
        "Monthly rows       :",
        len(monthly)
    )

    print(
        "Primary valid      :",
        int(
            monthly[
                "Primary_Valid"
            ].sum()
        )
    )

    print(
        "Cross-stack rows   :",
        int(
            monthly[
                "Cross_Stack"
            ].sum()
        )
    )

    print("\nState counts:")
    print(
        monthly[
            "HydroMechanical_State"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSummary rows       :",
        len(summary)
    )

    print("\nValid transition distribution:")
    print(
        summary[
            "Valid_Monthly_Transitions"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutputs:")
    print(monthly_out)
    print(summary_out)

    print(
        "\nSTAGE 14 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
