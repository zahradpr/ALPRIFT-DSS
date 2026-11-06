from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config


EPOCH_OUTPUT = (
    "12_Stitched_Cumulative_Epochs_1396_1403_37Wells.csv"
)

QC_OUTPUT = (
    "12_Stitching_Boundary_QC_37Wells.csv"
)


def _interp_at_date(
    dates: pd.Series,
    values: pd.Series,
    target: pd.Timestamp,
) -> float:

    x = pd.to_datetime(dates).astype("int64").to_numpy(
        dtype=float
    )

    y = values.to_numpy(dtype=float)

    t = float(
        pd.Timestamp(target).value
    )

    valid = np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise RuntimeError(
            "Insufficient points for interpolation."
        )

    order = np.argsort(x)

    return float(
        np.interp(
            t,
            x[order],
            y[order],
        )
    )


def reconstruct_stage12(
    cfg: dict | None = None,
):

    if cfg is None:
        cfg = load_config()

    regen = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage02_path = (
        regen
        / "stage02"
        / "02_InSAR_Epochs_41Wells.csv"
    )

    stage07_path = (
        regen
        / "stage07"
        / "07_Canonical_Monthly_Pairs_37Wells.csv"
    )

    for p in [
        stage02_path,
        stage07_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    s2 = pd.read_csv(stage02_path)

    s7 = pd.read_csv(
        stage07_path,
        usecols=["Piezometer"],
    )

    eligible_wells = sorted(
        s7["Piezometer"]
        .drop_duplicates()
        .tolist()
    )

    if len(eligible_wells) != 37:
        raise RuntimeError(
            f"Expected 37 eligible wells; "
            f"found {len(eligible_wells)}."
        )

    s2 = s2[
        s2["Piezometer"].isin(
            eligible_wells
        )
        &
        s2["StackYear"].between(
            1396,
            1403,
        )
    ].copy()

    s2["AcquisitionDate"] = pd.to_datetime(
        s2["AcquisitionDate"]
    )

    expected_rows = (
        37 * 8 * 16
    )

    if len(s2) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} source rows; "
            f"found {len(s2)}."
        )

    stitched_parts = []
    qc_rows = []

    # ========================================================
    # Well-by-well sequential stitching
    # ========================================================

    for well in eligible_wells:

        w = (
            s2[
                s2["Piezometer"]
                == well
            ]
            .sort_values(
                [
                    "StackYear",
                    "AcquisitionDate",
                ]
            )
            .copy()
        )

        previous_stitched = None
        previous_stack = None

        for stack_year in range(
            1396,
            1404,
        ):

            cur = (
                w[
                    w["StackYear"]
                    == stack_year
                ]
                .sort_values(
                    "AcquisitionDate"
                )
                .copy()
            )

            if len(cur) != 16:
                raise RuntimeError(
                    f"{well} stack {stack_year}: "
                    f"expected 16 epochs, "
                    f"found {len(cur)}."
                )

            # ------------------------------------------------
            # Base stack: 1396
            # ------------------------------------------------

            if stack_year == 1396:

                first_raw = float(
                    cur.iloc[0]["InSAR_mm"]
                )

                offset = -first_raw

                method = "BASE_1396"

                cur[
                    "Stitched_Cumulative_mm"
                ] = (
                    cur["InSAR_mm"]
                    + offset
                )

                cur["Stitch_Method"] = (
                    method
                )

                cur["Offset_mm"] = (
                    offset
                )

                previous_stitched = (
                    cur.copy()
                )

                previous_stack = (
                    stack_year
                )

                stitched_parts.append(cur)

                continue

            # ------------------------------------------------
            # Boundary information
            # ------------------------------------------------

            prev = (
                previous_stitched
                .sort_values(
                    "AcquisitionDate"
                )
                .copy()
            )

            prev_end = pd.Timestamp(
                prev[
                    "AcquisitionDate"
                ].max()
            )

            next_start = pd.Timestamp(
                cur[
                    "AcquisitionDate"
                ].min()
            )

            gap_days = int(
                (
                    next_start
                    - prev_end
                ).days
            )

            common_dates = sorted(
                set(
                    prev[
                        "AcquisitionDate"
                    ]
                )
                &
                set(
                    cur[
                        "AcquisitionDate"
                    ]
                )
            )

            exact_valid_dates = []

            # ------------------------------------------------
            # Exact common date(s)
            # ------------------------------------------------

            for dt in common_dates:

                prev_value = prev.loc[
                    prev["AcquisitionDate"]
                    == dt,
                    "Stitched_Cumulative_mm",
                ]

                next_value = cur.loc[
                    cur["AcquisitionDate"]
                    == dt,
                    "InSAR_mm",
                ]

                if (
                    len(prev_value)
                    and len(next_value)
                    and np.isfinite(
                        prev_value.iloc[0]
                    )
                    and np.isfinite(
                        next_value.iloc[0]
                    )
                ):
                    exact_valid_dates.append(
                        pd.Timestamp(dt)
                    )

            if exact_valid_dates:

                differences = []

                for dt in (
                    exact_valid_dates
                ):

                    prev_value = float(
                        prev.loc[
                            prev[
                                "AcquisitionDate"
                            ]
                            == dt,
                            "Stitched_Cumulative_mm",
                        ].iloc[0]
                    )

                    next_value = float(
                        cur.loc[
                            cur[
                                "AcquisitionDate"
                            ]
                            == dt,
                            "InSAR_mm",
                        ].iloc[0]
                    )

                    differences.append(
                        prev_value
                        - next_value
                    )

                offset = float(
                    np.median(
                        differences
                    )
                )

                method = (
                    "EXACT_COMMON_DATE_MEDIAN"
                )

                anchor = ",".join(
                    dt.strftime(
                        "%Y-%m-%d"
                    )
                    for dt in (
                        exact_valid_dates
                    )
                )

            # ------------------------------------------------
            # Temporal overlap, no exact common date
            # ------------------------------------------------

            elif gap_days < 0:

                midpoint = (
                    next_start
                    + (
                        prev_end
                        - next_start
                    ) / 2
                )

                prev_interp = (
                    _interp_at_date(
                        prev[
                            "AcquisitionDate"
                        ],
                        prev[
                            "Stitched_Cumulative_mm"
                        ],
                        midpoint,
                    )
                )

                next_interp = (
                    _interp_at_date(
                        cur[
                            "AcquisitionDate"
                        ],
                        cur["InSAR_mm"],
                        midpoint,
                    )
                )

                offset = float(
                    prev_interp
                    - next_interp
                )

                method = (
                    "OVERLAP_INTERPOLATED_MIDPOINT"
                )

                anchor = (
                    midpoint.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )

            # ------------------------------------------------
            # Positive temporal gap:
            # zero bridge between end/start.
            # ------------------------------------------------

            else:

                prev_end_value = float(
                    prev.loc[
                        prev[
                            "AcquisitionDate"
                        ]
                        == prev_end,
                        "Stitched_Cumulative_mm",
                    ].iloc[0]
                )

                next_start_value = float(
                    cur.loc[
                        cur[
                            "AcquisitionDate"
                        ]
                        == next_start,
                        "InSAR_mm",
                    ].iloc[0]
                )

                offset = float(
                    prev_end_value
                    - next_start_value
                )

                method = (
                    "SHORT_GAP_ZERO_BRIDGE"
                )

                anchor = (
                    f"{prev_end:%Y-%m-%d} "
                    f"-> "
                    f"{next_start:%Y-%m-%d}"
                )

            # ------------------------------------------------
            # Apply offset
            # ------------------------------------------------

            cur[
                "Stitched_Cumulative_mm"
            ] = (
                cur["InSAR_mm"]
                + offset
            )

            cur["Stitch_Method"] = (
                method
            )

            cur["Offset_mm"] = (
                offset
            )

            # ------------------------------------------------
            # Exact-common residual QC
            # ------------------------------------------------

            if exact_valid_dates:

                residuals = []

                for dt in (
                    exact_valid_dates
                ):

                    prev_value = float(
                        prev.loc[
                            prev[
                                "AcquisitionDate"
                            ]
                            == dt,
                            "Stitched_Cumulative_mm",
                        ].iloc[0]
                    )

                    next_value = float(
                        cur.loc[
                            cur[
                                "AcquisitionDate"
                            ]
                            == dt,
                            "Stitched_Cumulative_mm",
                        ].iloc[0]
                    )

                    residuals.append(
                        abs(
                            prev_value
                            - next_value
                        )
                    )

                max_abs_residual = float(
                    np.max(residuals)
                )

                mean_abs_residual = float(
                    np.mean(residuals)
                )

            else:

                max_abs_residual = np.nan
                mean_abs_residual = np.nan

            qc_rows.append({
                "Piezometer":
                    well,

                "FromStack":
                    int(
                        previous_stack
                    ),

                "ToStack":
                    int(
                        stack_year
                    ),

                "Prev_End":
                    prev_end.strftime(
                        "%Y-%m-%d"
                    ),

                "Next_Start":
                    next_start.strftime(
                        "%Y-%m-%d"
                    ),

                "Gap_days":
                    gap_days,

                "Exact_Common_Valid_Dates":
                    len(
                        exact_valid_dates
                    ),

                "Method":
                    method,

                "Anchor":
                    anchor,

                "Applied_Offset_mm":
                    offset,

                "Exact_Common_MaxAbsResidual_mm":
                    max_abs_residual,

                "Exact_Common_MeanAbsResidual_mm":
                    mean_abs_residual,
            })

            stitched_parts.append(cur)

            previous_stitched = (
                cur.copy()
            )

            previous_stack = (
                stack_year
            )

    # ========================================================
    # Final epoch table
    # ========================================================

    stitched = pd.concat(
        stitched_parts,
        ignore_index=True,
    )

    stitched = (
        stitched.sort_values(
            [
                "Piezometer",
                "StackYear",
                "AcquisitionDate",
            ]
        )
        .reset_index(drop=True)
    )

    stitched[
        "AcquisitionDate"
    ] = (
        stitched[
            "AcquisitionDate"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    epoch_columns = [
        "StackYear",
        "Piezometer",
        "AcquisitionDate",
        "JYear",
        "JMonth",
        "JDay",
        "MonthIndex",
        "InSAR_mm",
        "Raster",
        "Stitched_Cumulative_mm",
        "Stitch_Method",
        "Offset_mm",
    ]

    stitched = stitched[
        epoch_columns
    ]

    qc = pd.DataFrame(
        qc_rows
    )

    qc = (
        qc.sort_values(
            [
                "Piezometer",
                "FromStack",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Structural QC
    # ========================================================

    if len(stitched) != 4736:
        raise RuntimeError(
            "Stage-12 epoch row-count QC failed."
        )

    if len(qc) != 259:
        raise RuntimeError(
            "Stage-12 boundary row-count QC failed."
        )

    method_counts = (
        qc["Method"]
        .value_counts()
        .to_dict()
    )

    expected_methods = {
        "SHORT_GAP_ZERO_BRIDGE":
            148,
        "EXACT_COMMON_DATE_MEDIAN":
            74,
        "OVERLAP_INTERPOLATED_MIDPOINT":
            37,
    }

    if method_counts != expected_methods:
        raise RuntimeError(
            "Stage-12 method-count QC failed: "
            f"{method_counts}"
        )

    # ========================================================
    # Write outputs
    # ========================================================

    out_dir = (
        regen
        / "stage12"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_out = (
        out_dir
        / EPOCH_OUTPUT
    )

    qc_out = (
        out_dir
        / QC_OUTPUT
    )

    stitched.to_csv(
        epoch_out,
        index=False,
    )

    qc.to_csv(
        qc_out,
        index=False,
    )

    return (
        stitched,
        qc,
        epoch_out,
        qc_out,
    )


def main():

    cfg = load_config()

    (
        stitched,
        qc,
        epoch_out,
        qc_out,
    ) = reconstruct_stage12(cfg)

    print("=" * 105)
    print(
        "STAGE 12 — STITCHED CUMULATIVE SBAS"
    )
    print("=" * 105)

    print(
        "Epoch rows        :",
        len(stitched)
    )

    print(
        "Wells             :",
        stitched[
            "Piezometer"
        ].nunique()
    )

    print(
        "Boundary QC rows  :",
        len(qc)
    )

    print("\nBoundary methods:")
    print(
        qc["Method"]
        .value_counts()
        .to_string()
    )

    print("\nEpoch stitch methods:")
    print(
        stitched[
            "Stitch_Method"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOutputs:")
    print(epoch_out)
    print(qc_out)

    print(
        "\nSTAGE 12 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
