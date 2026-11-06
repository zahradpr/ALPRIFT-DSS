from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config


OUTPUT_FILENAME = "11_SBAS_Stack_Stitching_QC.csv"


def reconstruct_stage11(cfg: dict | None = None):

    if cfg is None:
        cfg = load_config()

    regenerated_root = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage02_path = (
        regenerated_root
        / "stage02"
        / "02_InSAR_Epochs_41Wells.csv"
    )

    if not stage02_path.exists():
        raise FileNotFoundError(stage02_path)

    s2 = pd.read_csv(stage02_path)

    required = {
        "StackYear",
        "AcquisitionDate",
    }

    missing = required - set(s2.columns)

    if missing:
        raise RuntimeError(
            f"Stage 02 missing columns: {sorted(missing)}"
        )

    # One unique acquisition-date set per SBAS stack.
    epochs = (
        s2[
            ["StackYear", "AcquisitionDate"]
        ]
        .drop_duplicates()
        .copy()
    )

    epochs["AcquisitionDate"] = pd.to_datetime(
        epochs["AcquisitionDate"]
    )

    stack_years = sorted(
        epochs["StackYear"]
        .astype(int)
        .unique()
    )

    rows = []

    for from_year, to_year in zip(
        stack_years[:-1],
        stack_years[1:],
    ):

        prev_dates = (
            epochs.loc[
                epochs["StackYear"] == from_year,
                "AcquisitionDate",
            ]
            .drop_duplicates()
            .sort_values()
        )

        next_dates = (
            epochs.loc[
                epochs["StackYear"] == to_year,
                "AcquisitionDate",
            ]
            .drop_duplicates()
            .sort_values()
        )

        prev_end = prev_dates.max()
        next_start = next_dates.min()

        gap_days = int(
            (next_start - prev_end).days
        )

        common = sorted(
            set(prev_dates)
            & set(next_dates)
        )

        common_dates = len(common)

        if common_dates > 0:

            status = "EXACT_COMMON_DATE"

            exact_common_dates = ",".join(
                pd.Timestamp(d).strftime(
                    "%Y-%m-%d"
                )
                for d in common
            )

        elif gap_days < 0:

            status = (
                "TEMPORAL_OVERLAP_NO_EXACT_DATE"
            )

            exact_common_dates = None

        elif gap_days <= 24:

            status = "SHORT_GAP"
            exact_common_dates = None

        else:

            status = "LONG_GAP"
            exact_common_dates = None

        rows.append({
            "From":
                int(from_year),

            "To":
                int(to_year),

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

            "Common_dates":
                int(common_dates),

            "Status":
                status,

            "Exact_common_dates":
                exact_common_dates,
        })

    result = pd.DataFrame(
        rows,
        columns=[
            "From",
            "To",
            "Prev_End",
            "Next_Start",
            "Gap_days",
            "Common_dates",
            "Status",
            "Exact_common_dates",
        ],
    )

    # --------------------------------------------------------
    # Structural QC
    # --------------------------------------------------------

    if len(result) != 8:
        raise RuntimeError(
            f"Expected 8 stack boundaries; "
            f"found {len(result)}."
        )

    expected_status_counts = {
        "SHORT_GAP": 4,
        "EXACT_COMMON_DATE": 2,
        "LONG_GAP": 1,
        "TEMPORAL_OVERLAP_NO_EXACT_DATE": 1,
    }

    observed_status_counts = (
        result["Status"]
        .value_counts()
        .to_dict()
    )

    if observed_status_counts != expected_status_counts:
        raise RuntimeError(
            "Stage-11 status-count QC failed: "
            f"{observed_status_counts}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage11"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        out_dir
        / OUTPUT_FILENAME
    )

    result.to_csv(
        out,
        index=False,
    )

    return result, out


def main():

    cfg = load_config()

    result, out = (
        reconstruct_stage11(cfg)
    )

    print("=" * 100)
    print(
        "STAGE 11 — SBAS STACK STITCHING QC"
    )
    print("=" * 100)

    print(result.to_string(index=False))

    print("\nStatus counts:")
    print(
        result["Status"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:")
    print(out)

    print(
        "\nSTAGE 11 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
