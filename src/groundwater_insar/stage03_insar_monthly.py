from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .stage02_insar_epochs import frozen_file_for_stage


def reconstruct_stage03(
    cfg: dict | None = None
):
    if cfg is None:
        cfg = load_config()

    # ----------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------

    stage02_path = (
        Path(cfg["paths"]["regenerated_root"])
        / "stage02"
        / "02_InSAR_Epochs_41Wells.csv"
    )

    if not stage02_path.exists():
        raise FileNotFoundError(
            f"Regenerated Stage 02 not found: {stage02_path}"
        )

    reference_path = frozen_file_for_stage(
        cfg,
        3
    )

    s2 = pd.read_csv(stage02_path)
    reference = pd.read_csv(reference_path)

    expected_columns = [
        "StackYear",
        "Piezometer",
        "JYear",
        "JMonth",
        "MonthIndex",
        "InSAR_Monthly_mm",
        "N_Epochs",
        "First_Acquisition",
        "Last_Acquisition",
        "Prev_MonthIndex",
        "Prev_InSAR_mm",
        "InSAR_Increment_mm",
    ]

    if list(reference.columns) != expected_columns:
        raise RuntimeError(
            "Unexpected Stage-03 frozen schema:\n"
            f"{list(reference.columns)}"
        )

    required_stage02 = {
        "StackYear",
        "Piezometer",
        "AcquisitionDate",
        "JYear",
        "JMonth",
        "MonthIndex",
        "InSAR_mm",
    }

    missing = required_stage02 - set(s2.columns)

    if missing:
        raise RuntimeError(
            f"Stage-02 columns missing: {sorted(missing)}"
        )

    # ----------------------------------------------------------
    # Monthly aggregation
    #
    # Rules reconstructed from the frozen terminal baseline:
    #
    # InSAR_Monthly_mm = mean of valid epoch values
    # N_Epochs         = number of valid epoch values
    #
    # First/Last acquisition describe all acquisitions in
    # that Jalali month, even if InSAR values are NaN.
    # ----------------------------------------------------------

    keys = [
        "StackYear",
        "Piezometer",
        "JYear",
        "JMonth",
        "MonthIndex",
    ]

    monthly = (
        s2.groupby(
            keys,
            sort=True,
            dropna=False
        )
        .agg(
            InSAR_Monthly_mm=(
                "InSAR_mm",
                "mean"
            ),
            N_Epochs=(
                "InSAR_mm",
                "count"
            ),
            First_Acquisition=(
                "AcquisitionDate",
                "min"
            ),
            Last_Acquisition=(
                "AcquisitionDate",
                "max"
            ),
        )
        .reset_index()
    )

    # Canonical deterministic ordering.
    monthly = (
        monthly
        .sort_values(
            [
                "StackYear",
                "Piezometer",
                "MonthIndex"
            ]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Previous-month variables
    #
    # IMPORTANT:
    # The chain is reset at the beginning of every
    # StackYear × Piezometer combination.
    # ----------------------------------------------------------

    group_keys = [
        "StackYear",
        "Piezometer",
    ]

    monthly["Prev_MonthIndex"] = (
        monthly
        .groupby(
            group_keys,
            sort=False
        )["MonthIndex"]
        .shift(1)
    )

    monthly["Prev_InSAR_mm"] = (
        monthly
        .groupby(
            group_keys,
            sort=False
        )["InSAR_Monthly_mm"]
        .shift(1)
    )

    # Positive increment = increase in cumulative
    # subsidence relative to the preceding monthly state.
    monthly["InSAR_Increment_mm"] = (
        monthly["InSAR_Monthly_mm"]
        - monthly["Prev_InSAR_mm"]
    )

    regenerated = monthly[
        expected_columns
    ].copy()

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != len(reference):
        raise RuntimeError(
            "Stage-03 row-count mismatch: "
            f"generated={len(regenerated)}, "
            f"reference={len(reference)}"
        )

    if regenerated[
        ["StackYear", "Piezometer", "MonthIndex"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Stage-03 monthly records."
        )

    # Number of valid Stage-02 observations must be exactly
    # preserved by monthly N_Epochs.
    n_valid_s2 = int(
        s2["InSAR_mm"].notna().sum()
    )

    n_monthly = int(
        regenerated["N_Epochs"].sum()
    )

    if n_valid_s2 != n_monthly:
        raise RuntimeError(
            "Stage-03 valid-epoch conservation failed: "
            f"Stage02={n_valid_s2}, "
            f"Stage03={n_monthly}"
        )

    # First row of every stack/well group must have no
    # previous-month state.
    first_mask = (
        regenerated
        .groupby(
            group_keys,
            sort=False
        )
        .cumcount()
        == 0
    )

    if not regenerated.loc[
        first_mask,
        "Prev_MonthIndex"
    ].isna().all():
        raise RuntimeError(
            "Stage-03 group-boundary QC failed."
        )

    # Within every stack/well group, monthly temporal
    # continuity must be exactly one Jalali month.
    nonfirst = ~first_mask

    gap = (
        regenerated.loc[
            nonfirst,
            "MonthIndex"
        ]
        -
        regenerated.loc[
            nonfirst,
            "Prev_MonthIndex"
        ]
    )

    if not (gap == 1).all():
        raise RuntimeError(
            "Stage-03 monthly continuity QC failed."
        )

    # ----------------------------------------------------------
    # Verify independently derived key structure against
    # the frozen contract without copying scientific values.
    # ----------------------------------------------------------

    key_cols = [
        "StackYear",
        "Piezometer",
        "JYear",
        "JMonth",
        "MonthIndex",
    ]

    ref_keys = (
        reference[key_cols]
        .sort_values(
            [
                "StackYear",
                "Piezometer",
                "MonthIndex"
            ]
        )
        .reset_index(drop=True)
    )

    new_keys = (
        regenerated[key_cols]
        .sort_values(
            [
                "StackYear",
                "Piezometer",
                "MonthIndex"
            ]
        )
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        ref_keys,
        new_keys,
        check_dtype=False
    )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        Path(cfg["paths"]["regenerated_root"])
        / "stage03"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out = (
        out_dir
        / reference_path.name
    )

    regenerated.to_csv(
        out,
        index=False
    )

    return (
        regenerated,
        out,
        reference_path
    )


def main():

    cfg = load_config()

    result, out, ref = reconstruct_stage03(
        cfg
    )

    print("=" * 100)
    print("STAGE 03 — MONTHLY INSAR")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)

    print("Rows      :", len(result))

    print(
        "Wells     :",
        result["Piezometer"].nunique()
    )

    print(
        "Stacks    :",
        result["StackYear"].nunique()
    )

    print(
        "Valid monthly values :",
        int(
            result[
                "InSAR_Monthly_mm"
            ].notna().sum()
        )
    )

    print(
        "Missing monthly values:",
        int(
            result[
                "InSAR_Monthly_mm"
            ].isna().sum()
        )
    )

    print(
        "Valid increments      :",
        int(
            result[
                "InSAR_Increment_mm"
            ].notna().sum()
        )
    )

    print(
        "Missing increments    :",
        int(
            result[
                "InSAR_Increment_mm"
            ].isna().sum()
        )
    )

    print("\nN_Epochs distribution:")
    print(
        result["N_Epochs"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("STAGE 03 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
