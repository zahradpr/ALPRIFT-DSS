from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .stage02_insar_epochs import frozen_file_for_stage


def reconstruct_stage04(cfg: dict | None = None):

    if cfg is None:
        cfg = load_config()

    regenerated_root = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage01_path = (
        regenerated_root
        / "stage01"
        / "01_Groundwater_Monthly_41Wells.csv"
    )

    stage03_path = (
        regenerated_root
        / "stage03"
        / "03_InSAR_Monthly_41Wells.csv"
    )

    if not stage01_path.exists():
        raise FileNotFoundError(stage01_path)

    if not stage03_path.exists():
        raise FileNotFoundError(stage03_path)

    reference_path = frozen_file_for_stage(
        cfg,
        4
    )

    s1 = pd.read_csv(stage01_path)
    s3 = pd.read_csv(stage03_path)
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
        "Head_m",
        "Prev_Head_m",
        "GW_Decline_m",
        "Valid_Change_Pair",
    ]

    if list(reference.columns) != expected_columns:
        raise RuntimeError(
            "Unexpected Stage-04 frozen schema:\n"
            f"{list(reference.columns)}"
        )

    # ----------------------------------------------------------
    # Groundwater fields required for temporal pairing
    # ----------------------------------------------------------

    join_keys = [
        "Piezometer",
        "JYear",
        "JMonth",
        "MonthIndex",
    ]

    gw = s1[
        join_keys
        + [
            "Head_m",
            "Prev_Head_m",
            "GW_Decline_m",
        ]
    ].copy()

    # Stage 03 defines the temporal support of Stage 04.
    paired = s3.merge(
        gw,
        on=join_keys,
        how="left",
        validate="many_to_one"
    )

    # ----------------------------------------------------------
    # Valid change pair
    #
    # A change-pair is valid only if both current and previous
    # states exist for groundwater and InSAR.
    # ----------------------------------------------------------

    paired["Valid_Change_Pair"] = (
        paired["Head_m"].notna()
        & paired["Prev_Head_m"].notna()
        & paired["InSAR_Monthly_mm"].notna()
        & paired["Prev_InSAR_mm"].notna()
    )

    # Frozen Stage-04 canonical ordering.
    paired = (
        paired
        .sort_values(
            [
                "Piezometer",
                "StackYear",
                "MonthIndex",
            ]
        )
        .reset_index(drop=True)
    )

    regenerated = paired[
        expected_columns
    ].copy()

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != 4715:
        raise RuntimeError(
            f"Stage 04 expected 4715 rows; "
            f"found {len(regenerated)}."
        )

    if regenerated[
        [
            "Piezometer",
            "StackYear",
            "MonthIndex",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Stage-04 monthly pair records."
        )

    valid_count = int(
        regenerated[
            "Valid_Change_Pair"
        ].sum()
    )

    if valid_count != 3650:
        raise RuntimeError(
            f"Expected 3650 valid change pairs; "
            f"found {valid_count}."
        )

    # Independent logical QC:
    expected_valid = (
        regenerated["GW_Decline_m"].notna()
        & regenerated["InSAR_Increment_mm"].notna()
    )

    if not (
        regenerated["Valid_Change_Pair"]
        == expected_valid
    ).all():
        raise RuntimeError(
            "Valid_Change_Pair logical QC failed."
        )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage04"
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

    result, out, ref = reconstruct_stage04(
        cfg
    )

    print("=" * 100)
    print("STAGE 04 — GW / INSAR MONTHLY PAIRS")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)
    print("Rows      :", len(result))

    print(
        "Wells     :",
        result["Piezometer"].nunique()
    )

    print(
        "Valid change pairs :",
        int(result["Valid_Change_Pair"].sum())
    )

    print(
        "Invalid pairs      :",
        int((~result["Valid_Change_Pair"]).sum())
    )

    print()
    print("STAGE 04 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
