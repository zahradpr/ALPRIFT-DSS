from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .stage02_insar_epochs import frozen_file_for_stage


def reconstruct_stage06(cfg: dict | None = None):

    if cfg is None:
        cfg = load_config()

    regenerated_root = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage04_path = (
        regenerated_root
        / "stage04"
        / "04_GW_InSAR_Monthly_Pairs.csv"
    )

    stage05_path = (
        regenerated_root
        / "stage05"
        / "05_Piezometer_Coverage_QC.csv"
    )

    if not stage04_path.exists():
        raise FileNotFoundError(stage04_path)

    if not stage05_path.exists():
        raise FileNotFoundError(stage05_path)

    reference_path = frozen_file_for_stage(
        cfg,
        6
    )

    s4 = pd.read_csv(stage04_path)
    s5 = pd.read_csv(stage05_path)
    reference = pd.read_csv(reference_path)

    expected_columns = [
        "Piezometer",
        "MonthIndex",
        "N",
        "Min",
        "Max",
        "Mean",
        "Range_mm",
    ]

    if list(reference.columns) != expected_columns:
        raise RuntimeError(
            "Unexpected Stage-06 frozen schema:\n"
            f"{list(reference.columns)}"
        )

    # ----------------------------------------------------------
    # Eligible wells
    # ----------------------------------------------------------

    eligible = (
        s5.loc[
            s5["Valid_change_pairs"] > 0,
            "Piezometer"
        ]
        .tolist()
    )

    expected_eligible = int(
        cfg["analysis"][
            "expected_lag_eligible_wells"
        ]
    )

    if len(eligible) != expected_eligible:
        raise RuntimeError(
            f"Expected {expected_eligible} eligible wells; "
            f"found {len(eligible)}."
        )

    x = s4[
        s4["Piezometer"].isin(eligible)
    ].copy()

    # ----------------------------------------------------------
    # Overlap increment QC
    #
    # For each piezometer and Jalali month, summarize only
    # cases where at least TWO valid InSAR increments exist.
    # These correspond to months represented by overlapping
    # SBAS stacks with valid increments in both stacks.
    # ----------------------------------------------------------

    overlap = (
        x.groupby(
            ["Piezometer", "MonthIndex"],
            sort=True
        )["InSAR_Increment_mm"]
        .agg(
            N="count",
            Min="min",
            Max="max",
            Mean="mean",
        )
        .reset_index()
    )

    overlap = overlap[
        overlap["N"] >= 2
    ].copy()

    overlap["Range_mm"] = (
        overlap["Max"]
        - overlap["Min"]
    )

    regenerated = (
        overlap[
            expected_columns
        ]
        .sort_values(
            ["Piezometer", "MonthIndex"]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != 74:
        raise RuntimeError(
            f"Expected 74 Stage-06 rows; "
            f"found {len(regenerated)}."
        )

    if regenerated[
        "Piezometer"
    ].nunique() != 37:
        raise RuntimeError(
            "Expected 37 piezometers in Stage 06."
        )

    expected_months = [
        16812,
        16836,
    ]

    observed_months = sorted(
        regenerated[
            "MonthIndex"
        ].unique().tolist()
    )

    if observed_months != expected_months:
        raise RuntimeError(
            f"Unexpected overlap months: "
            f"{observed_months}"
        )

    if not (
        regenerated["N"] == 2
    ).all():
        raise RuntimeError(
            "Stage-06 overlap multiplicity QC failed."
        )

    expected_range = (
        regenerated["Max"]
        - regenerated["Min"]
    )

    if not (
        regenerated["Range_mm"]
        == expected_range
    ).all():
        raise RuntimeError(
            "Stage-06 Range_mm formula QC failed."
        )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage06"
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

    result, out, ref = reconstruct_stage06(
        cfg
    )

    print("=" * 100)
    print("STAGE 06 — OVERLAP INCREMENT QC")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)
    print("Rows      :", len(result))

    print(
        "Wells     :",
        result["Piezometer"].nunique()
    )

    print(
        "Overlap months:",
        sorted(
            result["MonthIndex"]
            .unique()
            .tolist()
        )
    )

    print(
        "All groups have N=2:",
        bool((result["N"] == 2).all())
    )

    print()
    print("STAGE 06 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
