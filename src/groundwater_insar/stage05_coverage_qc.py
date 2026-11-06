from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .stage02_insar_epochs import frozen_file_for_stage


def reconstruct_stage05(cfg: dict | None = None):

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

    if not stage04_path.exists():
        raise FileNotFoundError(stage04_path)

    reference_path = frozen_file_for_stage(
        cfg,
        5
    )

    s4 = pd.read_csv(stage04_path)
    reference = pd.read_csv(reference_path)

    expected_columns = [
        "Piezometer",
        "Monthly_records",
        "Valid_InSAR_months",
        "Valid_InSAR_increments",
        "Valid_GW_declines",
        "Valid_change_pairs",
        "Pair_coverage_pct",
    ]

    if list(reference.columns) != expected_columns:
        raise RuntimeError(
            "Unexpected Stage-05 frozen schema:\n"
            f"{list(reference.columns)}"
        )

    # ----------------------------------------------------------
    # Coverage summary by piezometer
    # ----------------------------------------------------------

    coverage = (
        s4.groupby(
            "Piezometer",
            sort=True
        )
        .agg(
            Monthly_records=(
                "MonthIndex",
                "size"
            ),
            Valid_InSAR_months=(
                "InSAR_Monthly_mm",
                "count"
            ),
            Valid_InSAR_increments=(
                "InSAR_Increment_mm",
                "count"
            ),
            Valid_GW_declines=(
                "GW_Decline_m",
                "count"
            ),
            Valid_change_pairs=(
                "Valid_Change_Pair",
                "sum"
            ),
        )
        .reset_index()
    )

    coverage["Valid_change_pairs"] = (
        coverage["Valid_change_pairs"]
        .astype(int)
    )

    coverage["Pair_coverage_pct"] = (
        100.0
        * coverage["Valid_change_pairs"]
        / coverage["Monthly_records"]
    )

    regenerated = coverage[
        expected_columns
    ].copy()

    # ----------------------------------------------------------
    # QC
    # ----------------------------------------------------------

    expected_wells = int(
        cfg["analysis"]["official_wells"]
    )

    if len(regenerated) != expected_wells:
        raise RuntimeError(
            f"Expected {expected_wells} wells; "
            f"found {len(regenerated)}."
        )

    if regenerated["Piezometer"].duplicated().any():
        raise RuntimeError(
            "Duplicate piezometer rows in Stage 05."
        )

    if not (
        regenerated["Monthly_records"] == 115
    ).all():
        raise RuntimeError(
            "Unexpected Stage-05 monthly record count."
        )

    zero_insar = (
        regenerated.loc[
            regenerated["Valid_InSAR_months"] == 0,
            "Piezometer"
        ]
        .tolist()
    )

    expected_zero = [
        "P06",
        "P24",
        "P28",
        "P47",
    ]

    if zero_insar != expected_zero:
        raise RuntimeError(
            "Unexpected zero-InSAR wells: "
            f"{zero_insar}"
        )

    # Coverage formula QC
    expected_pct = (
        100.0
        * regenerated["Valid_change_pairs"]
        / regenerated["Monthly_records"]
    )

    if not (
        regenerated["Pair_coverage_pct"]
        == expected_pct
    ).all():
        raise RuntimeError(
            "Pair coverage formula QC failed."
        )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage05"
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

    result, out, ref = reconstruct_stage05(
        cfg
    )

    print("=" * 100)
    print("STAGE 05 — PIEZOMETER COVERAGE QC")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)
    print("Wells     :", len(result))

    zero = result.loc[
        result["Valid_change_pairs"] == 0,
        "Piezometer"
    ].tolist()

    print(
        "Zero valid-pair wells:",
        ", ".join(zero)
    )

    positive = result[
        result["Valid_change_pairs"] > 0
    ]

    print(
        "Wells with usable pairs:",
        len(positive)
    )

    print(
        "Minimum nonzero pair coverage:",
        positive["Pair_coverage_pct"].min()
    )

    print()
    print("STAGE 05 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
