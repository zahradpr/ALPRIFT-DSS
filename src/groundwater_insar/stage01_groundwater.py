from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import (
    load_config,
    reference_root,
)


# =====================================================================
# Semantic column aliases
# =====================================================================

ALIASES = {

    "well": [
        "Piezometer",
        "Well",
        "Well_ID",
        "Piezometer_ID"
    ],

    "year": [
        "JYear",
        "Year"
    ],

    "month": [
        "JMonth",
        "Month"
    ],

    "month_index": [
        "MonthIndex",
        "Month_Index"
    ],

    "head": [
        "Head_m",
        "Head",
        "Groundwater_Head_m",
        "GW_Head_m"
    ],

    "decline": [
        "GW_Decline_m",
        "Groundwater_Decline_m",
        "Decline_m",
        "DeltaH_m",
        "dH_m"
    ],
}


def frozen_stage01_path(
    cfg: dict
) -> Path:

    version = cfg["analysis"][
        "frozen_reference_version"
    ]

    root = (
        reference_root(cfg)
        / version
    )

    manifest_path = (
        root
        / "baseline_manifest.json"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    files = [
        x["filename"]
        for x in manifest["files"]
        if int(x["stage"]) == 1
    ]

    if len(files) != 1:
        raise RuntimeError(
            "Stage 01 must currently have exactly "
            f"one frozen CSV; found: {files}"
        )

    return (
        root
        / "csv"
        / files[0]
    )


def find_column(
    columns,
    semantic: str,
    required=True
):

    for candidate in ALIASES[semantic]:

        if candidate in columns:
            return candidate

    if required:
        raise RuntimeError(
            f"Unable to identify '{semantic}' column. "
            f"Reference columns: {list(columns)}"
        )

    return None


def reference_contract(
    reference: pd.DataFrame
) -> dict:

    cols = list(reference.columns)

    contract = {
        "columns": cols,

        "well":
            find_column(
                cols,
                "well"
            ),

        "year":
            find_column(
                cols,
                "year"
            ),

        "month":
            find_column(
                cols,
                "month"
            ),

        "month_index":
            find_column(
                cols,
                "month_index",
                required=False
            ),

        "head":
            find_column(
                cols,
                "head"
            ),

        "decline":
            find_column(
                cols,
                "decline",
                required=False
            ),
    }

    contract["min_year"] = int(
        reference[
            contract["year"]
        ].min()
    )

    contract["max_year"] = int(
        reference[
            contract["year"]
        ].max()
    )

    contract["wells"] = sorted(
        reference[
            contract["well"]
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return contract


def build_long_source(
    cfg: dict
) -> pd.DataFrame:

    source = Path(
        cfg["paths"][
            "piezometer_monthly"
        ]
    )

    raw = pd.read_csv(
        source
    )

    if not {
        "Year",
        "Month"
    }.issubset(raw.columns):

        raise RuntimeError(
            "Raw groundwater source requires "
            "Year and Month columns."
        )

    well_cols = [
        c
        for c in raw.columns
        if c not in {
            "Year",
            "Month"
        }
    ]

    expected_raw = int(
        cfg["analysis"][
            "raw_monthly_wells_expected"
        ]
    )

    if len(well_cols) != expected_raw:

        raise RuntimeError(
            f"Expected {expected_raw} raw wells; "
            f"found {len(well_cols)}."
        )

    excluded = set(
        cfg["analysis"][
            "excluded_from_official_T"
        ]
    )

    official = [
        w
        for w in well_cols
        if w not in excluded
    ]

    expected_official = int(
        cfg["analysis"][
            "official_wells"
        ]
    )

    if len(official) != expected_official:

        raise RuntimeError(
            f"Expected {expected_official} "
            f"official wells; found {len(official)}."
        )

    long = raw.melt(
        id_vars=[
            "Year",
            "Month"
        ],
        value_vars=official,
        var_name="Piezometer",
        value_name="Head_m"
    )

    long["MonthIndex"] = (
        long["Year"].astype(int) * 12
        + long["Month"].astype(int)
        - 1
    )

    long = long.sort_values(
        [
            "Piezometer",
            "MonthIndex"
        ]
    ).reset_index(drop=True)

    # Previous observed monthly record for each piezometer
    long["Prev_MonthIndex"] = (
        long
        .groupby(
            "Piezometer",
            sort=False
        )["MonthIndex"]
        .shift(1)
    )

    long["Prev_Head_m"] = (
        long
        .groupby(
            "Piezometer",
            sort=False
        )["Head_m"]
        .shift(1)
    )

    # Positive value means groundwater decline:
    #
    # ΔH_t = H_(t-1) - H_t
    #
    long["GW_Decline_m"] = (
        long["Prev_Head_m"]
        - long["Head_m"]
    )

    return long


def reconstruct_stage01(
    cfg: dict | None = None
) -> tuple[
    pd.DataFrame,
    Path,
    Path
]:

    if cfg is None:
        cfg = load_config()

    ref_path = frozen_stage01_path(
        cfg
    )

    reference = pd.read_csv(
        ref_path
    )

    contract = reference_contract(
        reference
    )

    source = build_long_source(
        cfg
    )

    # Restrict to the exact temporal/well contract
    # of the frozen Stage 01.
    source = source[
        source["Year"].between(
            contract["min_year"],
            contract["max_year"]
        )
        &
        source["Piezometer"].isin(
            contract["wells"]
        )
    ].copy()

    # ----------------------------------------------------------
    # Re-establish the Stage-01 temporal boundary BEFORE
    # calculating previous-month variables.
    #
    # Baseline rule:
    # - first record of each piezometer in the Stage-01 period:
    #       Prev_MonthIndex = NaN
    #       Prev_Head_m     = NaN
    # - all subsequent records use the immediately preceding
    #   calendar month, regardless of missing Head values.
    # ----------------------------------------------------------

    source = (
        source
        .sort_values(
            [
                "Piezometer",
                "MonthIndex"
            ]
        )
        .reset_index(drop=True)
    )

    source["Prev_MonthIndex"] = (
        source
        .groupby(
            "Piezometer",
            sort=False
        )["MonthIndex"]
        .shift(1)
    )

    source["Prev_Head_m"] = (
        source
        .groupby(
            "Piezometer",
            sort=False
        )["Head_m"]
        .shift(1)
    )

    source["GW_Decline_m"] = (
        source["Prev_Head_m"]
        - source["Head_m"]
    )

    # Internal rule QC
    first_mask = (
        source.groupby(
            "Piezometer",
            sort=False
        ).cumcount()
        == 0
    )

    if not source.loc[
        first_mask,
        "Prev_MonthIndex"
    ].isna().all():

        raise RuntimeError(
            "Stage-01 boundary QC failed: "
            "first record must have no previous month."
        )

    delta_index = (
        source.loc[
            ~first_mask,
            "MonthIndex"
        ]
        -
        source.loc[
            ~first_mask,
            "Prev_MonthIndex"
        ]
    )

    if not (delta_index == 1).all():

        raise RuntimeError(
            "Stage-01 temporal continuity QC failed."
        )

    # ----------------------------------------------------------
    # Build a semantic pool.
    # These are calculations from source data, not values
    # copied from the frozen reference.
    # ----------------------------------------------------------

    pool = pd.DataFrame(
        index=source.index
    )

    pool["Piezometer"] = (
        source["Piezometer"]
    )

    pool["Well"] = (
        source["Piezometer"]
    )

    pool["Well_ID"] = (
        source["Piezometer"]
    )

    pool["Piezometer_ID"] = (
        source["Piezometer"]
    )

    pool["Year"] = (
        source["Year"].astype(int)
    )

    pool["JYear"] = (
        source["Year"].astype(int)
    )

    pool["Month"] = (
        source["Month"].astype(int)
    )

    pool["JMonth"] = (
        source["Month"].astype(int)
    )

    pool["MonthIndex"] = (
        source["MonthIndex"].astype(int)
    )

    pool["Month_Index"] = (
        source["MonthIndex"].astype(int)
    )

    pool["Prev_MonthIndex"] = (
        source["Prev_MonthIndex"]
    )

    pool["Head_m"] = (
        source["Head_m"]
    )

    pool["Prev_Head_m"] = (
        source["Prev_Head_m"]
    )

    pool["Head"] = (
        source["Head_m"]
    )

    pool["Groundwater_Head_m"] = (
        source["Head_m"]
    )

    pool["GW_Head_m"] = (
        source["Head_m"]
    )

    pool["GW_Decline_m"] = (
        source["GW_Decline_m"]
    )

    pool["Groundwater_Decline_m"] = (
        source["GW_Decline_m"]
    )

    pool["Decline_m"] = (
        source["GW_Decline_m"]
    )

    pool["DeltaH_m"] = (
        source["GW_Decline_m"]
    )

    pool["dH_m"] = (
        source["GW_Decline_m"]
    )

    # ----------------------------------------------------------
    # Exact frozen schema
    # ----------------------------------------------------------

    unsupported = [
        c
        for c in contract["columns"]
        if c not in pool.columns
    ]

    if unsupported:

        raise RuntimeError(
            "Stage 01 contains unsupported columns: "
            + ", ".join(unsupported)
        )

    regenerated = pool[
        contract["columns"]
    ].copy()

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != len(reference):

        raise RuntimeError(
            "Stage 01 row-count mismatch before regression: "
            f"generated={len(regenerated)}, "
            f"reference={len(reference)}"
        )

    well_col = contract["well"]
    year_col = contract["year"]
    month_col = contract["month"]

    duplicate_count = int(
        regenerated.duplicated(
            [
                well_col,
                year_col,
                month_col
            ]
        ).sum()
    )

    if duplicate_count != 0:

        raise RuntimeError(
            "Duplicate well-year-month rows "
            f"detected: {duplicate_count}"
        )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        Path(
            cfg["paths"][
                "regenerated_root"
            ]
        )
        / "stage01"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out = (
        out_dir
        / ref_path.name
    )

    regenerated.to_csv(
        out,
        index=False
    )

    return (
        regenerated,
        out,
        ref_path
    )


def main():

    cfg = load_config()

    result, out, ref = (
        reconstruct_stage01(cfg)
    )

    print("=" * 100)
    print("STAGE 01 — GROUNDWATER RECONSTRUCTION")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)
    print("Rows      :", len(result))
    print(
        "Wells     :",
        result.iloc[:, 0].nunique()
    )

    print()
    print("Columns:")
    for c in result.columns:
        print(" -", c)

    print()
    print("STAGE 01 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
