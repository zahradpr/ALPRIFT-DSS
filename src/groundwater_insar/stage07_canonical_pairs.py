from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config
from .stage02_insar_epochs import frozen_file_for_stage


def reconstruct_stage07(cfg: dict | None = None):

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
        7
    )

    s4 = pd.read_csv(stage04_path)
    s5 = pd.read_csv(stage05_path)
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
        "Valid_InSAR_Increment",
        "SameYearStack",
        "StackDistance",
        "SelectionRule",
    ]

    if list(reference.columns) != expected_columns:
        raise RuntimeError(
            "Unexpected Stage-07 frozen schema:\n"
            f"{list(reference.columns)}"
        )

    # ----------------------------------------------------------
    # Eligible piezometers
    # ----------------------------------------------------------

    eligible = (
        s5.loc[
            s5["Valid_change_pairs"] > 0,
            "Piezometer"
        ]
        .tolist()
    )

    expected_eligible = int(
        cfg["analysis"]["expected_lag_eligible_wells"]
    )

    if len(eligible) != expected_eligible:
        raise RuntimeError(
            f"Expected {expected_eligible} eligible wells; "
            f"found {len(eligible)}."
        )

    start_year = int(
        cfg["analysis"]["jalali_start_year"]
    )

    end_year = int(
        cfg["analysis"]["jalali_end_year"]
    )

    # Stage 07 is the point where records outside the official
    # Jalali analysis period are removed.
    x = s4[
        s4["Piezometer"].isin(eligible)
        & s4["JYear"].between(
            start_year,
            end_year
        )
    ].copy()

    # ----------------------------------------------------------
    # Derived selection fields
    # ----------------------------------------------------------

    x["Valid_InSAR_Increment"] = (
        x["InSAR_Increment_mm"].notna()
    )

    x["SameYearStack"] = (
        x["StackYear"] == x["JYear"]
    )

    x["StackDistance"] = (
        x["StackYear"] - x["JYear"]
    ).abs().astype(int)

    keys = [
        "Piezometer",
        "MonthIndex",
    ]

    # ----------------------------------------------------------
    # Canonical stack selection
    #
    # Priority:
    #
    # 1. valid InSAR increment from same-year stack
    # 2. nearest valid adjacent stack
    # 3. if no valid increment exists:
    #       same-year stack when available
    # 4. otherwise nearest available stack
    #
    # In the current 1395–1403 dataset, no canonical record
    # requires the adjacent-valid fallback; however the rule
    # is retained explicitly for reproducibility.
    # ----------------------------------------------------------

    selected_rows = []

    for (well, month_index), group in x.groupby(
        keys,
        sort=False
    ):

        g = group.copy()

        same_valid = g[
            g["SameYearStack"]
            & g["Valid_InSAR_Increment"]
        ]

        if len(same_valid):

            row = (
                same_valid
                .sort_values("StackYear")
                .iloc[0]
                .copy()
            )

            rule = "same_year_stack"

        else:

            valid = g[
                g["Valid_InSAR_Increment"]
            ]

            if len(valid):

                row = (
                    valid
                    .sort_values(
                        [
                            "StackDistance",
                            "StackYear",
                        ]
                    )
                    .iloc[0]
                    .copy()
                )

                rule = "adjacent_valid_stack"

            else:

                same = g[
                    g["SameYearStack"]
                ]

                if len(same):

                    row = (
                        same
                        .sort_values("StackYear")
                        .iloc[0]
                        .copy()
                    )

                else:

                    row = (
                        g
                        .sort_values(
                            [
                                "StackDistance",
                                "StackYear",
                            ]
                        )
                        .iloc[0]
                        .copy()
                    )

                rule = "no_valid_insar_increment"

        row["Piezometer"] = well
        row["MonthIndex"] = month_index
        row["SelectionRule"] = rule

        selected_rows.append(row)

    regenerated = pd.DataFrame(
        selected_rows
    )

    regenerated = (
        regenerated[
            expected_columns
        ]
        .sort_values(
            [
                "Piezometer",
                "MonthIndex",
            ]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != len(reference):
        raise RuntimeError(
            "Stage-07 row-count mismatch: "
            f"generated={len(regenerated)}, "
            f"reference={len(reference)}"
        )

    if regenerated[
        ["Piezometer", "MonthIndex"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical well-month records."
        )

    if regenerated[
        "Piezometer"
    ].nunique() != expected_eligible:
        raise RuntimeError(
            "Stage-07 eligible-well QC failed."
        )

    if not regenerated[
        "JYear"
    ].between(
        start_year,
        end_year
    ).all():
        raise RuntimeError(
            "Stage-07 official-period QC failed."
        )

    # Derived field QC
    if not (
        regenerated["Valid_InSAR_Increment"]
        ==
        regenerated["InSAR_Increment_mm"].notna()
    ).all():
        raise RuntimeError(
            "Valid_InSAR_Increment QC failed."
        )

    if not (
        regenerated["SameYearStack"]
        ==
        (
            regenerated["StackYear"]
            == regenerated["JYear"]
        )
    ).all():
        raise RuntimeError(
            "SameYearStack QC failed."
        )

    expected_distance = (
        regenerated["StackYear"]
        - regenerated["JYear"]
    ).abs().astype(int)

    if not (
        regenerated["StackDistance"]
        == expected_distance
    ).all():
        raise RuntimeError(
            "StackDistance QC failed."
        )

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage07"
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

    result, out, ref = reconstruct_stage07(
        cfg
    )

    print("=" * 100)
    print("STAGE 07 — CANONICAL MONTHLY PAIRS")
    print("=" * 100)

    print("Reference :", ref)
    print("Generated :", out)
    print("Rows      :", len(result))

    print(
        "Wells     :",
        result["Piezometer"].nunique()
    )

    print(
        "Valid change pairs:",
        int(result["Valid_Change_Pair"].sum())
    )

    print("\nSelection rules:")
    print(
        result["SelectionRule"]
        .value_counts()
        .to_string()
    )

    print()
    print("STAGE 07 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
