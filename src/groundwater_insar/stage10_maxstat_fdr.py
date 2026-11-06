from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .stage09_ccf import (
    _seasonal_series,
    _prewhiten,
)


OUTPUT_FILENAME = (
    "10_Lag_Significance_MaxStat_FDR.csv"
)

MAX_LAG = 12


def _calendar_max_pearson(
    gw: pd.DataFrame,
    insar: pd.DataFrame,
):

    correlations = []

    for lag in range(MAX_LAG + 1):

        a = (
            gw.rename(
                columns={"Value": "GW"}
            )
            .copy()
        )

        b = (
            insar.rename(
                columns={"Value": "InSAR"}
            )
            .copy()
        )

        # GW(t) versus InSAR(t + lag)
        b["MonthIndex"] -= lag

        q = (
            a.merge(
                b,
                on="MonthIndex",
                how="inner",
                validate="one_to_one",
            )
            .dropna(
                subset=["GW", "InSAR"]
            )
        )

        if len(q) < 3:
            continue

        result = stats.pearsonr(
            q["GW"],
            q["InSAR"],
        )

        correlations.append(
            float(result.statistic)
        )

    if not correlations:
        raise RuntimeError(
            "No valid CCF correlations."
        )

    return max(correlations)


def _bh_fdr(p_values):

    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    order = np.argsort(p)

    ranked = p[order]

    q = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0,
    )

    result = np.empty(
        n,
        dtype=float,
    )

    result[order] = q

    return result


def reconstruct_stage10(
    cfg: dict | None = None,
):

    if cfg is None:
        cfg = load_config()

    regenerated_root = Path(
        cfg["paths"]["regenerated_root"]
    )

    stage07_path = (
        regenerated_root
        / "stage07"
        / "07_Canonical_Monthly_Pairs_37Wells.csv"
    )

    stage09_ccf_path = (
        regenerated_root
        / "stage09"
        / "09_CCF_Lag0_12_ByWell.csv"
    )

    stage09_ar_path = (
        regenerated_root
        / "stage09"
        / "09_GW_AR1_Prewhitening_Coefficients.csv"
    )

    for path in [
        stage07_path,
        stage09_ccf_path,
        stage09_ar_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    s7 = pd.read_csv(
        stage07_path
    )

    ccf = pd.read_csv(
        stage09_ccf_path
    )

    ar = pd.read_csv(
        stage09_ar_path
    )

    expected_wells = int(
        cfg["analysis"][
            "expected_lag_eligible_wells"
        ]
    )

    start_year = int(
        cfg["analysis"][
            "jalali_start_year"
        ]
    )

    end_year = int(
        cfg["analysis"][
            "jalali_end_year"
        ]
    )

    # --------------------------------------------------------
    # Official complete monthly grid
    # --------------------------------------------------------

    first_month = (
        start_year * 12
    )

    last_month = (
        end_year * 12 + 11
    )

    full_months = np.arange(
        first_month,
        last_month + 1,
        dtype=int,
    )

    expected_months = (
        (end_year - start_year + 1)
        * 12
    )

    if len(full_months) != expected_months:
        raise RuntimeError(
            "Official monthly-grid QC failed."
        )

    # Exclude circular shifts that overlap the tested
    # +/-12-month neighborhood.
    shifts = list(
        range(
            MAX_LAG + 1,
            len(full_months) - MAX_LAG,
        )
    )

    if len(shifts) != 83:
        raise RuntimeError(
            f"Expected 83 circular shifts; "
            f"found {len(shifts)}."
        )

    # --------------------------------------------------------
    # Observed best statistics from Stage 09 PREWHITENED CCF
    # --------------------------------------------------------

    observed = (
        ccf[
            ccf["Version"]
            == "PREWHITENED"
        ]
        .sort_values(
            [
                "Piezometer",
                "Pearson",
                "Lag_months",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "Piezometer"
        )
        .set_index(
            "Piezometer"
        )
    )

    phi_map = dict(
        zip(
            ar["Piezometer"],
            ar["GW_AR1_phi"],
        )
    )

    rows = []

    # --------------------------------------------------------
    # Well-by-well circular-shift max-stat null
    # --------------------------------------------------------

    for well, g in s7.groupby(
        "Piezometer",
        sort=True,
    ):

        g = (
            g.sort_values(
                "MonthIndex"
            )
            .copy()
        )

        if well not in observed.index:
            raise RuntimeError(
                f"Missing Stage-09 observed statistic "
                f"for {well}."
            )

        if well not in phi_map:
            raise RuntimeError(
                f"Missing AR1 coefficient for {well}."
            )

        phi = float(
            phi_map[well]
        )

        gw_seasonal = _seasonal_series(
            g,
            "GW_Decline_m",
        )

        insar_seasonal = _seasonal_series(
            g,
            "InSAR_Increment_mm",
        )

        gw_pre = _prewhiten(
            gw_seasonal,
            phi,
        )

        insar_pre = _prewhiten(
            insar_seasonal,
            phi,
        )

        # Put prewhitened InSAR onto the complete
        # official 108-month grid before circular shifting.
        insar_grid = (
            pd.DataFrame({
                "MonthIndex":
                    full_months
            })
            .merge(
                insar_pre,
                on="MonthIndex",
                how="left",
                validate="one_to_one",
            )
        )

        null_max = []

        for shift in shifts:

            shifted = (
                insar_grid.copy()
            )

            shifted["Value"] = np.roll(
                shifted[
                    "Value"
                ].to_numpy(float),
                shift,
            )

            null_max.append(
                _calendar_max_pearson(
                    gw_pre,
                    shifted,
                )
            )

        null_max = np.asarray(
            null_max,
            dtype=float,
        )

        obs = observed.loc[well]

        observed_r = float(
            obs["Pearson"]
        )

        maxstat_p = float(
            (
                1
                + np.sum(
                    null_max
                    >= observed_r
                )
            )
            /
            (
                len(null_max) + 1
            )
        )

        rows.append({
            "Piezometer":
                well,

            "Observed_Best_Lag_months":
                int(
                    obs[
                        "Lag_months"
                    ]
                ),

            "Observed_N":
                int(
                    obs["N"]
                ),

            "Observed_Max_Pearson":
                observed_r,

            "N_Circular_Shifts":
                int(
                    len(null_max)
                ),

            "Null_MaxR_95pct":
                float(
                    np.quantile(
                        null_max,
                        0.95,
                    )
                ),

            "Null_MaxR_99pct":
                float(
                    np.quantile(
                        null_max,
                        0.99,
                    )
                ),

            "MaxStat_p":
                maxstat_p,
        })

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Benjamini-Hochberg FDR across 37 wells
    # --------------------------------------------------------

    result["FDR_q"] = _bh_fdr(
        result["MaxStat_p"]
        .to_numpy(float)
    )

    result["Significant_p05"] = (
        result["MaxStat_p"]
        < 0.05
    )

    result["Significant_FDR05"] = (
        result["FDR_q"]
        < 0.05
    )

    # Historical output ordering:
    # increasing max-stat p-value.
    result = (
        result
        .sort_values(
            [
                "MaxStat_p",
                "Piezometer",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Structural QC
    # --------------------------------------------------------

    if len(result) != expected_wells:
        raise RuntimeError(
            "Stage-10 well-count QC failed."
        )

    if not (
        result["N_Circular_Shifts"]
        == 83
    ).all():
        raise RuntimeError(
            "Stage-10 shift-count QC failed."
        )

    if result[
        "Piezometer"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Stage-10 piezometers."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    out_dir = (
        regenerated_root
        / "stage10"
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
        reconstruct_stage10(cfg)
    )

    print("=" * 100)
    print(
        "STAGE 10 — MAX-STAT / FDR"
    )
    print("=" * 100)

    print(
        "Wells                 :",
        len(result)
    )

    print(
        "Circular shifts       :",
        result[
            "N_Circular_Shifts"
        ].unique().tolist()
    )

    print(
        "Significant p<0.05    :",
        int(
            result[
                "Significant_p05"
            ].sum()
        )
    )

    print(
        "Significant FDR<0.05  :",
        int(
            result[
                "Significant_FDR05"
            ].sum()
        )
    )

    print("\nOutput:")
    print(out)

    print()
    print(
        "STAGE 10 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
