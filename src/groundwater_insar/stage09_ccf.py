from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config


CCF_FILENAME = "09_CCF_Lag0_12_ByWell.csv"
SUMMARY_FILENAME = "09_CCF_Lag_Summary.csv"
AR_FILENAME = "09_GW_AR1_Prewhitening_Coefficients.csv"
BEST_FILENAME = "09_BestLag_Stability.csv"

MAX_LAG = 12

VERSIONS = [
    "RAW",
    "SEASONAL_ADJUSTED",
    "PREWHITENED",
]


def _raw_series(
    g: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:

    return (
        g[
            ["MonthIndex", variable]
        ]
        .rename(
            columns={variable: "Value"}
        )
        .copy()
    )


def _seasonal_series(
    g: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:

    d = g[
        [
            "MonthIndex",
            "JMonth",
            variable,
        ]
    ].copy()

    seasonal_median = (
        d.groupby("JMonth")[variable]
        .transform("median")
    )

    d["Value"] = (
        d[variable]
        - seasonal_median
    )

    return d[
        ["MonthIndex", "Value"]
    ]


def _calendar_pairs(
    x: pd.DataFrame,
    y: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:

    a = (
        x.rename(
            columns={"Value": "GW"}
        )
        .copy()
    )

    b = (
        y.rename(
            columns={"Value": "InSAR"}
        )
        .copy()
    )

    # Positive lag means:
    #
    # GW(t) versus InSAR(t + lag)
    #
    # Therefore the InSAR time index is shifted
    # backward by lag before calendar matching.
    b["MonthIndex"] = (
        b["MonthIndex"] - lag
    )

    return (
        a.merge(
            b,
            on="MonthIndex",
            how="inner",
            validate="one_to_one",
        )
        .dropna(
            subset=["GW", "InSAR"]
        )
        .sort_values("MonthIndex")
        .reset_index(drop=True)
    )


def _estimate_gw_ar1(
    seasonal_gw: pd.DataFrame,
):

    current = (
        seasonal_gw
        .rename(
            columns={"Value": "Current"}
        )
        .copy()
    )

    previous = (
        seasonal_gw
        .rename(
            columns={"Value": "Previous"}
        )
        .copy()
    )

    # Exact one-calendar-month pairing.
    previous["MonthIndex"] = (
        previous["MonthIndex"] + 1
    )

    q = (
        current.merge(
            previous,
            on="MonthIndex",
            how="inner",
            validate="one_to_one",
        )
        .dropna(
            subset=[
                "Current",
                "Previous",
            ]
        )
    )

    x = q["Previous"].to_numpy(
        dtype=float
    )

    y = q["Current"].to_numpy(
        dtype=float
    )

    if len(q) == 0:
        raise RuntimeError(
            "No valid AR(1) pairs."
        )

    denominator = np.sum(
        x * x
    )

    if denominator == 0:
        raise RuntimeError(
            "Zero AR(1) denominator."
        )

    # Historical Stage-09 rule:
    # AR(1) regression through the origin.
    phi = (
        np.sum(x * y)
        /
        denominator
    )

    return len(q), float(phi)


def _prewhiten(
    d: pd.DataFrame,
    phi: float,
) -> pd.DataFrame:

    current = (
        d.rename(
            columns={"Value": "Current"}
        )
        .copy()
    )

    previous = (
        d.rename(
            columns={"Value": "Previous"}
        )
        .copy()
    )

    previous["MonthIndex"] = (
        previous["MonthIndex"] + 1
    )

    q = (
        current.merge(
            previous,
            on="MonthIndex",
            how="inner",
            validate="one_to_one",
        )
        .dropna(
            subset=[
                "Current",
                "Previous",
            ]
        )
        .sort_values("MonthIndex")
    )

    q["Value"] = (
        q["Current"]
        - phi * q["Previous"]
    )

    return q[
        ["MonthIndex", "Value"]
    ]


def _correlations(
    q: pd.DataFrame,
):

    if len(q) < 3:
        return (
            len(q),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    pearson = stats.pearsonr(
        q["GW"],
        q["InSAR"],
    )

    spearman = stats.spearmanr(
        q["GW"],
        q["InSAR"],
    )

    return (
        len(q),
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )


def reconstruct_stage09(
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

    if not stage07_path.exists():
        raise FileNotFoundError(
            stage07_path
        )

    s7 = pd.read_csv(
        stage07_path
    )

    expected_wells = int(
        cfg["analysis"][
            "expected_lag_eligible_wells"
        ]
    )

    observed_wells = (
        s7["Piezometer"].nunique()
    )

    if observed_wells != expected_wells:
        raise RuntimeError(
            f"Expected {expected_wells} wells; "
            f"found {observed_wells}."
        )

    # ========================================================
    # AR(1) + CCF
    # ========================================================

    ar_rows = []
    ccf_rows = []

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

        gw_raw = _raw_series(
            g,
            "GW_Decline_m",
        )

        insar_raw = _raw_series(
            g,
            "InSAR_Increment_mm",
        )

        gw_seasonal = _seasonal_series(
            g,
            "GW_Decline_m",
        )

        insar_seasonal = _seasonal_series(
            g,
            "InSAR_Increment_mm",
        )

        (
            n_ar1,
            phi,
        ) = _estimate_gw_ar1(
            gw_seasonal
        )

        ar_rows.append({
            "Piezometer":
                well,

            "N_AR1_pairs":
                int(n_ar1),

            "GW_AR1_phi":
                float(phi),
        })

        gw_prewhitened = _prewhiten(
            gw_seasonal,
            phi,
        )

        # Historical rule:
        # the groundwater AR(1) coefficient is
        # applied to BOTH seasonally adjusted series.
        insar_prewhitened = _prewhiten(
            insar_seasonal,
            phi,
        )

        series = {
            "RAW": (
                gw_raw,
                insar_raw,
            ),

            "SEASONAL_ADJUSTED": (
                gw_seasonal,
                insar_seasonal,
            ),

            "PREWHITENED": (
                gw_prewhitened,
                insar_prewhitened,
            ),
        }

        for version in VERSIONS:

            gw_series, insar_series = (
                series[version]
            )

            for lag in range(
                MAX_LAG + 1
            ):

                q = _calendar_pairs(
                    gw_series,
                    insar_series,
                    lag,
                )

                (
                    n,
                    pearson,
                    pearson_p,
                    spearman,
                    spearman_p,
                ) = _correlations(q)

                ccf_rows.append({
                    "Piezometer":
                        well,

                    "Version":
                        version,

                    "Lag_months":
                        lag,

                    "N":
                        int(n),

                    "Pearson":
                        pearson,

                    "Pearson_p_naive":
                        pearson_p,

                    "Spearman":
                        spearman,

                    "Spearman_p_naive":
                        spearman_p,
                })

    ar = pd.DataFrame(
        ar_rows
    )

    ccf = pd.DataFrame(
        ccf_rows
    )

    # ========================================================
    # CCF lag summary
    # ========================================================

    summary_rows = []

    for (
        version,
        lag,
    ), g in ccf.groupby(
        [
            "Version",
            "Lag_months",
        ],
        sort=True,
    ):

        summary_rows.append({
            "Version":
                version,

            "Lag_months":
                int(lag),

            "Wells":
                int(
                    g["Piezometer"]
                    .nunique()
                ),

            "Median_Pearson":
                float(
                    g["Pearson"]
                    .median()
                ),

            "Mean_Pearson":
                float(
                    g["Pearson"]
                    .mean()
                ),

            "Median_Spearman":
                float(
                    g["Spearman"]
                    .median()
                ),

            "Mean_Spearman":
                float(
                    g["Spearman"]
                    .mean()
                ),

            "Positive_Pearson_Wells":
                int(
                    (
                        g["Pearson"] > 0
                    ).sum()
                ),

            "Pearson_ge_03_Wells":
                int(
                    (
                        g["Pearson"] >= 0.30
                    ).sum()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Best lag stability
    #
    # Best lag = maximum SIGNED Pearson coefficient.
    # Ties are resolved toward the smaller lag.
    # ========================================================

    best_maps = {}

    for version in [
        "PREWHITENED",
        "RAW",
        "SEASONAL_ADJUSTED",
    ]:

        d = (
            ccf[
                ccf["Version"]
                == version
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
        )

        best_maps[version] = dict(
            zip(
                d["Piezometer"],
                d["Lag_months"],
            )
        )

    wells = sorted(
        s7["Piezometer"].unique()
    )

    best = pd.DataFrame({
        "Piezometer":
            wells,
    })

    for version in [
        "PREWHITENED",
        "RAW",
        "SEASONAL_ADJUSTED",
    ]:

        best[version] = (
            best["Piezometer"]
            .map(
                best_maps[version]
            )
            .astype(int)
        )

    best["All_3_same"] = (
        (
            best["PREWHITENED"]
            == best["RAW"]
        )
        &
        (
            best["RAW"]
            == best[
                "SEASONAL_ADJUSTED"
            ]
        )
    )

    best["Adjusted_2_same"] = (
        best["PREWHITENED"]
        ==
        best["SEASONAL_ADJUSTED"]
    )

    # ========================================================
    # Structural QC
    # ========================================================

    expected_ccf_rows = (
        expected_wells
        * len(VERSIONS)
        * (MAX_LAG + 1)
    )

    if len(ccf) != expected_ccf_rows:
        raise RuntimeError(
            "Stage-09 CCF row-count QC failed: "
            f"{len(ccf)} != "
            f"{expected_ccf_rows}"
        )

    if len(summary) != (
        len(VERSIONS)
        * (MAX_LAG + 1)
    ):
        raise RuntimeError(
            "Stage-09 summary row-count QC failed."
        )

    if len(ar) != expected_wells:
        raise RuntimeError(
            "Stage-09 AR row-count QC failed."
        )

    if len(best) != expected_wells:
        raise RuntimeError(
            "Stage-09 best-lag row-count QC failed."
        )

    if ccf[
        [
            "Piezometer",
            "Version",
            "Lag_months",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Stage-09 CCF keys."
        )

    if set(
        ccf["Lag_months"].unique()
    ) != set(
        range(MAX_LAG + 1)
    ):
        raise RuntimeError(
            "Stage-09 lag-range QC failed."
        )

    # ========================================================
    # Output
    # ========================================================

    out_dir = (
        regenerated_root
        / "stage09"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ccf_out = (
        out_dir
        / CCF_FILENAME
    )

    summary_out = (
        out_dir
        / SUMMARY_FILENAME
    )

    ar_out = (
        out_dir
        / AR_FILENAME
    )

    best_out = (
        out_dir
        / BEST_FILENAME
    )

    ccf.to_csv(
        ccf_out,
        index=False,
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    ar.to_csv(
        ar_out,
        index=False,
    )

    best.to_csv(
        best_out,
        index=False,
    )

    return (
        ccf,
        summary,
        ar,
        best,
        ccf_out,
        summary_out,
        ar_out,
        best_out,
    )


def main():

    cfg = load_config()

    (
        ccf,
        summary,
        ar,
        best,
        ccf_out,
        summary_out,
        ar_out,
        best_out,
    ) = reconstruct_stage09(cfg)

    print("=" * 100)
    print(
        "STAGE 09 — CCF 0–12 MONTHS"
    )
    print("=" * 100)

    print("CCF rows      :", len(ccf))
    print("Summary rows  :", len(summary))
    print("AR rows       :", len(ar))
    print("Best-lag rows :", len(best))

    print("\nVersions:")
    print(
        ccf["Version"]
        .value_counts()
        .to_string()
    )

    print("\nBest-lag stability:")
    print(
        best[
            [
                "All_3_same",
                "Adjusted_2_same",
            ]
        ]
        .sum()
        .to_string()
    )

    print("\nOutputs:")
    print(ccf_out)
    print(summary_out)
    print(ar_out)
    print(best_out)

    print()
    print(
        "STAGE 09 RECONSTRUCTION: PASS"
    )


if __name__ == "__main__":
    main()
