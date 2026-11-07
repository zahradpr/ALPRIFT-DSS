from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .config import load_config


WINDOWS = [
    ("DD_0_6", 0, 6),
    ("DD_7_12", 7, 12),
    ("DD_13_24", 13, 24),
    ("DD_25_36", 25, 36),
]


def _standardized_regression(df, ycol, xcols):
    q = df[[ycol] + xcols].dropna().copy()

    if len(q) < 3:
        return len(q), np.nan, [np.nan] * len(xcols)

    y = q[ycol].to_numpy(float)
    X = q[xcols].to_numpy(float)

    ysd = y.std(ddof=0)
    xsd = X.std(axis=0, ddof=0)

    if ysd == 0 or np.any(xsd == 0):
        return len(q), np.nan, [np.nan] * len(xcols)

    yz = (y - y.mean()) / ysd
    Xz = (X - X.mean(axis=0)) / xsd

    design = np.column_stack([np.ones(len(Xz)), Xz])
    coef, *_ = np.linalg.lstsq(design, yz, rcond=None)

    pred = design @ coef

    ss_res = float(np.sum((yz - pred) ** 2))
    ss_tot = float(np.sum((yz - yz.mean()) ** 2))

    r2 = 1.0 - ss_res / ss_tot

    return len(q), r2, coef[1:].tolist()


def reconstruct_stage16(cfg=None):
    if cfg is None:
        cfg = load_config()

    regen = Path(cfg["paths"]["regenerated_root"])

    stage14 = pd.read_csv(
        regen / "stage14" / "14_Monthly_HydroMechanical_States.csv"
    )

    panel = (
        stage14[stage14["Primary_Valid"]]
        [
            [
                "Piezometer",
                "JYear",
                "JMonth",
                "MonthIndex",
                "Subsidence_Increment_mm",
            ]
        ]
        .copy()
        .sort_values(["Piezometer", "MonthIndex"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Full historical groundwater source.
    # This preserves pre-1395 groundwater history needed by
    # the 25–36 month windows.
    # ---------------------------------------------------------

    raw = pd.read_csv(cfg["paths"]["piezometer_monthly"])

    if "Year" not in raw.columns or "Month" not in raw.columns:
        raise RuntimeError("Expected Year and Month in raw piezometer file.")

    raw["MonthIndex"] = (
        raw["Year"].astype(int) * 12
        + raw["Month"].astype(int)
        - 1
    )

    wells = sorted(panel["Piezometer"].unique())

    histories = {}

    for well in wells:
        g = raw[
            ["Year", "Month", "MonthIndex", well]
        ].copy()

        g = g.rename(columns={well: "Head_m"})
        g = g.sort_values("MonthIndex")

        # Historical distributed-lag windows use groundwater
        # head differences between the two ENDPOINTS of each
        # lag interval. Missing values inside the interval do
        # not invalidate the window if both endpoints exist.
        histories[well] = (
            g.set_index("MonthIndex")["Head_m"]
            .to_dict()
        )

    # ---------------------------------------------------------
    # Fixed lag-window groundwater decline.
    # Windows are sums of exact monthly groundwater decline
    # values over past lags.
    # ---------------------------------------------------------

    for name, lo, hi in WINDOWS:
        vals = []

        for r in panel.itertuples(index=False):
            h = histories[r.Piezometer]
            t = int(r.MonthIndex)

            # Sum of monthly declines over lag lo..hi
            # telescopes to the head difference between
            # the two interval endpoints:
            #
            # H(t-hi-1) - H(t-lo)
            #
            # Historical workflow used this endpoint form.
            old_head = h.get(
                t - hi - 1,
                np.nan,
            )

            recent_head = h.get(
                t - lo,
                np.nan,
            )

            if (
                np.isfinite(old_head)
                and np.isfinite(recent_head)
            ):
                vals.append(
                    float(
                        old_head
                        - recent_head
                    )
                )
            else:
                vals.append(np.nan)

        panel[f"{name}_m"] = vals

    # ---------------------------------------------------------
    # Seasonal anomalies
    # ---------------------------------------------------------

    panel["Subs_SeasonalMedian"] = (
        panel.groupby(["Piezometer", "JMonth"])
        ["Subsidence_Increment_mm"]
        .transform("median")
    )

    panel["Subs_Anomaly"] = (
        panel["Subsidence_Increment_mm"]
        - panel["Subs_SeasonalMedian"]
    )

    for name, _, _ in WINDOWS:
        col = f"{name}_m"

        med = (
            panel.groupby(["Piezometer", "JMonth"])[col]
            .transform("median")
        )

        panel[f"{name}_m_ANOM"] = (
            panel[col] - med
        )

    panel_cols = [
        "Piezometer",
        "JYear",
        "JMonth",
        "MonthIndex",
        "Subsidence_Increment_mm",
        "DD_0_6_m",
        "DD_7_12_m",
        "DD_13_24_m",
        "DD_25_36_m",
        "Subs_SeasonalMedian",
        "Subs_Anomaly",
        "DD_0_6_m_ANOM",
        "DD_7_12_m_ANOM",
        "DD_13_24_m_ANOM",
        "DD_25_36_m_ANOM",
    ]

    panel = panel[panel_cols]

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    corr_rows = []

    for well, g in panel.groupby("Piezometer", sort=True):
        for name, _, _ in WINDOWS:
            xcol = f"{name}_m_ANOM"

            q = g[
                ["Subs_Anomaly", xcol]
            ].dropna()

            if len(q) >= 3:
                pr = pearsonr(
                    q[xcol],
                    q["Subs_Anomaly"],
                )
                p = float(pr.statistic)
            else:
                p = np.nan

            corr_rows.append({
                "Piezometer": well,
                "Window": name,
                "N": len(q),
                "Pearson": p,
            })

    corr = pd.DataFrame(corr_rows)

    # ---------------------------------------------------------
    # Per-well distributed-lag standardized regression
    # ---------------------------------------------------------

    xcols = [
        "DD_0_6_m_ANOM",
        "DD_7_12_m_ANOM",
        "DD_13_24_m_ANOM",
        "DD_25_36_m_ANOM",
    ]

    beta_rows = []

    for well, g in panel.groupby("Piezometer", sort=True):
        n, r2, beta = _standardized_regression(
            g,
            "Subs_Anomaly",
            xcols,
        )

        beta_rows.append({
            "Piezometer": well,
            "N": n,
            "R2": r2,
            "Beta_0_6": beta[0],
            "Beta_7_12": beta[1],
            "Beta_13_24": beta[2],
            "Beta_25_36": beta[3],
        })

    betas = pd.DataFrame(beta_rows)

    # ---------------------------------------------------------
    # Pooled standardized regression
    # ---------------------------------------------------------

    pooled_n, pooled_r2, pooled_beta = (
        _standardized_regression(
            panel,
            "Subs_Anomaly",
            xcols,
        )
    )

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------

    outdir = regen / "stage16"
    outdir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(
        outdir / "16_Binned_DistributedLag_Panel.csv",
        index=False,
    )

    corr.to_csv(
        outdir / "16_Binned_DistributedLag_Correlations_ByWell.csv",
        index=False,
    )

    betas.to_csv(
        outdir / "16_Binned_DistributedLag_Betas_ByWell.csv",
        index=False,
    )

    # Historical multi-section summary format.
    corr_summary = (
        corr.groupby("Window", sort=True)
        .agg(
            Wells=("Piezometer", "nunique"),
            Median_N=("N", "median"),
            Median_Pearson=("Pearson", "median"),
            Mean_Pearson=("Pearson", "mean"),
            Positive_Wells=("Pearson", lambda x: int((x > 0).sum())),
            Pearson_ge_03_Wells=("Pearson", lambda x: int((x.abs() >= 0.3).sum())),
        )
        .reset_index()
    )

    beta_summary_rows = []

    beta_map = [
        ("0_6", "Beta_0_6", pooled_beta[0]),
        ("7_12", "Beta_7_12", pooled_beta[1]),
        ("13_24", "Beta_13_24", pooled_beta[2]),
        ("25_36", "Beta_25_36", pooled_beta[3]),
    ]

    for label, col, pooled in beta_map:
        beta_summary_rows.append({
            "Window": label,
            "Median_Standardized_Beta": float(betas[col].median()),
            "Positive_Beta_Wells": int((betas[col] > 0).sum()),
            "Pooled_Standardized_Beta": float(pooled),
        })

    beta_summary = pd.DataFrame(beta_summary_rows)

    summary_path = (
        outdir / "16_Binned_DistributedLag_Summary.csv"
    )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        f.write("FIXED-WINDOW CORRELATION SUMMARY\n")
        corr_summary.to_csv(f, index=False)

        f.write("\nSTANDARDIZED DISTRIBUTED-LAG BETA SUMMARY\n")
        beta_summary.to_csv(f, index=False)

        f.write(f"\nPOOLED_R2,{pooled_r2}\n")

    print("=" * 105)
    print("STAGE 16 — BINNED DISTRIBUTED LAG")
    print("=" * 105)
    print("Panel rows       :", len(panel))
    print("Wells            :", panel["Piezometer"].nunique())
    print("Correlation rows :", len(corr))
    print("Beta rows        :", len(betas))
    print("Pooled N         :", pooled_n)
    print("Pooled R2        :", pooled_r2)

    print("\nPer-well regression N distribution:")
    print(
        betas["N"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing predictor counts:")
    for name, _, _ in WINDOWS:
        c = f"{name}_m"
        print(c, ":", int(panel[c].isna().sum()))

    print("\nSTAGE 16 RECONSTRUCTION: PASS")


def main():
    reconstruct_stage16()


if __name__ == "__main__":
    main()
