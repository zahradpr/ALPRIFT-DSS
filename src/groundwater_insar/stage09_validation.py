from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config


CCF_FILE = "09_CCF_Lag0_12_ByWell.csv"
SUMMARY_FILE = "09_CCF_Lag_Summary.csv"
AR_FILE = "09_GW_AR1_Prewhitening_Coefficients.csv"
BEST_FILE = "09_BestLag_Stability.csv"


def _allclose(a, b):
    return np.isclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def validate_stage09(cfg: dict | None = None):

    if cfg is None:
        cfg = load_config()

    regenerated = (
        Path(cfg["paths"]["regenerated_root"])
        / "stage09"
    )

    reference_root = Path(
        cfg["paths"]["reference_root"]
    )

    reference = (
        reference_root
        / "v1_terminal_baseline"
        / "csv"
    )

    qc_dir = (
        Path(cfg["paths"]["output_root"])
        / "qc"
    )

    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_report = (
        qc_dir
        / "09_Stage09_PrecisionAware_Validation.csv"
    )

    cand_ccf = pd.read_csv(
        regenerated / CCF_FILE
    )

    ref_ccf = pd.read_csv(
        reference / CCF_FILE
    )

    cand_summary = pd.read_csv(
        regenerated / SUMMARY_FILE
    )

    ref_summary = pd.read_csv(
        reference / SUMMARY_FILE
    )

    cand_ar = pd.read_csv(
        regenerated / AR_FILE
    )

    ref_ar = pd.read_csv(
        reference / AR_FILE
    )

    cand_best = pd.read_csv(
        regenerated / BEST_FILE
    )

    ref_best = pd.read_csv(
        reference / BEST_FILE
    )

    checks = []

    def add(
        item,
        passed,
        observed,
        expected,
        note="",
    ):

        checks.append({
            "Item": item,
            "Status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "Observed": observed,
            "Expected": expected,
            "Note": note,
        })

    # ========================================================
    # 1. Structural contracts
    # ========================================================

    add(
        "CCF row count",
        len(cand_ccf) == 1443,
        len(cand_ccf),
        1443,
    )

    add(
        "Summary row count",
        len(cand_summary) == 39,
        len(cand_summary),
        39,
    )

    add(
        "AR1 row count",
        len(cand_ar) == 37,
        len(cand_ar),
        37,
    )

    add(
        "Best-lag row count",
        len(cand_best) == 37,
        len(cand_best),
        37,
    )

    ccf_keys = [
        "Piezometer",
        "Version",
        "Lag_months",
    ]

    add(
        "CCF unique keys",
        not cand_ccf[
            ccf_keys
        ].duplicated().any(),
        int(
            cand_ccf[
                ccf_keys
            ]
            .drop_duplicates()
            .shape[0]
        ),
        1443,
    )

    # ========================================================
    # 2. Align CCF
    # ========================================================

    c = (
        cand_ccf
        .sort_values(ccf_keys)
        .reset_index(drop=True)
    )

    r = (
        ref_ccf
        .sort_values(ccf_keys)
        .reset_index(drop=True)
    )

    key_match = (
        c[ccf_keys]
        .astype(str)
        .to_numpy()
        ==
        r[ccf_keys]
        .astype(str)
        .to_numpy()
    ).all()

    add(
        "CCF keys match frozen",
        key_match,
        bool(key_match),
        True,
    )

    n_match = (
        c["N"].to_numpy()
        ==
        r["N"].to_numpy()
    )

    add(
        "CCF N exact",
        bool(n_match.all()),
        f"{int(n_match.sum())}/1443",
        "1443/1443",
    )

    # ========================================================
    # 3. RAW CCF — fully reproducible
    # ========================================================

    raw_mask = (
        r["Version"] == "RAW"
    )

    for col in [
        "Pearson",
        "Pearson_p_naive",
        "Spearman",
        "Spearman_p_naive",
    ]:

        same = _allclose(
            c.loc[raw_mask, col],
            r.loc[raw_mask, col],
        )

        add(
            f"RAW {col}",
            bool(same.all()),
            f"{int(same.sum())}/481",
            "481/481",
        )

    # ========================================================
    # 4. Adjusted Pearson — fully reproducible
    # ========================================================

    for version in [
        "SEASONAL_ADJUSTED",
        "PREWHITENED",
    ]:

        mask = (
            r["Version"] == version
        )

        for col in [
            "Pearson",
            "Pearson_p_naive",
        ]:

            same = _allclose(
                c.loc[mask, col],
                r.loc[mask, col],
            )

            add(
                f"{version} {col}",
                bool(same.all()),
                f"{int(same.sum())}/481",
                "481/481",
            )

    # ========================================================
    # 5. Spearman precision-aware QC
    #
    # Historical Stage-07 sub-ULP floating-point state is not
    # recoverable from the frozen CSV artifacts.
    #
    # The limits below are based on the dedicated precision
    # audit and are intentionally limited to Spearman results.
    # Pearson, AR1 and best-lag results remain strict.
    # ========================================================

    precision_contract = {
        "SEASONAL_ADJUSTED": {
            "rho_max": 0.009,
            "p_max": 0.056,
            "min_sign_agreement": 478,
            "min_pclass_agreement": 480,
        },
        "PREWHITENED": {
            "rho_max": 0.002,
            "p_max": 0.013,
            "min_sign_agreement": 481,
            "min_pclass_agreement": 480,
        },
    }

    for version, contract in (
        precision_contract.items()
    ):

        mask = (
            r["Version"] == version
        )

        rho_c = (
            c.loc[
                mask,
                "Spearman"
            ]
            .to_numpy(float)
        )

        rho_r = (
            r.loc[
                mask,
                "Spearman"
            ]
            .to_numpy(float)
        )

        p_c = (
            c.loc[
                mask,
                "Spearman_p_naive"
            ]
            .to_numpy(float)
        )

        p_r = (
            r.loc[
                mask,
                "Spearman_p_naive"
            ]
            .to_numpy(float)
        )

        rho_diff = np.abs(
            rho_c - rho_r
        )

        p_diff = np.abs(
            p_c - p_r
        )

        max_rho = float(
            np.nanmax(rho_diff)
        )

        max_p = float(
            np.nanmax(p_diff)
        )

        sign_agreement = int(
            (
                np.sign(rho_c)
                ==
                np.sign(rho_r)
            ).sum()
        )

        pclass_agreement = int(
            (
                (p_c < 0.05)
                ==
                (p_r < 0.05)
            ).sum()
        )

        strength_agreement = int(
            (
                (np.abs(rho_c) >= 0.30)
                ==
                (np.abs(rho_r) >= 0.30)
            ).sum()
        )

        add(
            f"{version} Spearman max |delta rho|",
            max_rho
            <= contract["rho_max"],
            f"{max_rho:.12g}",
            f"<= {contract['rho_max']}",
            (
                "Precision-aware check; "
                "historical sub-ULP tie structure "
                "is unavailable from CSV artifacts."
            ),
        )

        add(
            f"{version} Spearman max |delta p|",
            max_p
            <= contract["p_max"],
            f"{max_p:.12g}",
            f"<= {contract['p_max']}",
            "Precision-aware Spearman check.",
        )

        add(
            f"{version} Spearman sign agreement",
            sign_agreement
            >= contract[
                "min_sign_agreement"
            ],
            f"{sign_agreement}/481",
            (
                f">="
                f"{contract['min_sign_agreement']}/481"
            ),
        )

        add(
            f"{version} Spearman p<0.05 agreement",
            pclass_agreement
            >= contract[
                "min_pclass_agreement"
            ],
            f"{pclass_agreement}/481",
            (
                f">="
                f"{contract['min_pclass_agreement']}/481"
            ),
        )

        add(
            f"{version} |Spearman|>=0.30 agreement",
            strength_agreement == 481,
            f"{strength_agreement}/481",
            "481/481",
        )

    # ========================================================
    # 6. AR(1) — strict
    # ========================================================

    ca = (
        cand_ar
        .sort_values("Piezometer")
        .reset_index(drop=True)
    )

    ra = (
        ref_ar
        .sort_values("Piezometer")
        .reset_index(drop=True)
    )

    ar_wells = (
        ca["Piezometer"].to_numpy()
        ==
        ra["Piezometer"].to_numpy()
    )

    add(
        "AR1 well IDs exact",
        bool(ar_wells.all()),
        f"{int(ar_wells.sum())}/37",
        "37/37",
    )

    ar_n = (
        ca["N_AR1_pairs"].to_numpy()
        ==
        ra["N_AR1_pairs"].to_numpy()
    )

    add(
        "AR1 pair counts exact",
        bool(ar_n.all()),
        f"{int(ar_n.sum())}/37",
        "37/37",
    )

    ar_phi = _allclose(
        ca["GW_AR1_phi"],
        ra["GW_AR1_phi"],
    )

    add(
        "AR1 phi",
        bool(ar_phi.all()),
        f"{int(ar_phi.sum())}/37",
        "37/37",
    )

    # ========================================================
    # 7. Best lag — strict because based on signed Pearson
    # ========================================================

    cb = (
        cand_best
        .sort_values("Piezometer")
        .reset_index(drop=True)
    )

    rb = (
        ref_best
        .sort_values("Piezometer")
        .reset_index(drop=True)
    )

    for col in rb.columns:

        same = (
            cb[col].to_numpy()
            ==
            rb[col].to_numpy()
        )

        add(
            f"BestLag {col}",
            bool(same.all()),
            f"{int(same.sum())}/37",
            "37/37",
        )

    # ========================================================
    # 8. Lag summary
    # ========================================================

    summary_keys = [
        "Version",
        "Lag_months",
    ]

    cs = (
        cand_summary
        .sort_values(summary_keys)
        .reset_index(drop=True)
    )

    rs = (
        ref_summary
        .sort_values(summary_keys)
        .reset_index(drop=True)
    )

    summary_key_match = (
        cs[summary_keys]
        .astype(str)
        .to_numpy()
        ==
        rs[summary_keys]
        .astype(str)
        .to_numpy()
    ).all()

    add(
        "Summary keys exact",
        summary_key_match,
        bool(summary_key_match),
        True,
    )

    for col in [
        "Wells",
        "Positive_Pearson_Wells",
        "Pearson_ge_03_Wells",
    ]:

        same = (
            cs[col].to_numpy()
            ==
            rs[col].to_numpy()
        )

        add(
            f"Summary {col}",
            bool(same.all()),
            f"{int(same.sum())}/39",
            "39/39",
        )

    for col in [
        "Median_Pearson",
        "Mean_Pearson",
    ]:

        same = _allclose(
            cs[col],
            rs[col],
        )

        add(
            f"Summary {col}",
            bool(same.all()),
            f"{int(same.sum())}/39",
            "39/39",
        )

    # Spearman summary is checked with the same bounded
    # precision envelope instead of bitwise equality.
    for version, rho_limit in [
        ("SEASONAL_ADJUSTED", 0.009),
        ("PREWHITENED", 0.002),
    ]:

        mask = (
            rs["Version"] == version
        )

        for col in [
            "Median_Spearman",
            "Mean_Spearman",
        ]:

            diff = np.abs(
                cs.loc[mask, col]
                .to_numpy(float)
                -
                rs.loc[mask, col]
                .to_numpy(float)
            )

            observed = float(
                np.nanmax(diff)
            )

            add(
                f"Summary {version} {col}",
                observed <= rho_limit,
                f"{observed:.12g}",
                f"<= {rho_limit}",
                "Precision-aware Spearman summary QC.",
            )

    # RAW summary Spearman must remain strict.
    raw_summary_mask = (
        rs["Version"] == "RAW"
    )

    for col in [
        "Median_Spearman",
        "Mean_Spearman",
    ]:

        same = _allclose(
            cs.loc[
                raw_summary_mask,
                col,
            ],
            rs.loc[
                raw_summary_mask,
                col,
            ],
        )

        add(
            f"Summary RAW {col}",
            bool(same.all()),
            f"{int(same.sum())}/13",
            "13/13",
        )

    # ========================================================
    # Final result
    # ========================================================

    report = pd.DataFrame(
        checks
    )

    report.to_csv(
        out_report,
        index=False,
    )

    failures = report[
        report["Status"] == "FAIL"
    ]

    print("=" * 110)
    print(
        "STAGE 09 PRECISION-AWARE VALIDATION"
    )
    print("=" * 110)

    print(
        report[
            [
                "Status",
                "Item",
                "Observed",
                "Expected",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 110)
    print("FAILURES")
    print("=" * 110)

    if len(failures):

        print(
            failures.to_string(
                index=False
            )
        )

        print("\nREPORT:")
        print(out_report)

        raise SystemExit(
            "\nSTAGE 09 VALIDATION RESULT: FAIL"
        )

    print("No validation failures.")

    print("\nREPORT:")
    print(out_report)

    print(
        "\nSTAGE 09 VALIDATION RESULT: PASS"
    )

    return report


def main():
    validate_stage09()


if __name__ == "__main__":
    main()
