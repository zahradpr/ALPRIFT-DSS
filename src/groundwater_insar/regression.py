from __future__ import annotations

from pathlib import Path

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .config import (
    load_config,
    qc_root
)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def canonicalize(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Make comparison independent of row order,
    while preserving column order and values.
    """

    x = df.copy()

    # Normalize infinities
    x = x.replace(
        [np.inf, -np.inf],
        [np.nan, np.nan]
    )

    # Sorting by all columns removes row-order dependence.
    # mergesort keeps duplicate-row behavior deterministic.
    if len(x) > 1 and len(x.columns) > 0:

        try:

            x = x.sort_values(
                by=list(x.columns),
                kind="mergesort",
                na_position="first"
            )

        except Exception:

            # Fallback stable row representation
            helper = x.astype(
                "string"
            ).fillna("<NA>")

            key = helper.agg(
                "\x1f".join,
                axis=1
            )

            x = (
                x.assign(
                    __sort_key=key
                )
                .sort_values(
                    "__sort_key",
                    kind="mergesort"
                )
                .drop(
                    columns="__sort_key"
                )
            )

    return x.reset_index(
        drop=True
    )


def semantic_compare(
    reference: Path,
    candidate: Path,
    rtol: float,
    atol: float
) -> tuple[bool, str]:

    ref = pd.read_csv(
        reference
    )

    cand = pd.read_csv(
        candidate
    )

    # Column names/order are part of the contract
    if list(ref.columns) != list(cand.columns):

        return (
            False,
            "COLUMN_MISMATCH"
        )

    if len(ref) != len(cand):

        return (
            False,
            (
                f"ROW_COUNT_MISMATCH "
                f"{len(ref)} != {len(cand)}"
            )
        )

    ref = canonicalize(
        ref
    )

    cand = canonicalize(
        cand
    )

    try:

        assert_frame_equal(
            ref,
            cand,
            check_dtype=False,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            check_names=True
        )

        return (
            True,
            "SEMANTIC_MATCH"
        )

    except AssertionError as exc:

        short = str(exc).splitlines()[0]

        return (
            False,
            f"VALUE_MISMATCH: {short}"
        )


def load_manifest(
    cfg: dict
) -> tuple[dict, Path]:

    reference_root = Path(
        cfg["paths"]["reference_root"]
    )

    version = (
        cfg["analysis"][
            "frozen_reference_version"
        ]
    )

    baseline_root = (
        reference_root
        / version
    )

    manifest_path = (
        baseline_root
        / "baseline_manifest.json"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            "Frozen baseline manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    return (
        manifest,
        baseline_root
    )


def compare(
    candidate_dir: Path,
    stage: str = "all"
) -> tuple[pd.DataFrame, bool]:

    cfg = load_config()

    manifest, baseline_root = (
        load_manifest(cfg)
    )

    baseline_csv = (
        baseline_root
        / "csv"
    )

    rtol = float(
        cfg["analysis"][
            "regression_rtol"
        ]
    )

    atol = float(
        cfg["analysis"][
            "regression_atol"
        ]
    )

    refs = manifest["files"]

    if stage != "all":

        stage_num = int(stage)

        refs = [
            r
            for r in refs
            if int(r["stage"]) == stage_num
        ]

        if not refs:

            raise ValueError(
                f"No reference files for stage "
                f"{stage_num:02d}"
            )

    rows = []

    for rec in refs:

        name = rec["filename"]

        ref = (
            baseline_csv
            / name
        )

        cand = (
            candidate_dir
            / name
        )

        if not cand.exists():

            rows.append({
                "Stage":
                    rec["stage"],

                "File":
                    name,

                "Status":
                    "FAIL",

                "Comparison":
                    "MISSING_CANDIDATE",

                "Reference_Rows":
                    rec["rows"],

                "Candidate_Rows":
                    np.nan,
            })

            continue

        # ------------------------------------------------------
        # First: exact byte identity
        # ------------------------------------------------------

        cand_hash = sha256_file(
            cand
        )

        if (
            cand_hash
            == rec["snapshot_sha256"]
        ):

            rows.append({
                "Stage":
                    rec["stage"],

                "File":
                    name,

                "Status":
                    "PASS",

                "Comparison":
                    "BYTE_IDENTICAL",

                "Reference_Rows":
                    rec["rows"],

                "Candidate_Rows":
                    rec["rows"],
            })

            continue

        # ------------------------------------------------------
        # Second: semantic/numerical comparison
        # ------------------------------------------------------

        try:

            ok, reason = semantic_compare(
                ref,
                cand,
                rtol=rtol,
                atol=atol
            )

            try:
                candidate_rows = len(
                    pd.read_csv(cand)
                )
            except Exception:
                candidate_rows = np.nan

            rows.append({
                "Stage":
                    rec["stage"],

                "File":
                    name,

                "Status":
                    "PASS" if ok else "FAIL",

                "Comparison":
                    reason,

                "Reference_Rows":
                    rec["rows"],

                "Candidate_Rows":
                    candidate_rows,
            })

        except Exception as exc:

            rows.append({
                "Stage":
                    rec["stage"],

                "File":
                    name,

                "Status":
                    "FAIL",

                "Comparison":
                    f"ERROR: {repr(exc)}",

                "Reference_Rows":
                    rec["rows"],

                "Candidate_Rows":
                    np.nan,
            })

    report = pd.DataFrame(
        rows
    )

    passed = bool(
        len(report)
        and
        (report["Status"] == "PASS").all()
    )

    return (
        report,
        passed
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-dir",
        default=None,
        help=(
            "Directory containing regenerated CSVs. "
            "For initial self-test use the original "
            "Analysis directory."
        )
    )

    parser.add_argument(
        "--stage",
        default="all",
        help=(
            "all or stage number, e.g. 07"
        )
    )

    parser.add_argument(
        "--report-name",
        default="00_Regression_Report.csv"
    )

    args = parser.parse_args()

    cfg = load_config()

    if args.candidate_dir:

        candidate_dir = Path(
            args.candidate_dir
        )

    else:

        candidate_dir = Path(
            cfg["paths"][
                "regenerated_root"
            ]
        )

    if not candidate_dir.exists():

        raise FileNotFoundError(
            candidate_dir
        )

    report, passed = compare(
        candidate_dir=candidate_dir,
        stage=args.stage
    )

    out = (
        qc_root(cfg)
        / args.report_name
    )

    report.to_csv(
        out,
        index=False
    )

    print("=" * 120)
    print("REGRESSION TEST")
    print("=" * 120)

    print(
        "Candidate directory:",
        candidate_dir
    )

    print(
        "Stage:",
        args.stage
    )

    counts = (
        report["Comparison"]
        .value_counts()
    )

    print("\nCOMPARISON COUNTS:")
    print(
        counts.to_string()
    )

    failures = report[
        report["Status"] == "FAIL"
    ]

    print("\n" + "=" * 120)
    print("FAILURES")
    print("=" * 120)

    if len(failures):

        print(
            failures.to_string(
                index=False
            )
        )

    else:

        print(
            "No regression failures."
        )

    print("\nREPORT:")
    print(out)

    print("\n" + "=" * 120)

    if passed:

        print(
            "REGRESSION RESULT: PASS"
        )

    else:

        print(
            "REGRESSION RESULT: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
