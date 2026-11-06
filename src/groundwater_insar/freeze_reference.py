from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json
import shutil
import subprocess

import pandas as pd

from .config import load_config


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def git_info(root: Path) -> dict:

    def run(args):

        try:

            r = subprocess.run(
                ["git"] + args,
                cwd=root,
                check=True,
                capture_output=True,
                text=True
            )

            return r.stdout.strip()

        except Exception:
            return None

    status = run(
        ["status", "--porcelain"]
    )

    return {
        "commit": run(
            ["rev-parse", "HEAD"]
        ),
        "branch": run(
            ["branch", "--show-current"]
        ),
        "dirty": (
            bool(status)
            if status is not None
            else None
        )
    }


def is_stage_csv(path: Path) -> bool:
    """
    Include every CSV whose filename begins:
    01_ ... 26_
    """

    name = path.name

    if len(name) < 3:
        return False

    prefix = name[:2]

    return (
        prefix.isdigit()
        and 1 <= int(prefix) <= 26
        and name[2] == "_"
        and path.suffix.lower() == ".csv"
    )


def verify_existing_baseline(
    manifest_path: Path,
    baseline_csv_dir: Path
) -> bool:

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    failures = []

    for rec in manifest["files"]:

        path = (
            baseline_csv_dir
            / rec["filename"]
        )

        if not path.exists():

            failures.append(
                f"MISSING: {rec['filename']}"
            )

            continue

        current_hash = sha256_file(path)

        if (
            current_hash
            != rec["snapshot_sha256"]
        ):

            failures.append(
                f"HASH CHANGED: {rec['filename']}"
            )

    if failures:

        print("=" * 100)
        print("BASELINE INTEGRITY: FAIL")
        print("=" * 100)

        for x in failures:
            print(x)

        return False

    print("=" * 100)
    print("BASELINE ALREADY EXISTS")
    print("No files were overwritten.")
    print("BASELINE INTEGRITY: PASS")
    print("=" * 100)

    return True


def main():

    cfg = load_config()

    repository_root = Path(
        cfg["paths"]["repository_root"]
    )

    analysis_root = Path(
        cfg["paths"]["analysis_root"]
    )

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

    baseline_csv_dir = (
        baseline_root
        / "csv"
    )

    manifest_path = (
        baseline_root
        / "baseline_manifest.json"
    )

    config_snapshot = (
        baseline_root
        / "project_config_at_freeze.json"
    )

    # ----------------------------------------------------------
    # Never overwrite a frozen baseline
    # ----------------------------------------------------------

    if manifest_path.exists():

        ok = verify_existing_baseline(
            manifest_path,
            baseline_csv_dir
        )

        raise SystemExit(
            0 if ok else 1
        )

    baseline_csv_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    # ----------------------------------------------------------
    # Discover all stage outputs automatically
    # ----------------------------------------------------------

    source_files = sorted(
        p
        for p in analysis_root.glob("*.csv")
        if is_stage_csv(p)
    )

    if not source_files:

        raise RuntimeError(
            "No stage CSV files 01-26 found."
        )

    stages = sorted({
        int(p.name[:2])
        for p in source_files
    })

    expected_stages = list(
        range(1, 27)
    )

    if stages != expected_stages:

        raise RuntimeError(
            f"Stage coverage incomplete. "
            f"Observed={stages}"
        )

    records = []

    print("=" * 110)
    print("FREEZING CURRENT TERMINAL ANALYSIS AS BASELINE")
    print("=" * 110)

    for source in source_files:

        target = (
            baseline_csv_dir
            / source.name
        )

        source_hash = sha256_file(
            source
        )

        shutil.copy2(
            source,
            target
        )

        snapshot_hash = sha256_file(
            target
        )

        if source_hash != snapshot_hash:

            raise RuntimeError(
                f"Copy verification failed: "
                f"{source.name}"
            )

        try:

            df = pd.read_csv(
                source
            )

            rows = int(
                len(df)
            )

            columns = list(
                df.columns
            )

            dtypes = {
                c: str(df[c].dtype)
                for c in df.columns
            }

        except Exception as exc:

            raise RuntimeError(
                f"Unable to inspect "
                f"{source.name}: {exc}"
            )

        records.append({
            "filename":
                source.name,

            "stage":
                int(source.name[:2]),

            "source_path":
                str(source),

            "bytes":
                int(source.stat().st_size),

            "rows":
                rows,

            "column_count":
                len(columns),

            "columns":
                columns,

            "dtypes":
                dtypes,

            "source_sha256":
                source_hash,

            "snapshot_sha256":
                snapshot_hash,
        })

        print(
            f"[FROZEN] "
            f"{source.name:<60} "
            f"rows={rows}"
        )

    # ----------------------------------------------------------
    # Freeze config too
    # ----------------------------------------------------------

    cfg_path = Path(
        cfg["_config_path"]
    )

    shutil.copy2(
        cfg_path,
        config_snapshot
    )

    manifest = {

        "reference_version":
            version,

        "status":
            "FROZEN",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "description": (
            "Frozen reference of the verified "
            "Ubuntu/WSL terminal groundwater-InSAR "
            "analysis before Jupyter reconstruction."
        ),

        "source_analysis_root":
            str(analysis_root),

        "project_config_source":
            str(cfg_path),

        "project_config_snapshot":
            str(config_snapshot),

        "project_config_sha256":
            sha256_file(
                config_snapshot
            ),

        "git_at_freeze":
            git_info(
                repository_root
            ),

        "stage_range":
            [1, 26],

        "total_csv_files":
            len(records),

        "files":
            records,
    }

    with manifest_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 110)
    print("BASELINE FREEZE SUMMARY")
    print("=" * 110)

    print(
        "Reference version :",
        version
    )

    print(
        "Stages frozen     :",
        "01-26"
    )

    print(
        "CSV files frozen  :",
        len(records)
    )

    print(
        "Baseline root     :",
        baseline_root
    )

    print(
        "Manifest          :",
        manifest_path
    )

    print(
        "Config snapshot   :",
        config_snapshot
    )

    print(
        "Integrity         : PASS"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
