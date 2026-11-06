from __future__ import annotations

from pathlib import Path
import json
import subprocess

import pandas as pd

from .config import (
    load_config,
    repository_root,
    analysis_root,
    output_root,
    reference_root,
    notebook_root,
)


def _git(repo: Path, *args) -> str | None:

    try:
        r = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True
        )
        return r.stdout.strip()

    except Exception:
        return None


def git_state(cfg: dict) -> dict:

    repo = repository_root(cfg)

    status = _git(
        repo,
        "status",
        "--porcelain"
    )

    return {
        "branch": _git(
            repo,
            "branch",
            "--show-current"
        ),
        "commit": _git(
            repo,
            "rev-parse",
            "HEAD"
        ),
        "commit_short": _git(
            repo,
            "rev-parse",
            "--short",
            "HEAD"
        ),
        "dirty": (
            bool(status)
            if status is not None
            else None
        )
    }


def baseline_manifest(cfg: dict) -> dict:

    version = cfg["analysis"][
        "frozen_reference_version"
    ]

    path = (
        reference_root(cfg)
        / version
        / "baseline_manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def qc_report(cfg: dict) -> pd.DataFrame:

    path = (
        output_root(cfg)
        / "qc"
        / "00_Project_QC.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def regression_report(
    cfg: dict,
    filename: str = (
        "00_ResearchWorktree_SelfRegression.csv"
    )
) -> pd.DataFrame:

    path = (
        output_root(cfg)
        / "qc"
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def stage_inventory(
    cfg: dict
) -> pd.DataFrame:

    manifest = baseline_manifest(cfg)

    rows = []

    for stage in range(1, 27):

        files = [
            x
            for x in manifest["files"]
            if int(x["stage"]) == stage
        ]

        rows.append({
            "Stage": f"{stage:02d}",
            "Reference_Files": len(files),
            "Total_Rows": sum(
                int(x["rows"])
                for x in files
            ),
            "Files": " | ".join(
                x["filename"]
                for x in files
            )
        })

    return pd.DataFrame(rows)


def workflow_registry(
    cfg: dict
) -> dict:

    path = (
        repository_root(cfg)
        / "Config"
        / "workflow_registry.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def workflow_table(
    cfg: dict
) -> pd.DataFrame:

    registry = workflow_registry(cfg)

    stage_df = stage_inventory(cfg)

    available = set(
        int(x)
        for x in stage_df.loc[
            stage_df["Reference_Files"] > 0,
            "Stage"
        ]
    )

    nroot = notebook_root(cfg)

    rows = []

    for item in registry["notebooks"]:

        stages = [
            int(x)
            for x in item["stages"]
        ]

        reference_ready = (
            all(
                s in available
                for s in stages
            )
            if stages
            else True
        )

        notebook_exists = (
            nroot
            / item["filename"]
        ).exists()

        if item["order"] == 0:
            status = (
                "INDEX_PRESENT"
                if notebook_exists
                else "INDEX_PENDING"
            )

        elif notebook_exists:
            status = "NOTEBOOK_PRESENT"

        elif reference_ready:
            status = "READY_FOR_RECONSTRUCTION"

        else:
            status = "REFERENCE_INCOMPLETE"

        rows.append({
            "Order":
                item["order"],

            "Notebook":
                item["filename"],

            "Scientific_Title":
                item["title"],

            "Stages":
                ",".join(
                    f"{s:02d}"
                    for s in stages
                ),

            "Reference_Ready":
                reference_ready,

            "Notebook_Exists":
                notebook_exists,

            "Status":
                status,

            "Purpose":
                item["purpose"],
        })

    return pd.DataFrame(rows)


def project_summary(
    cfg: dict | None = None
) -> dict:

    if cfg is None:
        cfg = load_config()

    git = git_state(cfg)

    qc = qc_report(cfg)

    reg = regression_report(cfg)

    manifest = baseline_manifest(cfg)

    stages = stage_inventory(cfg)

    qc_counts = (
        qc["Status"]
        .value_counts()
        .to_dict()
    )

    reg_counts = (
        reg["Status"]
        .value_counts()
        .to_dict()
    )

    return {
        "git_branch":
            git["branch"],

        "git_commit":
            git["commit"],

        "git_commit_short":
            git["commit_short"],

        "git_dirty":
            git["dirty"],

        "qc_pass":
            int(
                qc_counts.get(
                    "PASS",
                    0
                )
            ),

        "qc_warn":
            int(
                qc_counts.get(
                    "WARN",
                    0
                )
            ),

        "qc_fail":
            int(
                qc_counts.get(
                    "FAIL",
                    0
                )
            ),

        "regression_pass":
            int(
                reg_counts.get(
                    "PASS",
                    0
                )
            ),

        "regression_fail":
            int(
                reg_counts.get(
                    "FAIL",
                    0
                )
            ),

        "baseline_csv_files":
            int(
                manifest[
                    "total_csv_files"
                ]
            ),

        "stages_with_reference":
            int(
                (
                    stages[
                        "Reference_Files"
                    ] > 0
                ).sum()
            ),

        "workspace":
            cfg["paths"][
                "workspace_root"
            ],

        "repository":
            cfg["paths"][
                "repository_root"
            ],

        "analysis_root":
            str(
                analysis_root(cfg)
            ),
    }
