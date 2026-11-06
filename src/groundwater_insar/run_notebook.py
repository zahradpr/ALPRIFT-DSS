from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .config import (
    load_config,
    repository_root,
    workspace_root,
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


def git(repo: Path, *args):

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "notebook",
        help=(
            "Notebook filename relative to "
            "notebooks/Groundwater_InSAR_Memory"
        )
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600
    )

    parser.add_argument(
        "--kernel",
        default="python3"
    )

    args = parser.parse_args()

    cfg = load_config()

    repo = repository_root(cfg)
    workspace = workspace_root(cfg)

    source = (
        repo
        / "notebooks"
        / "Groundwater_InSAR_Memory"
        / args.notebook
    )

    if not source.exists():
        raise FileNotFoundError(source)

    stem = source.stem

    executed_dir = (
        workspace
        / "outputs"
        / "Groundwater_InSAR_Memory"
        / "executed_notebooks"
    )

    report_dir = (
        workspace
        / "reports"
        / "Groundwater_InSAR_Memory"
        / "html"
    )

    manifest_dir = (
        workspace
        / "outputs"
        / "Groundwater_InSAR_Memory"
        / "manifests"
        / "notebook_runs"
    )

    for d in [
        executed_dir,
        report_dir,
        manifest_dir
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True
        )

    executed_file = (
        executed_dir
        / f"{stem}_executed.ipynb"
    )

    html_file = (
        report_dir
        / f"{stem}.html"
    )

    manifest_file = (
        manifest_dir
        / f"{stem}_run.json"
    )

    start = datetime.now(
        timezone.utc
    )

    branch = git(
        repo,
        "branch",
        "--show-current"
    )

    commit = git(
        repo,
        "rev-parse",
        "HEAD"
    )

    status = git(
        repo,
        "status",
        "--porcelain"
    )

    source_hash = sha256_file(
        source
    )

    config_path = Path(
        cfg["_config_path"]
    )

    config_hash = sha256_file(
        config_path
    )

    manifest = {
        "notebook":
            source.name,

        "source_path":
            str(source),

        "source_sha256":
            source_hash,

        "config_path":
            str(config_path),

        "config_sha256":
            config_hash,

        "started_utc":
            start.isoformat(),

        "git_branch":
            branch,

        "git_commit":
            commit,

        "git_dirty_before_run":
            bool(status),

        "python_executable":
            sys.executable,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "kernel":
            args.kernel,

        "timeout_seconds":
            args.timeout,

        "status":
            "RUNNING"
    }

    print("=" * 100)
    print("NOTEBOOK EXECUTION")
    print("=" * 100)

    print("Notebook :", source)
    print("Kernel   :", args.kernel)
    print("Git      :", commit)
    print("Dirty    :", bool(status))
    print()

    try:

        with source.open(
            "r",
            encoding="utf-8"
        ) as f:
            nb = nbformat.read(
                f,
                as_version=4
            )

        client = NotebookClient(
            nb,
            timeout=args.timeout,
            kernel_name=args.kernel,
            allow_errors=False,
            resources={
                "metadata": {
                    "path": str(
                        source.parent
                    )
                }
            }
        )

        executed = client.execute()

        # ----------------------------------------------------
        # Save executed notebook outside Git repo
        # ----------------------------------------------------

        with executed_file.open(
            "w",
            encoding="utf-8"
        ) as f:

            nbformat.write(
                executed,
                f
            )

        # ----------------------------------------------------
        # Export presentation/report HTML
        # ----------------------------------------------------

        exporter = HTMLExporter()

        body, _ = (
            exporter.from_notebook_node(
                executed
            )
        )

        html_file.write_text(
            body,
            encoding="utf-8"
        )

        end = datetime.now(
            timezone.utc
        )

        manifest.update({
            "status":
                "PASS",

            "finished_utc":
                end.isoformat(),

            "duration_seconds":
                (
                    end - start
                ).total_seconds(),

            "executed_notebook":
                str(executed_file),

            "executed_sha256":
                sha256_file(
                    executed_file
                ),

            "html_report":
                str(html_file),

            "html_sha256":
                sha256_file(
                    html_file
                ),
        })

        print(
            "NOTEBOOK EXECUTION: PASS"
        )

    except Exception as exc:

        end = datetime.now(
            timezone.utc
        )

        manifest.update({
            "status":
                "FAIL",

            "finished_utc":
                end.isoformat(),

            "duration_seconds":
                (
                    end - start
                ).total_seconds(),

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),
        })

        with manifest_file.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "NOTEBOOK EXECUTION: FAIL"
        )

        print()
        print(
            traceback.format_exc()
        )

        print(
            "Failure manifest:",
            manifest_file
        )

        raise

    # --------------------------------------------------------
    # Save successful manifest
    # --------------------------------------------------------

    with manifest_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Executed :", executed_file)
    print("HTML     :", html_file)
    print("Manifest :", manifest_file)

    print("=" * 100)


if __name__ == "__main__":
    main()
