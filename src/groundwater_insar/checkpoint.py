from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/mnt/d/ALPRIFT_DSS")

COMMIT_MESSAGE = (
    "Add spatial candidates and ALPRIFT Jupyter workflow"
)

MAX_FILE_MB = 50.0


def run(cmd, *, check=True, env=None):

    r = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )

    if check and r.returncode != 0:

        print("\nCOMMAND FAILED:")
        print(" ".join(cmd))

        if r.stdout:
            print("\nSTDOUT:")
            print(r.stdout)

        if r.stderr:
            print("\nSTDERR:")
            print(r.stderr)

        raise SystemExit(r.returncode)

    return r


def git(*args, check=True):

    return run(
        ["git", *args],
        check=check
    )


def module_env():

    env = os.environ.copy()

    src = str(ROOT / "src")

    old = env.get(
        "PYTHONPATH",
        ""
    )

    env["PYTHONPATH"] = (
        src
        if not old
        else src + os.pathsep + old
    )

    return env


def run_module(module, *args):

    print("\n" + "=" * 100)
    print("VALIDATION:", module)
    print("=" * 100)

    r = run(
        [
            sys.executable,
            "-m",
            module,
            *args
        ],
        env=module_env()
    )

    if r.stdout:
        print(r.stdout)

    return r


# =====================================================================
# 1. Confirm Git repository
# =====================================================================

r = git(
    "rev-parse",
    "--is-inside-work-tree"
)

if r.stdout.strip() != "true":
    raise SystemExit(
        "Not inside a Git working tree."
    )

print("=" * 100)
print("SAFE REPRODUCIBILITY CHECKPOINT")
print("=" * 100)

branch = git(
    "branch",
    "--show-current"
).stdout.strip()

parent = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print("Branch      :", branch)
print("Parent HEAD :", parent)


# =====================================================================
# 2. Refuse to continue if anything is already staged
# =====================================================================

already_staged = git(
    "diff",
    "--cached",
    "--name-only"
).stdout.strip()

if already_staged:

    print("\nREFUSING TO COMMIT.")
    print(
        "Files were already staged before this checkpoint:"
    )
    print(already_staged)

    raise SystemExit(
        "Clear/review the existing Git index first."
    )

print(
    "Pre-existing staged files: NONE"
)


# =====================================================================
# 3. Revalidate the frozen baseline itself
# =====================================================================

run_module(
    "groundwater_insar.freeze_reference"
)


# =====================================================================
# 4. Re-run full project QC
# =====================================================================

run_module(
    "groundwater_insar.qc"
)


# =====================================================================
# 5. Re-run baseline self-regression
# =====================================================================

run_module(
    "groundwater_insar.regression",
    "--candidate-dir",
    str(
        WORKSPACE
        / "Analysis"
        / "Groundwater_InSAR_Monthly_Lag_1395_1403"
    ),
    "--stage",
    "all",
    "--report-name",
    "00_Baseline_SelfRegression.csv"
)


# =====================================================================
# 6. Exact allow-list for this commit
# =====================================================================

allowed_paths = [
    ROOT / ".gitignore",
    ROOT / "PROJECT_STATE.md",
    ROOT / "environment.yml",

    ROOT
    / "Config"
    / "project_config.json",

    ROOT
    / "Config"
    / "workflow_registry.json",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "00_Project_Index.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "01_Data_and_Coverage_QC.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "02_Monthly_Pairs.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "03_CCF_and_Seasonality.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "04_Stitching_Cumulative_HydroMechanical.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "05_Extended_and_Distributed_Lag.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "06_Residual_Compaction.ipynb",

    ROOT
    / "notebooks"
    / "Groundwater_InSAR_Memory"
    / "07_Spatial_Candidates_and_ALPRIFT.ipynb",

    ROOT
    / "src"
    / "groundwater_insar",

    ROOT
    / "references"
    / "Groundwater_InSAR_Memory"
    / "v1_terminal_baseline",
]


# =====================================================================
# 7. File-size safety check
# =====================================================================

files = []

for p in allowed_paths:

    if not p.exists():
        raise SystemExit(
            f"Required checkpoint path missing: {p}"
        )

    if p.is_file():
        files.append(p)

    else:
        files.extend(
            f for f in p.rglob("*")
            if (
                f.is_file()
                and "__pycache__" not in f.parts
                and f.suffix != ".pyc"
            )
        )


oversized = []

total_bytes = 0

for f in files:

    size = f.stat().st_size

    total_bytes += size

    mb = size / (1024 ** 2)

    if mb > MAX_FILE_MB:

        oversized.append(
            (f, mb)
        )


print("\n" + "=" * 100)
print("CHECKPOINT SIZE QC")
print("=" * 100)

print("Files considered :", len(files))
print(
    "Total size       :",
    round(
        total_bytes / (1024 ** 2),
        2
    ),
    "MB"
)

print(
    "Per-file limit   :",
    MAX_FILE_MB,
    "MB"
)

if oversized:

    print("\nOVERSIZED FILES:")

    for f, mb in oversized:
        print(
            f"{mb:.2f} MB | {f}"
        )

    raise SystemExit(
        "Checkpoint aborted because a file "
        "should be handled with Git LFS."
    )

print("Large-file check : PASS")


# =====================================================================
# 8. Stage ONLY allow-listed paths
# =====================================================================

relative_paths = [
    str(
        p.relative_to(ROOT)
    )
    for p in allowed_paths
]

r = git(
    "add",
    "--",
    *relative_paths,
    check=False
)

if r.returncode != 0:

    print(r.stdout)
    print(r.stderr)

    raise SystemExit(
        "git add failed. Nothing will be committed."
    )


# =====================================================================
# 9. Verify staged content against allow-list
# =====================================================================

staged = [
    line.strip()
    for line in git(
        "diff",
        "--cached",
        "--name-only"
    ).stdout.splitlines()
    if line.strip()
]

if not staged:
    raise SystemExit(
        "Nothing was staged."
    )


allowed_exact = {
    ".gitignore",
    "PROJECT_STATE.md",
    "environment.yml",
    "Config/project_config.json",
    "Config/workflow_registry.json",
    "notebooks/Groundwater_InSAR_Memory/00_Project_Index.ipynb",
    "notebooks/Groundwater_InSAR_Memory/01_Data_and_Coverage_QC.ipynb",
    "notebooks/Groundwater_InSAR_Memory/02_Monthly_Pairs.ipynb",
    "notebooks/Groundwater_InSAR_Memory/03_CCF_and_Seasonality.ipynb",
    "notebooks/Groundwater_InSAR_Memory/04_Stitching_Cumulative_HydroMechanical.ipynb",
    "notebooks/Groundwater_InSAR_Memory/05_Extended_and_Distributed_Lag.ipynb",
    "notebooks/Groundwater_InSAR_Memory/06_Residual_Compaction.ipynb",
    "notebooks/Groundwater_InSAR_Memory/07_Spatial_Candidates_and_ALPRIFT.ipynb",
}

allowed_prefixes = (
    "src/groundwater_insar/",
    (
        "references/"
        "Groundwater_InSAR_Memory/"
        "v1_terminal_baseline/"
    ),
)


unexpected = []

for name in staged:

    ok = (
        name in allowed_exact
        or any(
            name.startswith(prefix)
            for prefix in allowed_prefixes
        )
    )

    if not ok:
        unexpected.append(name)


if unexpected:

    print("\nUNEXPECTED STAGED FILES:")

    for name in unexpected:
        print(name)

    git(
        "reset",
        "--",
        *relative_paths,
        check=False
    )

    raise SystemExit(
        "Safety stop: unexpected staged content."
    )


print("\n" + "=" * 100)
print("FILES STAGED FOR CHECKPOINT")
print("=" * 100)

for name in staged:
    print(name)

print(
    "\nStaged file count:",
    len(staged)
)


# =====================================================================
# 10. Commit
# =====================================================================

commit = git(
    "commit",
    "-m",
    COMMIT_MESSAGE,
    check=False
)

if commit.returncode != 0:

    print("\nGIT COMMIT FAILED")

    if commit.stdout:
        print(commit.stdout)

    if commit.stderr:
        print(commit.stderr)

    # We entered with a clean index, so it is safe
    # to unstage only our allow-listed paths.
    git(
        "reset",
        "--",
        *relative_paths,
        check=False
    )

    raise SystemExit(
        "Commit aborted; staged checkpoint files were reset."
    )


new_head = git(
    "rev-parse",
    "HEAD"
).stdout.strip()


# =====================================================================
# 11. Final audit
# =====================================================================

print("\n" + "=" * 100)
print("GIT CHECKPOINT RESULT")
print("=" * 100)

print("Result      : PASS")
print("Branch      :", branch)
print("Commit      :", new_head)
print("Message     :", COMMIT_MESSAGE)

print("\nCommitted files:")
print(
    git(
        "show",
        "--name-only",
        "--format=",
        "HEAD"
    ).stdout.strip()
)

print("\n" + "=" * 100)
print("REMAINING WORKING-TREE CHANGES")
print("=" * 100)

status = git(
    "status",
    "--short"
).stdout.strip()

if status:
    print(status)
    print(
        "\nThese were NOT included automatically "
        "in the checkpoint."
    )
else:
    print("Working tree clean.")

print("=" * 100)
