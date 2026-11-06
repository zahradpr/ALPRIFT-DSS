from __future__ import annotations

from pathlib import Path
import json

from .config import (
    load_config,
    notebook_root,
)


def markdown(text: str) -> dict:

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            line + "\n"
            for line in text.strip().splitlines()
        ],
    }


def code(text: str) -> dict:

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            line + "\n"
            for line in text.strip().splitlines()
        ],
    }


def validate_notebook(
    notebook: dict
) -> None:

    if notebook.get(
        "nbformat"
    ) != 4:
        raise ValueError(
            "Notebook format must be v4."
        )

    cells = notebook.get(
        "cells",
        []
    )

    if not cells:
        raise ValueError(
            "Notebook has no cells."
        )

    for i, cell in enumerate(cells):

        if cell["cell_type"] == "code":

            source = "".join(
                cell["source"]
            )

            try:
                compile(
                    source,
                    f"<cell-{i}>",
                    "exec"
                )

            except SyntaxError as exc:

                raise SyntaxError(
                    f"Code cell {i} "
                    f"failed validation: {exc}"
                ) from exc


def main():

    cfg = load_config()

    out_dir = notebook_root(cfg)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out = (
        out_dir
        / "00_Project_Index.ipynb"
    )

    cells = [

        markdown(
r"""
# Groundwater–InSAR Hydromechanical Memory Analysis

## Project Index and Reproducibility Dashboard

This notebook is the central entry point for the groundwater–InSAR
analysis of the Shabestar Plain.

It does **not** contain the primary scientific calculations.

Its purpose is to report, directly from machine-readable project files:

- Git state;
- central project QC;
- frozen reference status;
- regression-test status;
- stage `01–26` inventory;
- Jupyter reconstruction progress;
- current verified scientific checkpoint.

The numerical analysis is reconstructed in the downstream notebooks.
"""
        ),

        markdown(
r"""
## 1. Initialize the research environment

The repository root is discovered automatically from the notebook
location. No analysis path is typed manually in this notebook.
"""
        ),

        code(
r"""
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display, Markdown

cwd = Path.cwd().resolve()

repo_candidates = [
    cwd,
    *cwd.parents
]

REPO = next(
    (
        p
        for p in repo_candidates
        if (
            p
            / "Config"
            / "project_config.json"
        ).exists()
    ),
    None
)

if REPO is None:
    raise RuntimeError(
        "Research repository root could not be discovered."
    )

SRC = REPO / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC)
    )

from groundwater_insar.config import (
    load_config,
    workspace_root,
    repository_root,
    analysis_root,
    reference_root,
)

from groundwater_insar.project_status import (
    project_summary,
    stage_inventory,
    workflow_table,
    qc_report,
    regression_report,
)

cfg = load_config(
    REPO
    / "Config"
    / "project_config.json"
)

print(
    "Research repository:",
    repository_root(cfg)
)

print(
    "Data workspace     :",
    workspace_root(cfg)
)
"""
        ),

        markdown(
r"""
## 2. Reproducibility checkpoint

A scientific reconstruction is considered trustworthy only when:

1. central QC has no failures;
2. the frozen reference is intact;
3. regenerated results pass regression testing;
4. the Git state identifying the analysis is known.
"""
        ),

        code(
r"""
summary = project_summary(cfg)

summary_df = pd.DataFrame(
    {
        "Item": [
            "Git branch",
            "Git commit",
            "Git working tree dirty",
            "Central QC PASS",
            "Central QC WARN",
            "Central QC FAIL",
            "Regression PASS",
            "Regression FAIL",
            "Frozen baseline CSVs",
            "Reference stages"
        ],
        "Value": [
            summary["git_branch"],
            summary["git_commit_short"],
            summary["git_dirty"],
            summary["qc_pass"],
            summary["qc_warn"],
            summary["qc_fail"],
            summary["regression_pass"],
            summary["regression_fail"],
            summary["baseline_csv_files"],
            summary["stages_with_reference"]
        ]
    }
)

display(summary_df)

assert summary["qc_fail"] == 0
assert summary["regression_fail"] == 0
assert summary["baseline_csv_files"] > 0
assert summary["stages_with_reference"] == 26

print(
    "REPRODUCIBILITY CHECKPOINT: PASS"
)
"""
        ),

        markdown(
r"""
## 3. Project architecture

Raw/processed data and generated results are kept outside the research
Git working tree. Code, configuration, notebooks, documentation and
frozen references are version-controlled.
"""
        ),

        code(
r"""
architecture = pd.DataFrame(
    {
        "Role": [
            "Data / analysis workspace",
            "Research Git repository",
            "Analysis checkpoint directory",
            "Frozen reference directory",
            "Notebook directory"
        ],
        "Path": [
            cfg["paths"]["workspace_root"],
            cfg["paths"]["repository_root"],
            cfg["paths"]["analysis_root"],
            cfg["paths"]["reference_root"],
            cfg["paths"]["notebook_root"]
        ]
    }
)

display(architecture)
"""
        ),

        markdown(
r"""
## 4. Scientific workflow map

The original terminal analysis contains stages `01–26`.

The Jupyter reconstruction groups them by scientific question rather
than reproducing 26 disconnected notebooks.
"""
        ),

        code(
r"""
workflow = workflow_table(cfg)

display(
    workflow[
        [
            "Order",
            "Notebook",
            "Scientific_Title",
            "Stages",
            "Reference_Ready",
            "Notebook_Exists",
            "Status"
        ]
    ]
)
"""
        ),

        markdown(
r"""
## 5. Frozen stage inventory

This inventory is generated from the frozen baseline manifest.

It provides an auditable link between each terminal-analysis stage and
the reference files against which the Jupyter reconstruction will be
tested.
"""
        ),

        code(
r"""
inventory = stage_inventory(cfg)

display(
    inventory[
        [
            "Stage",
            "Reference_Files",
            "Total_Rows",
            "Files"
        ]
    ]
)

assert (
    inventory["Reference_Files"] > 0
).all()

print(
    "STAGE INVENTORY 01–26: COMPLETE"
)
"""
        ),

        markdown(
r"""
## 6. Central QC

Warnings and failures are displayed first. A clean result means the
central structural/data checks currently pass.
"""
        ),

        code(
r"""
qc = qc_report(cfg)

problems = qc[
    qc["Status"].isin(
        ["WARN", "FAIL"]
    )
]

if len(problems):
    display(problems)
else:
    print(
        "Central QC contains no "
        "warnings or failures."
    )

print()
print(
    qc["Status"]
    .value_counts()
    .to_string()
)
"""
        ),

        markdown(
r"""
## 7. Frozen-reference regression

The current verified terminal outputs are compared against the frozen
baseline. During reconstruction, the same mechanism will be used for
each regenerated stage.
"""
        ),

        code(
r"""
reg = regression_report(cfg)

display(
    reg[
        [
            "Stage",
            "File",
            "Status",
            "Comparison"
        ]
    ]
)

assert (
    reg["Status"] == "PASS"
).all()

print()
print(
    "REGRESSION CHECKPOINT: PASS"
)
"""
        ),

        markdown(
r"""
## 8. Current verified scientific checkpoint

The table below is read directly from stage `26`.

No post-recovery result is typed manually into this notebook.

The primary subset is the **SUSTAINED_RECOVERY** cohort, because its
selection is based on groundwater behavior rather than subsidence.
"""
        ),

        code(
r"""
stage26 = (
    analysis_root(cfg)
    / "26_PostRecovery_EventAligned_Summary.csv"
)

if not stage26.exists():
    raise FileNotFoundError(
        stage26
    )

result26 = pd.read_csv(
    stage26
)

sustained = result26[
    result26["Cohort"]
    == "SUSTAINED_RECOVERY"
].copy()

columns = [
    "Horizon_months",
    "Wells",
    "Median_Recovery_At_Horizon_m",
    "Median_Net_PostRecovery_Subsidence_mm",
    "Pct_Positive_Subsidence",
    "SignTest_FDR_q",
    "Wilcoxon_FDR_q"
]

display(
    sustained[columns]
)

significant = sustained[
    (
        sustained["SignTest_FDR_q"]
        < cfg["analysis"]["fdr_alpha"]
    )
    |
    (
        sustained["Wilcoxon_FDR_q"]
        < cfg["analysis"]["fdr_alpha"]
    )
]

print()
print(
    "FDR-supported horizons:",
    significant[
        "Horizon_months"
    ].tolist()
)
"""
        ),

        markdown(
r"""
## 9. Reconstruction rules

For every downstream notebook:

1. parameters are read from `Config/project_config.json`;
2. scientific calculations are implemented in `src/groundwater_insar`;
3. notebooks explain the scientific question, assumptions, QC and interpretation;
4. regenerated machine outputs are saved outside the Git repository;
5. regenerated outputs are regression-tested against the frozen reference;
6. a stage is marked complete only after regression testing passes;
7. intentional methodological changes create a new version rather than overwriting the original reference.

### Next notebook

`01_Data_and_Coverage_QC.ipynb`
"""
        ),
    ]

    notebook = {
        "cells": cells,

        "metadata": {
            "kernelspec": {
                "display_name":
                    "Python 3",

                "language":
                    "python",

                "name":
                    "python3"
            },

            "language_info": {
                "name":
                    "python",

                "version":
                    "3"
            }
        },

        "nbformat": 4,
        "nbformat_minor": 5
    }

    validate_notebook(
        notebook
    )

    with out.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            notebook,
            f,
            indent=1,
            ensure_ascii=False
        )

    # Read-back validation
    with out.open(
        "r",
        encoding="utf-8"
    ) as f:
        check = json.load(f)

    validate_notebook(
        check
    )

    print("=" * 90)
    print("PROJECT INDEX NOTEBOOK GENERATED")
    print("=" * 90)
    print(out)
    print(
        "Cells:",
        len(check["cells"])
    )
    print(
        "Code cells:",
        sum(
            c["cell_type"] == "code"
            for c in check["cells"]
        )
    )
    print(
        "Markdown cells:",
        sum(
            c["cell_type"] == "markdown"
            for c in check["cells"]
        )
    )
    print(
        "NOTEBOOK STRUCTURAL VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
