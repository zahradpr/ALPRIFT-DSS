# ALPRIFT-DSS Research Project State

> This file is generated from the actual project configuration,
> frozen baseline, QC reports, regression reports, and Git state.
> Do not use it as a substitute for the underlying machine-readable files.

## Current checkpoint

- Generated UTC: `2026-11-06T11:51:22.449654+00:00`
- Research branch: `thesis/groundwater-insar-memory`
- Parent Git commit: `9ef05bf`
- Remote: `https://github.com/zahradpr/ALPRIFT-DSS.git`
- Workflow: `Groundwater–InSAR hydromechanical memory analysis`

## Project architecture

### Data and analysis workspace

`/mnt/d/ALPRIFT_DSS`

Contains:

- SourceData
- Baseline rasters
- InSAR products
- Analysis CSV outputs
- regenerated outputs
- figures
- reports

### Research Git repository

`/mnt/d/ALPRIFT_DSS_Research`

Contains:

- configuration
- Python source code
- Jupyter notebooks
- frozen references
- documentation
- Git history

## Central configuration

`/mnt/d/ALPRIFT_DSS_Research/Config/project_config.json`

Important current parameters:

- Jalali analysis period: `1395–1403`
- Raw monthly piezometers expected: `43`
- Official T wells: `41`
- Excluded wells: `P08, P49`
- Lag-eligible wells expected: `37`
- Recovery threshold: `0.5 m`
- Sustained-recovery fraction: `0.8`
- FDR alpha: `0.05`
- Bootstrap iterations: `20000`
- Random seed: `20261106`

## Verified data state

Central QC:

- PASS: `129`
- WARN: `0`
- FAIL: `0`

Expected InSAR epochs:

- Total: `143`
- 1395: `15`
- 1396–1403: `16 epochs/year`

Known groundwater missingness:

- 1402 official-well missing values: `247`
- This missingness pattern has passed the central project QC.

## Frozen reference

Reference version:

`v1_terminal_baseline`

Location:

`/mnt/d/ALPRIFT_DSS_Research/references/Groundwater_InSAR_Memory`

Frozen CSV files:

`53`

The frozen reference represents the verified terminal/WSL analysis
before reconstruction in Jupyter.

## Regression status

Reference-vs-current comparison:

- Files tested: `53`
- PASS: `53`
- FAIL: `0`

Comparison modes:

- `BYTE_IDENTICAL`: `53`


Current regression status:

`PASS`

## Current scientific workflow

The existing verified analysis consists of stages `01–26`.

The Jupyter reconstruction will organize these stages by scientific
question rather than creating 26 independent notebooks.

Planned notebook structure:

1. `00_Project_Index.ipynb`
2. `01_Data_and_Coverage_QC.ipynb`
3. `02_Monthly_Pairs.ipynb`
4. `03_CCF_and_Seasonality.ipynb`
5. `04_Extended_and_Distributed_Lag.ipynb`
6. `05_Cumulative_Analysis.ipynb`
7. `06_Residual_Compaction.ipynb`
8. `07_Post_Recovery_Event.ipynb`
9. `08_Spatial_ALPRIFT_Comparison.ipynb`
10. `09_Final_Synthesis.ipynb`

## Reproducibility rule

For every reconstructed stage:

1. Read parameters from `Config/project_config.json`.
2. Read source data from the workspace.
3. Run centralized Python functions from `src/groundwater_insar/`.
4. Save regenerated outputs outside the Git repository.
5. Compare regenerated CSVs against the frozen reference.
6. Mark the stage complete only after regression testing passes.
7. Commit the verified code/notebook state to Git.

## Current next action

Create and validate:

`notebooks/Groundwater_InSAR_Memory/00_Project_Index.ipynb`

No new scientific analysis should be introduced before the existing
verified workflow has been reconstructed and regression-tested.

