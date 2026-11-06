from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import rasterio

from .config import (
    load_config,
    workspace_root,
    repository_root,
    analysis_root,
    qc_root,
    manifest_root,
)


class ProjectQC:

    def __init__(self, cfg: dict):

        self.cfg = cfg

        self.workspace = workspace_root(cfg)
        self.repo = repository_root(cfg)
        self.root = self.workspace  # backward compatibility
        self.analysis = analysis_root(cfg)

        self.records = []
        self.rasters = []

    # ================================================================
    # Generic QC recorder
    # ================================================================

    def add(
        self,
        category,
        item,
        status,
        observed="",
        expected="",
        details=""
    ):

        if status not in {
            "PASS",
            "WARN",
            "FAIL"
        }:
            raise ValueError(
                f"Invalid QC status: {status}"
            )

        self.records.append({
            "Category": category,
            "Item": item,
            "Status": status,
            "Observed": str(observed),
            "Expected": str(expected),
            "Details": str(details),
        })

    def exists(
        self,
        category,
        label,
        path: Path,
        required=True
    ):

        ok = path.exists()

        self.add(
            category,
            label,
            "PASS" if ok else (
                "FAIL" if required else "WARN"
            ),
            observed=path,
            expected="exists"
        )

        return ok

    # ================================================================
    # Raster metadata
    # ================================================================

    def inspect_raster(
        self,
        label,
        path: Path,
        year=None,
        expected_crs="EPSG:32638",
        expected_res=100.0
    ):

        if not path.exists():

            self.add(
                "Raster",
                label,
                "FAIL",
                observed=path,
                expected="exists"
            )

            return

        try:

            with rasterio.open(path) as src:

                crs = (
                    src.crs.to_string()
                    if src.crs
                    else None
                )

                rx = abs(
                    float(src.transform.a)
                )

                ry = abs(
                    float(src.transform.e)
                )

                nodata = src.nodata

                self.rasters.append({
                    "Label": label,
                    "Year": year,
                    "Path": str(path),
                    "CRS": crs,
                    "Resolution_X": rx,
                    "Resolution_Y": ry,
                    "Width": src.width,
                    "Height": src.height,
                    "Count": src.count,
                    "Dtype": src.dtypes[0],
                    "NoData": nodata,
                })

                crs_ok = (
                    crs == expected_crs
                )

                res_ok = (
                    np.isclose(
                        rx,
                        expected_res
                    )
                    and
                    np.isclose(
                        ry,
                        expected_res
                    )
                )

                self.add(
                    "Raster",
                    f"{label}: CRS",
                    "PASS" if crs_ok else "FAIL",
                    observed=crs,
                    expected=expected_crs
                )

                self.add(
                    "Raster",
                    f"{label}: resolution",
                    "PASS" if res_ok else "FAIL",
                    observed=f"{rx} x {ry}",
                    expected=f"{expected_res} x {expected_res} m"
                )

        except Exception as exc:

            self.add(
                "Raster",
                label,
                "FAIL",
                observed="unable to open",
                details=repr(exc)
            )

    # ================================================================
    # Directory structure
    # ================================================================

    def check_structure(self):

        workspace_dirs = [
            self.workspace / "SourceData" / "InSAR",
            self.workspace / "SourceData" / "Piezometers",
            self.workspace / "Baseline" / "SVI",
            self.workspace / "Baseline" / "T",
            self.analysis,
            Path(self.cfg["paths"]["output_root"]),
        ]

        repository_dirs = [
            self.repo / "Config",
            self.repo / "src" / "groundwater_insar",
            self.repo / "notebooks" / "Groundwater_InSAR_Memory",
            self.repo / "references" / "Groundwater_InSAR_Memory",
        ]

        for path in workspace_dirs:

            self.exists(
                "Workspace structure",
                str(path.relative_to(self.workspace)),
                path
            )

        for path in repository_dirs:

            self.exists(
                "Repository structure",
                str(path.relative_to(self.repo)),
                path
            )

        self.exists(
            "Repository structure",
            "Config/project_config.json",
            self.repo / "Config" / "project_config.json"
        )

    # ================================================================
    # Piezometer source QC
    # ================================================================

    def check_piezometers(self):

        path = Path(
            self.cfg["paths"][
                "piezometer_monthly"
            ]
        )

        if not self.exists(
            "Piezometer",
            "Monthly source",
            path
        ):
            return

        try:

            df = pd.read_csv(path)

        except Exception as exc:

            self.add(
                "Piezometer",
                "Read monthly CSV",
                "FAIL",
                details=repr(exc)
            )

            return

        # ------------------------------------------------------------
        # Required columns
        # ------------------------------------------------------------

        required = {
            "Year",
            "Month"
        }

        missing = (
            required
            - set(df.columns)
        )

        self.add(
            "Piezometer",
            "Year/Month columns",
            "PASS" if not missing else "FAIL",
            observed=sorted(df.columns[:5]),
            expected="Year, Month",
            details=(
                ""
                if not missing
                else f"Missing: {sorted(missing)}"
            )
        )

        if missing:
            return

        # ------------------------------------------------------------
        # Unique year-month
        # ------------------------------------------------------------

        dup = int(
            df.duplicated(
                ["Year", "Month"]
            ).sum()
        )

        self.add(
            "Piezometer",
            "Duplicate year-month rows",
            "PASS" if dup == 0 else "FAIL",
            observed=dup,
            expected=0
        )

        # ------------------------------------------------------------
        # Time coverage
        # ------------------------------------------------------------

        min_year = int(
            df["Year"].min()
        )

        max_year = int(
            df["Year"].max()
        )

        expected_rows = (
            (max_year - min_year + 1)
            * 12
        )

        self.add(
            "Piezometer",
            "Monthly row count",
            "PASS"
            if len(df) == expected_rows
            else "WARN",
            observed=len(df),
            expected=expected_rows
        )

        self.add(
            "Piezometer",
            "Source year range",
            "PASS",
            observed=f"{min_year}-{max_year}",
            expected="reported"
        )

        # ------------------------------------------------------------
        # Well count
        # ------------------------------------------------------------

        well_cols = [
            c
            for c in df.columns
            if c not in {
                "Year",
                "Month"
            }
        ]

        expected_raw = int(
            self.cfg["analysis"][
                "raw_monthly_wells_expected"
            ]
        )

        self.add(
            "Piezometer",
            "Raw monthly well count",
            "PASS"
            if len(well_cols) == expected_raw
            else "FAIL",
            observed=len(well_cols),
            expected=expected_raw
        )

        # ------------------------------------------------------------
        # Official well count after exclusions
        # ------------------------------------------------------------

        excluded = set(
            self.cfg["analysis"][
                "excluded_from_official_T"
            ]
        )

        missing_exclusions = (
            excluded
            - set(well_cols)
        )

        self.add(
            "Piezometer",
            "Excluded wells present in source",
            "PASS"
            if not missing_exclusions
            else "FAIL",
            observed=sorted(
                excluded & set(well_cols)
            ),
            expected=sorted(excluded),
            details=(
                ""
                if not missing_exclusions
                else (
                    "Missing exclusions: "
                    f"{sorted(missing_exclusions)}"
                )
            )
        )

        official = [
            c
            for c in well_cols
            if c not in excluded
        ]

        expected_official = int(
            self.cfg["analysis"][
                "official_wells"
            ]
        )

        self.add(
            "Piezometer",
            "Official T well count",
            "PASS"
            if len(official) == expected_official
            else "FAIL",
            observed=len(official),
            expected=expected_official
        )

        # ------------------------------------------------------------
        # 1402 known missingness — report explicitly
        # ------------------------------------------------------------

        y1402 = df[
            df["Year"] == 1402
        ].copy()

        if len(y1402) == 12:

            official_missing = int(
                y1402[
                    official
                ].isna().sum().sum()
            )

            self.add(
                "Piezometer",
                "1402 official-well missing values",
                "PASS"
                if official_missing == 247
                else "WARN",
                observed=official_missing,
                expected=247,
                details=(
                    "Known pattern: months 1,2,3,5,6,7 "
                    "missing for 41 wells + P53 month 8."
                )
            )

        else:

            self.add(
                "Piezometer",
                "1402 monthly rows",
                "FAIL",
                observed=len(y1402),
                expected=12
            )

    # ================================================================
    # InSAR source QC
    # ================================================================

    def check_insar(self):

        insar_root = Path(
            self.cfg["paths"]["insar_root"]
        )

        years = range(
            int(
                self.cfg["analysis"][
                    "jalali_start_year"
                ]
            ),
            int(
                self.cfg["analysis"][
                    "jalali_end_year"
                ]
            ) + 1
        )

        expected_epoch_map = (
            self.cfg["analysis"][
                "expected_insar_epochs_by_year"
            ]
        )

        total_epochs = 0

        for year in years:

            yr = insar_root / str(year)

            final = (
                yr
                / f"Shabestar_{year}_Subsidence_100m_FINAL.tif"
            )

            mask = (
                yr
                / f"CommonMask_ALPRIFT_InSAR_{year}.tif"
            )

            csv = (
                yr
                / (
                    "Shabestar_Subsidence_TimeSeries_"
                    f"100m_ALPRIFT_{year}.csv"
                )
            )

            ts_dir = (
                yr
                / "timeseries_Subsidence_100m_ALPRIFT"
            )

            self.exists(
                "InSAR",
                f"{year} final raster",
                final
            )

            self.exists(
                "InSAR",
                f"{year} common mask",
                mask
            )

            self.exists(
                "InSAR",
                f"{year} time-series CSV",
                csv
            )

            self.exists(
                "InSAR",
                f"{year} epoch raster directory",
                ts_dir
            )

            if ts_dir.exists():

                epoch_rasters = sorted(
                    ts_dir.glob(
                        "Subsidence_Cumulative_*_100m_ALPRIFT.tif"
                    )
                )

                observed = len(
                    epoch_rasters
                )

                expected = int(
                    expected_epoch_map[
                        str(year)
                    ]
                )

                total_epochs += observed

                self.add(
                    "InSAR",
                    f"{year} epoch count",
                    "PASS"
                    if observed == expected
                    else "FAIL",
                    observed=observed,
                    expected=expected
                )

            self.inspect_raster(
                f"InSAR final {year}",
                final,
                year=year
            )

        expected_total = int(
            self.cfg["analysis"][
                "expected_total_insar_epochs"
            ]
        )

        self.add(
            "InSAR",
            "Total epoch count",
            "PASS"
            if total_epochs == expected_total
            else "FAIL",
            observed=total_epochs,
            expected=expected_total
        )

        master = (
            insar_root
            / "SBAS_QC_1395_1403.csv"
        )

        self.exists(
            "InSAR",
            "Master SBAS QC",
            master
        )

    # ================================================================
    # SVI and T baseline QC
    # ================================================================

    def check_baseline_rasters(self):

        svi_root = Path(
            self.cfg["paths"]["svi_root"]
        )

        t_root = Path(
            self.cfg["paths"]["t_root"]
        )

        years = range(
            int(
                self.cfg["analysis"][
                    "jalali_start_year"
                ]
            ),
            int(
                self.cfg["analysis"][
                    "jalali_end_year"
                ]
            ) + 1
        )

        for year in years:

            svi = (
                svi_root
                / f"SVI_{year}_Basic.tif"
            )

            t = (
                t_root
                / f"T_{year}_Rate.tif"
            )

            self.inspect_raster(
                f"SVI Basic {year}",
                svi,
                year=year
            )

            self.inspect_raster(
                f"T Rate {year}",
                t,
                year=year
            )

    # ================================================================
    # Analysis checkpoint QC
    # ================================================================

    def check_analysis_outputs(self):

        if not self.analysis.exists():

            self.add(
                "Analysis",
                "Analysis root",
                "FAIL",
                observed=self.analysis,
                expected="exists"
            )

            return

        # One or more CSV outputs should exist for every stage prefix 01-26
        files = list(
            self.analysis.glob("*.csv")
        )

        names = [
            f.name
            for f in files
        ]

        missing_prefixes = []

        for n in range(1, 27):

            prefix = f"{n:02d}_"

            matches = [
                name
                for name in names
                if name.startswith(prefix)
            ]

            if not matches:
                missing_prefixes.append(
                    f"{n:02d}"
                )

        self.add(
            "Analysis",
            "Stage prefixes 01-26",
            "PASS"
            if not missing_prefixes
            else "WARN",
            observed=(
                "01-26 complete"
                if not missing_prefixes
                else (
                    "Missing: "
                    + ",".join(
                        missing_prefixes
                    )
                )
            ),
            expected="at least one CSV for every stage"
        )

        # Critical reproducibility checkpoints
        required = [
            "01_Groundwater_Monthly_41Wells.csv",
            "02_InSAR_Epochs_41Wells.csv",
            "03_InSAR_Monthly_41Wells.csv",
            "04_GW_InSAR_Monthly_Pairs.csv",
            "05_Piezometer_Coverage_QC.csv",
            "06_Overlap_Increment_QC.csv",
            "07_Canonical_Monthly_Pairs_37Wells.csv",

            "08_Trend_Seasonality_Autocorrelation_ByWell.csv",

            "09_CCF_Lag0_12_ByWell.csv",
            "10_Lag_Significance_MaxStat_FDR.csv",

            "11_SBAS_Stack_Stitching_QC.csv",

            "12_Stitched_Cumulative_Epochs_1396_1403_37Wells.csv",
            "12_Stitching_Boundary_QC_37Wells.csv",

            "13_Cumulative_GW_InSAR_Monthly_37Wells.csv",
            "13_Cumulative_Relationship_Metrics_ByWell.csv",

            "14_Monthly_HydroMechanical_States.csv",
            "14_Recovery_Continued_Subsidence_ByWell.csv",

            "15_CCF_Lag0_36_ByWell.csv",
            "15_CCF_Lag0_36_Summary.csv",

            "16_Binned_DistributedLag_Panel.csv",
            "16_Binned_DistributedLag_Summary.csv",

            "17_PostMinimumHead_ResidualCompaction.csv",
            "18_ResidualCompaction_Strength_QC.csv",
            "19_ResidualCompaction_WithinStack_Validation.csv",

            "20_ResidualCompaction_Spatial_Candidates.csv",

            "21_StrongResidual_SpatialClustering_QC.csv",

            "22_ResidualCompaction_GroupComparison.csv",

            "23_EventAligned_GroupSummary.csv",

            "24_NonCircular_FixedHorizon_Summary.csv",

            "25_NonCircular_ExtendedHorizons_12to84_Summary.csv",

            "26_FirstRecovery0p5_Event_QC.csv",
            "26_PostRecovery_EventAligned_Summary.csv",
        ]

        missing = []

        for name in required:

            path = (
                self.analysis
                / name
            )

            if not path.exists():
                missing.append(name)

        self.add(
            "Analysis",
            "Critical checkpoint files",
            "PASS"
            if not missing
            else "WARN",
            observed=(
                len(required)
                - len(missing)
            ),
            expected=len(required),
            details=(
                ""
                if not missing
                else (
                    "Missing: "
                    + " | ".join(missing)
                )
            )
        )

        # ------------------------------------------------------------
        # Structural QC on key tables
        # No scientific result is hard-coded here.
        # ------------------------------------------------------------

        key_tables = {
            "07_Canonical_Monthly_Pairs_37Wells.csv":
                ["Piezometer", "JYear", "JMonth"],

            "13_Cumulative_GW_InSAR_Monthly_37Wells.csv":
                ["Piezometer", "MonthIndex"],

            "17_PostMinimumHead_ResidualCompaction.csv":
                ["Piezometer"],

            "20_ResidualCompaction_Spatial_Candidates.csv":
                ["Piezometer", "utmx", "utmy"],

            "26_FirstRecovery0p5_Event_QC.csv":
                ["Piezometer"],

            "26_PostRecovery_EventAligned_Summary.csv":
                ["Cohort", "Horizon_months"],
        }

        for name, key in key_tables.items():

            path = (
                self.analysis
                / name
            )

            if not path.exists():
                continue

            try:

                df = pd.read_csv(path)

                missing_cols = [
                    c
                    for c in key
                    if c not in df.columns
                ]

                if missing_cols:

                    self.add(
                        "Analysis",
                        f"{name}: key columns",
                        "FAIL",
                        observed=missing_cols,
                        expected=key
                    )

                    continue

                duplicates = int(
                    df.duplicated(key).sum()
                )

                self.add(
                    "Analysis",
                    f"{name}: duplicate key",
                    "PASS"
                    if duplicates == 0
                    else "FAIL",
                    observed=duplicates,
                    expected=0,
                    details=(
                        " + ".join(key)
                    )
                )

            except Exception as exc:

                self.add(
                    "Analysis",
                    f"{name}: readable",
                    "FAIL",
                    details=repr(exc)
                )

    # ================================================================
    # Git/environment manifest
    # ================================================================

    def git_info(self):

        def run_git(args):

            try:

                result = subprocess.run(
                    ["git"] + args,
                    cwd=self.repo,
                    check=True,
                    capture_output=True,
                    text=True
                )

                return (
                    result.stdout.strip()
                )

            except Exception:
                return None

        commit = run_git(
            ["rev-parse", "HEAD"]
        )

        branch = run_git(
            ["branch", "--show-current"]
        )

        status = run_git(
            ["status", "--porcelain"]
        )

        return {
            "commit": commit,
            "branch": branch,
            "dirty": (
                bool(status)
                if status is not None
                else None
            ),
        }

    # ================================================================
    # Run all QC
    # ================================================================

    def run(self):

        self.check_structure()
        self.check_piezometers()
        self.check_insar()
        self.check_baseline_rasters()
        self.check_analysis_outputs()

        return self.records

    # ================================================================
    # Save outputs
    # ================================================================

    def save(self):

        qdir = qc_root(
            self.cfg
        )

        mdir = manifest_root(
            self.cfg
        )

        qc_df = pd.DataFrame(
            self.records
        )

        raster_df = pd.DataFrame(
            self.rasters
        )

        qc_csv = (
            qdir
            / "00_Project_QC.csv"
        )

        raster_csv = (
            qdir
            / "00_Raster_Inventory.csv"
        )

        manifest_json = (
            mdir
            / "00_Project_QC_Manifest.json"
        )

        qc_df.to_csv(
            qc_csv,
            index=False
        )

        raster_df.to_csv(
            raster_csv,
            index=False
        )

        manifest = {
            "created_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            "project_config":
                self.cfg["_config_path"],

            "python_version":
                sys.version,

            "platform":
                platform.platform(),

            "pandas_version":
                pd.__version__,

            "numpy_version":
                np.__version__,

            "rasterio_version":
                rasterio.__version__,

            "git":
                self.git_info(),

            "qc_counts":
                qc_df[
                    "Status"
                ]
                .value_counts()
                .to_dict(),
        }

        with manifest_json.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False
            )

        return (
            qc_df,
            raster_df,
            qc_csv,
            raster_csv,
            manifest_json
        )


def main():

    cfg = load_config()

    qc = ProjectQC(cfg)

    qc.run()

    (
        qc_df,
        raster_df,
        qc_csv,
        raster_csv,
        manifest_json
    ) = qc.save()

    print("=" * 100)
    print("CENTRAL PROJECT QC")
    print("=" * 100)

    counts = (
        qc_df["Status"]
        .value_counts()
    )

    for status in [
        "PASS",
        "WARN",
        "FAIL"
    ]:

        print(
            f"{status:5s}:",
            int(
                counts.get(
                    status,
                    0
                )
            )
        )

    problems = qc_df[
        qc_df["Status"].isin(
            ["WARN", "FAIL"]
        )
    ]

    print("\n" + "=" * 100)
    print("WARNINGS / FAILURES")
    print("=" * 100)

    if len(problems) == 0:

        print(
            "No warnings or failures."
        )

    else:

        print(
            problems[
                [
                    "Category",
                    "Item",
                    "Status",
                    "Observed",
                    "Expected",
                    "Details",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n" + "=" * 100)
    print("KEY SYSTEM FACTS")
    print("=" * 100)

    facts = qc_df[
        qc_df["Item"].isin(
            [
                "Raw monthly well count",
                "Official T well count",
                "1402 official-well missing values",
                "Total epoch count",
                "Stage prefixes 01-26",
                "Critical checkpoint files",
            ]
        )
    ]

    print(
        facts[
            [
                "Item",
                "Status",
                "Observed",
                "Expected",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOUTPUTS:")
    print(qc_csv)
    print(raster_csv)
    print(manifest_json)

    print("=" * 100)

    if (
        qc_df["Status"] == "FAIL"
    ).any():

        print(
            "PROJECT QC RESULT: FAIL"
        )

        raise SystemExit(1)

    print(
        "PROJECT QC RESULT: PASS"
    )


if __name__ == "__main__":
    main()
