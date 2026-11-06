from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import re

import numpy as np
import pandas as pd
import rasterio

from .config import (
    load_config,
    reference_root,
)


RASTER_RE = re.compile(
    r"^Subsidence_Cumulative_"
    r"(?P<date>\d{8})_"
    r"ref_(?P<ref>\d{8})_"
    r"100m_ALPRIFT\.tif$"
)


# =====================================================================
# Gregorian -> Jalali
# =====================================================================

def gregorian_to_jalali(
    gy: int,
    gm: int,
    gd: int
) -> tuple[int, int, int]:

    gdm = [
        0, 31, 59, 90, 120, 151,
        181, 212, 243, 273, 304, 334
    ]

    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621

    gy2 = gy + 1 if gm > 2 else gy

    days = (
        365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        - 80
        + gd
        + gdm[gm - 1]
    )

    jy += 33 * (days // 12053)
    days %= 12053

    jy += 4 * (days // 1461)
    days %= 1461

    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365

    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30

    return int(jy), int(jm), int(jd)


# =====================================================================
# Frozen reference helpers
# =====================================================================

def frozen_file_for_stage(
    cfg: dict,
    stage: int
) -> Path:

    version = cfg["analysis"][
        "frozen_reference_version"
    ]

    root = (
        reference_root(cfg)
        / version
    )

    manifest_path = (
        root
        / "baseline_manifest.json"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    files = [
        x["filename"]
        for x in manifest["files"]
        if int(x["stage"]) == stage
    ]

    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage {stage:02d} "
            f"reference CSV; found {files}"
        )

    return root / "csv" / files[0]


def official_wells_from_stage01(
    cfg: dict
) -> list[str]:

    p = frozen_file_for_stage(
        cfg,
        1
    )

    df = pd.read_csv(p)

    if "Piezometer" not in df.columns:
        raise RuntimeError(
            "Stage 01 reference has no Piezometer column."
        )

    # Preserve original Stage-01 order.
    wells = (
        df["Piezometer"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    expected = int(
        cfg["analysis"]["official_wells"]
    )

    if len(wells) != expected:
        raise RuntimeError(
            f"Expected {expected} official wells; "
            f"found {len(wells)}."
        )

    return wells


# =====================================================================
# Coordinate source
# =====================================================================

def load_official_coordinates(
    cfg: dict,
    wells: list[str]
) -> pd.DataFrame:

    csv_path = Path(
        cfg["paths"]["piezometer_monthly"]
    )

    xlsx = (
        csv_path.parent
        / "Piezometer_Monthly_Head.xlsx"
    )

    if not xlsx.exists():
        raise FileNotFoundError(xlsx)

    coords = pd.read_excel(
        xlsx,
        sheet_name="Coordinates"
    )

    required = {
        "Piezometer",
        "utmx",
        "utmy"
    }

    missing = required - set(coords.columns)

    if missing:
        raise RuntimeError(
            f"Coordinate columns missing: {sorted(missing)}"
        )

    duplicate_ids = int(
        coords["Piezometer"]
        .astype(str)
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise RuntimeError(
            f"Duplicate coordinate IDs: {duplicate_ids}"
        )

    coords["Piezometer"] = (
        coords["Piezometer"].astype(str)
    )

    official = (
        pd.DataFrame({
            "Piezometer": wells
        })
        .merge(
            coords[
                [
                    "Piezometer",
                    "utmx",
                    "utmy"
                ]
            ],
            on="Piezometer",
            how="left",
            validate="one_to_one"
        )
    )

    missing_coord = int(
        official[
            ["utmx", "utmy"]
        ].isna().any(axis=1).sum()
    )

    if missing_coord:
        bad = official.loc[
            official[
                ["utmx", "utmy"]
            ].isna().any(axis=1),
            "Piezometer"
        ].tolist()

        raise RuntimeError(
            f"Missing coordinates for: {bad}"
        )

    if len(official) != 41:
        raise RuntimeError(
            f"Coordinate match expected 41; "
            f"found {len(official)}."
        )

    return official


# =====================================================================
# Raster inventory
# =====================================================================

def collect_epoch_rasters(
    cfg: dict
) -> list[dict]:

    root = Path(
        cfg["paths"]["insar_root"]
    )

    expected_map = {
        int(k): int(v)
        for k, v in cfg["analysis"][
            "expected_insar_epochs_by_year"
        ].items()
    }

    records = []

    for stack_year in range(
        int(cfg["analysis"]["jalali_start_year"]),
        int(cfg["analysis"]["jalali_end_year"]) + 1
    ):

        tsdir = (
            root
            / str(stack_year)
            / "timeseries_Subsidence_100m_ALPRIFT"
        )

        files = sorted(
            tsdir.glob(
                "Subsidence_Cumulative_*_100m_ALPRIFT.tif"
            )
        )

        # Keep ONLY valid cumulative epoch filenames.
        parsed = []

        for path in files:

            match = RASTER_RE.match(
                path.name
            )

            if match is None:
                continue

            date8 = match.group("date")

            parsed.append({
                "StackYear":
                    stack_year,

                "path":
                    path,

                "date8":
                    date8,

                "date":
                    datetime.strptime(
                        date8,
                        "%Y%m%d"
                    ).date(),
            })

        expected = expected_map[
            stack_year
        ]

        if len(parsed) != expected:
            raise RuntimeError(
                f"{stack_year}: expected {expected} "
                f"epochs, found {len(parsed)}."
            )

        records.extend(parsed)

    if len(records) != int(
        cfg["analysis"]["expected_total_insar_epochs"]
    ):
        raise RuntimeError(
            "Total epoch count is not 143."
        )

    return records


# =====================================================================
# Reconstruction
# =====================================================================

def reconstruct_stage02(
    cfg: dict | None = None
):

    if cfg is None:
        cfg = load_config()

    reference_path = frozen_file_for_stage(
        cfg,
        2
    )

    reference = pd.read_csv(
        reference_path
    )

    expected_columns = list(
        reference.columns
    )

    required_schema = [
        "StackYear",
        "Piezometer",
        "AcquisitionDate",
        "JYear",
        "JMonth",
        "JDay",
        "MonthIndex",
        "InSAR_mm",
        "Raster",
    ]

    if expected_columns != required_schema:
        raise RuntimeError(
            "Unexpected Stage-02 frozen schema:\n"
            f"{expected_columns}"
        )

    wells = official_wells_from_stage01(
        cfg
    )

    coords = load_official_coordinates(
        cfg,
        wells
    )

    epochs = collect_epoch_rasters(
        cfg
    )

    # ----------------------------------------------------------
    # QC before sampling
    # ----------------------------------------------------------

    if len(wells) != 41:
        raise RuntimeError(
            "Official well QC failed."
        )

    if len(epochs) != 143:
        raise RuntimeError(
            "Epoch QC failed."
        )

    expected_rows = (
        len(wells)
        * len(epochs)
    )

    if expected_rows != 5863:
        raise RuntimeError(
            f"Expected potential samples 5863; "
            f"calculated {expected_rows}."
        )

    xy = list(
        zip(
            coords["utmx"].astype(float),
            coords["utmy"].astype(float)
        )
    )

    rows = []

    # ----------------------------------------------------------
    # Sample every epoch at all 41 official piezometers.
    # ----------------------------------------------------------

    for epoch in epochs:

        path = epoch["path"]

        with rasterio.open(path) as src:

            crs = (
                src.crs.to_string()
                if src.crs
                else None
            )

            if crs != cfg["project"]["crs"]:
                raise RuntimeError(
                    f"CRS mismatch in {path.name}: {crs}"
                )

            if not (
                np.isclose(
                    abs(src.transform.a),
                    cfg["project"]["grid_resolution_m"]
                )
                and
                np.isclose(
                    abs(src.transform.e),
                    cfg["project"]["grid_resolution_m"]
                )
            ):
                raise RuntimeError(
                    f"Resolution mismatch: {path.name}"
                )

            samples = list(
                src.sample(
                    xy,
                    indexes=1
                )
            )

        if len(samples) != 41:
            raise RuntimeError(
                f"Sampling count error for {path.name}"
            )

        gy = epoch["date"].year
        gm = epoch["date"].month
        gd = epoch["date"].day

        jy, jm, jd = gregorian_to_jalali(
            gy,
            gm,
            gd
        )

        month_index = (
            jy * 12
            + jm
            - 1
        )

        acquisition = (
            epoch["date"].isoformat()
        )

        for well, sample in zip(
            wells,
            samples
        ):

            value = float(
                sample[0]
            )

            # Raster NoData handling.
            #
            # Most missing samples are NaN, but the 1395
            # stack also contains the float32 sentinel
            # approximately -3.402823e+38. Both represent
            # invalid InSAR observations.
            if (
                not np.isfinite(value)
                or value < -1e30
            ):
                value = np.nan

            rows.append({
                "StackYear":
                    int(epoch["StackYear"]),

                "Piezometer":
                    well,

                "AcquisitionDate":
                    acquisition,

                "JYear":
                    jy,

                "JMonth":
                    jm,

                "JDay":
                    jd,

                "MonthIndex":
                    month_index,

                "InSAR_mm":
                    value,

                "Raster":
                    path.name,
            })

    regenerated = pd.DataFrame(
        rows,
        columns=expected_columns
    )

    # ----------------------------------------------------------
    # Structural QC
    # ----------------------------------------------------------

    if len(regenerated) != 5863:
        raise RuntimeError(
            f"Stage 02 row count={len(regenerated)}, "
            "expected 5863."
        )

    if regenerated[
        "Piezometer"
    ].nunique() != 41:
        raise RuntimeError(
            "Stage 02 well-count QC failed."
        )

    if regenerated[
        [
            "StackYear",
            "Piezometer",
            "AcquisitionDate"
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate Stage-02 epoch/well records."
        )

    # Gregorian->Jalali anchor QC from known first epoch
    first = regenerated.iloc[0]

    if (
        first["AcquisitionDate"] != "2015-12-29"
        or int(first["JYear"]) != 1394
        or int(first["JMonth"]) != 10
        or int(first["JDay"]) != 8
    ):
        raise RuntimeError(
            "Gregorian/Jalali conversion anchor failed."
        )

    # ----------------------------------------------------------
    # Save regenerated output
    # ----------------------------------------------------------

    out_dir = (
        Path(
            cfg["paths"]["regenerated_root"]
        )
        / "stage02"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out = (
        out_dir
        / reference_path.name
    )

    regenerated.to_csv(
        out,
        index=False
    )

    return (
        regenerated,
        out,
        reference_path,
        coords,
        epochs
    )


def main():

    cfg = load_config()

    (
        result,
        out,
        ref,
        coords,
        epochs
    ) = reconstruct_stage02(cfg)

    print("=" * 100)
    print("STAGE 02 — INSAR EPOCH SAMPLING")
    print("=" * 100)

    print("Reference             :", ref)
    print("Generated             :", out)
    print("Official wells        :", len(coords))
    print("Epochs                :", len(epochs))
    print("Rows                  :", len(result))
    print(
        "Valid InSAR values    :",
        int(result["InSAR_mm"].notna().sum())
    )
    print(
        "Missing InSAR values  :",
        int(result["InSAR_mm"].isna().sum())
    )

    print("\nEpochs by StackYear:")
    print(
        result[
            ["StackYear", "AcquisitionDate"]
        ]
        .drop_duplicates()
        ["StackYear"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("STAGE 02 RECONSTRUCTION: PASS")


if __name__ == "__main__":
    main()
