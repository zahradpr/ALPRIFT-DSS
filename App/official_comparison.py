
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


OFFICIAL_YEARS = {
    1395,
    1396,
}


CLASS_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High",
}


CLASS_THRESHOLDS = {
    1: "[24, 78)",
    2: "[78, 133)",
    3: "[133, 186)",
    4: "[186, 240]",
}


def has_official_comparison(year):
    """
    Return True when the official Reference2018 annual
    ALPRIFT-InSAR comparison is available.
    """

    return int(year) in OFFICIAL_YEARS


def resolve_official_comparison_paths(
    year,
    root=DEFAULT_ROOT,
):
    """
    Resolve the official Reference2018 comparison products.
    """

    root = Path(root)
    year = int(year)

    if year not in OFFICIAL_YEARS:
        raise ValueError(
            f"Official annual comparison is not available "
            f"for model year {year}."
        )

    return {

        "svi": (
            root
            / "Baseline"
            / "SVI"
            / f"SVI_{year}_Reference2018.tif"
        ),

        "rank": (
            root
            / "Baseline"
            / "Rank"
            / f"SVI_{year}_Reference2018_Rank.tif"
        ),

        "pixel_pairs": (
            root
            / "Baseline"
            / "SVI"
            / f"ALPRIFT_InSAR_PixelPairs_{year}_Official.csv"
        ),

        "class_stats": (
            root
            / "Baseline"
            / "SVI"
            / f"ALPRIFT_InSAR_ClassStats_{year}_Official.csv"
        ),

        "class_comparison": (
            root
            / "Baseline"
            / "SVI"
            / f"ALPRIFT_InSAR_ClassComparison_{year}_Official.csv"
        ),

        "metrics": (
            root
            / "Baseline"
            / "SVI"
            / f"ALPRIFT_InSAR_{year}_Official_Metrics.csv"
        ),

        "insar": (
            root
            / "SourceData"
            / "InSAR"
            / str(year)
            / f"Shabestar_{year}_Subsidence_100m_FINAL.tif"
        ),
    }


def load_official_pixel_pairs(
    year,
    root=DEFAULT_ROOT,
):
    """
    Load the official raw annual ALPRIFT-InSAR pixel pairs.
    """

    paths = resolve_official_comparison_paths(
        year,
        root=root,
    )

    path = paths["pixel_pairs"]

    if not path.exists():
        raise FileNotFoundError(
            f"Official pixel-pair file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "Pixel_Row",
        "Pixel_Col",
        "X_UTM",
        "Y_UTM",
        "SVI",
        "InSAR_mm",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing official pixel-pair columns: "
            + ", ".join(missing)
        )

    return df


def load_official_class_stats(
    year,
    root=DEFAULT_ROOT,
):
    """
    Load official InSAR statistics by Reference2018 class.
    """

    paths = resolve_official_comparison_paths(
        year,
        root=root,
    )

    path = paths["class_stats"]

    if not path.exists():
        raise FileNotFoundError(
            f"Official class-statistics file not found: {path}"
        )

    return pd.read_csv(path)


def load_official_class_comparison(
    year,
    root=DEFAULT_ROOT,
):
    """
    Load adjacent-class comparison statistics.
    """

    paths = resolve_official_comparison_paths(
        year,
        root=root,
    )

    path = paths["class_comparison"]

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def compute_official_metrics(
    year,
    root=DEFAULT_ROOT,
):
    """
    Compute the official annual agreement metrics directly
    from the official raw annual pixel-pair CSV.

    No ML model, temporal holdout, or optimized weights
    are used here.
    """

    year = int(year)

    df = load_official_pixel_pairs(
        year,
        root=root,
    )

    svi = df["SVI"].to_numpy(
        dtype=float
    )

    insar = df["InSAR_mm"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(svi)
        &
        np.isfinite(insar)
    )

    svi = svi[valid]
    insar = insar[valid]

    pearson_r, pearson_p = pearsonr(
        svi,
        insar,
    )

    spearman_rho, spearman_p = spearmanr(
        svi,
        insar,
    )

    common_pixels = int(
        len(svi)
    )

    # Official grid = 100 m x 100 m.
    common_area_km2 = (
        common_pixels
        * 0.01
    )

    return {

        "year": year,

        "common_pixels":
            common_pixels,

        "common_area_km2":
            float(common_area_km2),

        "svi_min":
            float(np.min(svi)),

        "svi_max":
            float(np.max(svi)),

        "svi_mean":
            float(np.mean(svi)),

        "svi_std":
            float(np.std(svi, ddof=1)),

        "insar_min_mm":
            float(np.min(insar)),

        "insar_max_mm":
            float(np.max(insar)),

        "insar_mean_mm":
            float(np.mean(insar)),

        "insar_std_mm":
            float(np.std(insar, ddof=1)),

        "pearson_r":
            float(pearson_r),

        "pearson_r2":
            float(pearson_r ** 2),

        "pearson_p":
            float(pearson_p),

        "spearman_rho":
            float(spearman_rho),

        "spearman_p":
            float(spearman_p),
    }


def load_official_comparison(
    year,
    root=DEFAULT_ROOT,
):
    """
    Load all official annual comparison products for the
    dashboard in one call.
    """

    paths = resolve_official_comparison_paths(
        year,
        root=root,
    )

    required_paths = [
        "svi",
        "rank",
        "pixel_pairs",
        "class_stats",
        "insar",
    ]

    missing = [
        str(paths[key])
        for key in required_paths
        if not paths[key].exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing official comparison product(s):\n"
            + "\n".join(missing)
        )

    pixel_pairs = load_official_pixel_pairs(
        year,
        root=root,
    )

    class_stats = load_official_class_stats(
        year,
        root=root,
    )

    class_comparison = (
        load_official_class_comparison(
            year,
            root=root,
        )
    )

    metrics = compute_official_metrics(
        year,
        root=root,
    )

    class_sum = int(
        class_stats["Cells"].sum()
    )

    if class_sum != metrics["common_pixels"]:
        raise ValueError(
            "Official class-statistics sample does not "
            "match official pixel-pair sample: "
            f"{class_sum:,} != "
            f"{metrics['common_pixels']:,}"
        )

    return {

        "paths":
            paths,

        "pixel_pairs":
            pixel_pairs,

        "class_stats":
            class_stats,

        "class_comparison":
            class_comparison,

        "metrics":
            metrics,
    }
