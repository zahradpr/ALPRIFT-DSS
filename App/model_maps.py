
from pathlib import Path


STATIC_LAYERS = {
    "A",
    "L",
    "P",
    "R",
    "I",
    "F",
}


def resolve_model_raster_path(
    layer_code,
    year,
    root,
):
    """
    Return the raster used by the real ALPRIFT model.

    A, L, P, R, I, F are static baseline layers.
    T is annual and depends on the selected model year.
    """

    root = Path(root)

    layer_code = str(
        layer_code
    ).upper().strip()

    year = int(year)

    if layer_code in STATIC_LAYERS:

        return (
            root
            / "Baseline"
            / layer_code
            / f"{layer_code}_Baseline.tif"
        )

    if layer_code == "T":

        return (
            root
            / "Baseline"
            / "T"
            / f"T_{year}_Rate.tif"
        )

    raise ValueError(
        f"Unknown ALPRIFT layer: {layer_code}"
    )



def load_model_raster(path):
    """
    Load an ALPRIFT raster safely for visualization.

    NoData cells are converted to NaN so they do not
    contaminate plotting or color normalization.
    """

    import numpy as np
    import rasterio

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found: {path}"
        )

    with rasterio.open(path) as src:

        masked = src.read(
            1,
            masked=True
        )

        data = np.asarray(
            masked.astype(
                np.float64
            ).filled(
                np.nan
            ),
            dtype=np.float64
        )

        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return {
        "data": data,
        "bounds": bounds,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "path": path,
    }



def resolve_model_output_paths(
    year,
    root,
):
    """
    Return the real annual ALPRIFT output rasters
    for the selected model year.
    """

    root = Path(root)
    year = int(year)

    return {
        "svi": (
            root
            / "Baseline"
            / "SVI"
            / f"SVI_{year}_Basic.tif"
        ),
        "rank": (
            root
            / "Baseline"
            / "Rank"
            / f"SVI_{year}_Rank.tif"
        ),
    }
