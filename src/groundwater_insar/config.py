from __future__ import annotations

from pathlib import Path
import json
import os


def package_repository_root() -> Path:
    """
    Infer repository root from:
    <repo>/src/groundwater_insar/config.py
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """
    Configuration discovery order:

    1) ALPRIFT_DSS_CONFIG environment variable
    2) Config/project_config.json beside this repository
    3) config/project_config.json beside this repository
       (legacy compatibility)
    4) legacy workspace configuration
    """

    env = os.environ.get("ALPRIFT_DSS_CONFIG")

    if env:
        return Path(env)

    repo = package_repository_root()

    candidates = [
        repo / "Config" / "project_config.json",
        repo / "config" / "project_config.json",

        Path(
            "/mnt/d/ALPRIFT_DSS/config/project_config.json"
        ),
        Path(
            "/mnt/d/ALPRIFT_DSS/Config/project_config.json"
        ),
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def load_config(
    config_path: str | Path | None = None
) -> dict:

    path = (
        Path(config_path)
        if config_path is not None
        else default_config_path()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Project configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(
        path.resolve()
    )

    return cfg


def workspace_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"].get(
            "workspace_root",
            cfg["paths"]["project_root"]
        )
    )


def project_root(cfg: dict) -> Path:
    """
    Backward-compatible alias.
    Project/data root means workspace root.
    """
    return workspace_root(cfg)


def repository_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"]["repository_root"]
    )


def analysis_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"]["analysis_root"]
    )


def output_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"]["output_root"]
    )


def reference_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"]["reference_root"]
    )


def notebook_root(cfg: dict) -> Path:

    return Path(
        cfg["paths"]["notebook_root"]
    )


def qc_root(cfg: dict) -> Path:

    path = (
        output_root(cfg)
        / "qc"
    )

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


def manifest_root(cfg: dict) -> Path:

    path = (
        output_root(cfg)
        / "manifests"
    )

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path
