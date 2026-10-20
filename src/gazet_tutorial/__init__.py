"""gazet-tutorial: a simplified, single-country reproduction of gazet for teaching.

The learner-facing logic (templates, generation loop, prompts, training config,
metrics) lives in the notebooks. This package holds the boilerplate they import.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

__version__ = "0.2.0"

COUNTRY = "IN"
COUNTRY_NAME = "India"
# (min_lon, min_lat, max_lon, max_lat): mainland India plus both island groups.
# Used only to prune Overture row groups by their bbox statistics.
INDIA_BBOX = (68.0, 6.4, 97.6, 35.8)
COUNTRY_BBOX = INDIA_BBOX
# Symbolic table name used in every SQL string the model sees. Rewritten to the
# real parquet path at execution time (see sqlutil.rewrite_data_paths).
TABLE = "divisions_area"
SUBTYPES = ("country", "region", "county")
RANDOM_SEED = 108

# Code, notebooks and the bundled artifacts all come from this repo: `pip install git+...`
# installs the package together with artifacts/ (see [tool.hatch...force-include]).
GITHUB_REPO = "developmentseed/gazet-tutorial"


def data_dir() -> Path:
    """Scratch directory for downloaded geodata and generated outputs.

    Colab: /content/gazet-tutorial-work. Elsewhere: $GAZET_TUTORIAL_DATA or ./work.
    """
    env = os.environ.get("GAZET_TUTORIAL_DATA")
    if env:
        p = Path(env)
    elif Path("/content").is_dir():
        p = Path("/content/gazet-tutorial-work")
    else:
        p = Path.cwd() / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_geodata_path() -> Path:
    return data_dir() / f"divisions_area_{COUNTRY.lower()}.parquet"


def artifacts_dir() -> Path:
    """Directory holding the pre-built fallback artifacts (dataset/, benchmark/, ...).

    Resolution order: $GAZET_TUTORIAL_ARTIFACTS, a repo checkout's ./artifacts
    (when running from the repo root), then the copy bundled inside the wheel.
    """
    env = os.environ.get("GAZET_TUTORIAL_ARTIFACTS")
    if env and Path(env).is_dir():
        return Path(env)
    repo_local = Path(__file__).resolve().parents[2] / "artifacts"
    if (repo_local / "dataset").is_dir():
        return repo_local
    bundled = Path(str(resources.files("gazet_tutorial") / "artifacts"))
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        "Could not locate the bundled artifacts; set GAZET_TUTORIAL_ARTIFACTS to a "
        "directory containing dataset/ and benchmark/."
    )
