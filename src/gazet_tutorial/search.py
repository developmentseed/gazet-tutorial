"""Fuzzy place-name search against the divisions table.

Port of gazet's src/gazet/search.py (simple_fuzzy_search, fetch_by_id) for a
single table, plus the candidates-CSV formatter from export_training_data.py.
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import pandas as pd

# Columns the SQL model sees for each candidate. `source`, `country` and
# `admin_level` from gazet are dropped: one table, one country, subtype suffices.
CANDIDATE_COLUMNS = ["id", "name", "subtype", "region"]

_SEARCH_SQL = """
WITH ranked AS (
    SELECT
        id,
        names."primary" AS name,
        subtype,
        country,
        region,
        admin_level,
        jaro_winkler_similarity(lower(names."primary"), lower(?)) AS similarity,
        contains(lower(names."primary"), lower(?)) AS is_substring_match
    FROM read_parquet(?)
    WHERE names."primary" IS NOT NULL
      AND trim(names."primary") != ''
      AND geometry IS NOT NULL
      AND id != ?
)
SELECT * FROM ranked
ORDER BY is_substring_match DESC, similarity DESC, admin_level ASC, id
LIMIT ?
"""


def fuzzy_search(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    name: str,
    limit: int = 10,
    exclude_id: str | None = None,
) -> pd.DataFrame:
    """Jaro-Winkler search ranked with exact-substring matches first.

    Substring containment ("Loja" in "Loja Province") is a stronger signal
    than a slightly higher edit-distance score for a near-miss ("Lodja").
    """
    return con.execute(_SEARCH_SQL, [name, name, str(path), exclude_id or "", limit]).fetchdf()


def fetch_by_id(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    feature_id: str,
    geometry: bool = False,
) -> pd.DataFrame:
    geom = ", ST_AsGeoJSON(geometry) AS geometry" if geometry else ""
    return con.execute(
        f"""
        SELECT id, names."primary" AS name, subtype, country, region, admin_level{geom}
        FROM read_parquet(?) WHERE id = ? LIMIT 1
        """,
        [str(path), feature_id],
    ).fetchdf()


def candidates_csv(candidates: pd.DataFrame | list[dict]) -> str:
    """Render candidates as the CSV block the model is trained on."""
    df = pd.DataFrame(candidates) if not isinstance(candidates, pd.DataFrame) else candidates
    cols = [c for c in CANDIDATE_COLUMNS if c in df.columns]
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().strip()


def per_place_limit(num_places: int) -> int:
    """Candidates to fetch per extracted place so the prompt stays ~10 rows."""
    if num_places <= 1:
        return 10
    if num_places == 2:
        return 5
    return 4


def merge_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-place results, drop duplicate ids, rank like gazet's api.py."""
    if not frames:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    return (
        df.sort_values(["is_substring_match", "similarity", "admin_level"],
                       ascending=[False, False, True])
        .reset_index(drop=True)
    )
