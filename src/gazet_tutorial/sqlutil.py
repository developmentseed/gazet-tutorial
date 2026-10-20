"""SQL helpers: symbolic-path rewriting, fence stripping, formatting, execution.

Ported from gazet's src/gazet/sql.py (_rewrite_data_paths, _strip_fences),
src/gazet/lm.py (_postprocess_sql) and dataset/scripts/export_training_data.py
(_format_sql), reduced to a single table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import sqlparse

from . import TABLE

SYMBOLIC_TABLE = f"read_parquet('{TABLE}')"

# Any quoted path that mentions the table name, including hallucinated variants
# like read_parquet('/data/divisions_area.parquet') or read_parquet("divisions_area").
_TABLE_REF = re.compile(r"read_parquet\(\s*['\"][^'\"]*divisions?_area[^'\"]*['\"]\s*\)")
# The bare table name used as if it were a registered table ("FROM divisions_area d"),
# which zero-shot models write often. Not matched inside a path or an identifier.
_BARE_TABLE = re.compile(r"(?<![\w'\"./\\-])divisions_area(?![\w'\"./\\-])")


def rewrite_data_paths(sql: str, path: str | Path) -> str:
    """Replace every symbolic table reference with the real parquet path.

    Handles ``read_parquet('divisions_area')`` (the training format), quoted
    path variants, and the bare name ``divisions_area`` used as a table.
    """
    real = f"read_parquet('{path}')"
    sql = _TABLE_REF.sub(real, sql)
    return _BARE_TABLE.sub(real, sql)


def to_symbolic(sql: str, path: str | Path) -> str:
    """Inverse of rewrite_data_paths: make SQL path-independent before storing it."""
    return sql.replace(str(path), TABLE)


def strip_fences(text: str | None) -> str:
    """Remove ```sql fences, a stray <think> block, and surrounding whitespace."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "```sql" in cleaned:
        cleaned = cleaned.split("```sql", 1)[1]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
        cleaned = cleaned.removeprefix("json").removeprefix("SQL").removeprefix("sql")
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    return cleaned.strip()


def format_sql(sql: str) -> str:
    """Pretty-print SQL so the model learns one consistent style."""
    return sqlparse.format(sql, reindent=True, keyword_case="upper", indent_width=4).strip()


def normalize_ws(sql: str) -> str:
    """Whitespace-insensitive form used for exact-match comparison."""
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")


def execute(con: duckdb.DuckDBPyConnection, sql: str, path: str | Path) -> pd.DataFrame:
    """Rewrite symbolic paths and run the query. Raises on SQL errors."""
    return con.execute(rewrite_data_paths(sql, path)).fetchdf()


def result_ids(df: pd.DataFrame, ordered: bool = False) -> list[str]:
    """Feature ids returned by a query; sorted set by default, order-preserving if ordered."""
    if "id" not in df.columns:
        return []
    ids = [str(x) for x in df["id"].tolist()]
    return ids if ordered else sorted(set(ids))


def to_feature_collection(df: pd.DataFrame, geometry_col: str = "geometry") -> dict[str, Any]:
    """Turn a result DataFrame (GeoJSON string geometry) into a GeoJSON FeatureCollection."""
    features = []
    for rec in df.to_dict(orient="records"):
        geom = rec.pop(geometry_col, None)
        if isinstance(geom, (bytes, bytearray)):
            raise TypeError("geometry is WKB; select ST_AsGeoJSON(geometry) instead")
        if isinstance(geom, str):
            geom = json.loads(geom)
        props = {k: (None if pd.isna(v) else v) if not isinstance(v, (list, dict)) else v
                 for k, v in rec.items()}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": features}
