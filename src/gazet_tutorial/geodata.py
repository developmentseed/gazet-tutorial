"""Overture divisions for one country: download, clean, and spatial relations.

Ports gazet's dataset/scripts/normalize_geodata.py (WKB round-trip, subtype and
is_land filter) and dataset/scripts/build_relations.py (adjacency and
containment self-joins), reduced to a single table and a single country, plus
the small cleaning steps the India data needs.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from . import COUNTRY, COUNTRY_BBOX, SUBTYPES, default_geodata_path

OVERTURE_RELEASES = "s3://overturemaps-us-west-2/release"
FALLBACK_RELEASE = "2026-09-23.1"
DIVISION_AREA = "theme=divisions/type=division_area/*.parquet"

# Name clean-ups applied at download time, so questions read naturally.
#   India: ~35 districts carry a " district" suffix ("Leh district") while the
#   other 750 do not.  Ecuador: three Galápagos cantons carry a "Cantón " prefix.
# A small reminder that real gazetteers are inconsistent.
NAME_STRIP_PATTERNS = [r"\s+[Dd]istrict$", r"^Cantón\s+(de\s+|del\s+)?"]


def _name_expr(col: str = 'names."primary"') -> str:
    expr = f"trim({col})"
    for pat in NAME_STRIP_PATTERNS:
        expr = f"regexp_replace({expr}, '{pat}', '')"
    return expr


def connect(spatial: bool = True, httpfs: bool = False, threads: int | None = None):
    """DuckDB connection with the spatial (and optionally httpfs) extension loaded."""
    con = duckdb.connect()
    try:
        con.execute("SET enable_progress_bar=false")
    except duckdb.Error:  # Jupyter without ipywidgets refuses to touch this setting
        pass
    if spatial:
        con.execute("INSTALL spatial; LOAD spatial;")
    if httpfs:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    return con


def latest_release(con: duckdb.DuckDBPyConnection) -> str:
    """Newest Overture release that has a divisions/division_area folder."""
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT regexp_extract(file, 'release/([^/]+)/', 1) AS release
            FROM glob('{OVERTURE_RELEASES}/*/theme=divisions/type=division_area/*.parquet')
            ORDER BY release DESC
            """
        ).fetchall()
        releases = [r[0] for r in rows if r[0]]
        if releases:
            return releases[0]
    except Exception as exc:  # network / permissions
        print(f"Could not list releases ({exc.__class__.__name__}); using {FALLBACK_RELEASE}")
    return FALLBACK_RELEASE


def overture_source(release: str) -> str:
    return f"{OVERTURE_RELEASES}/{release}/{DIVISION_AREA}"


def download_divisions(
    out_path: str | Path,
    source: str | None = None,
    release: str | None = None,
    country: str = COUNTRY,
    bbox: tuple[float, float, float, float] = COUNTRY_BBOX,
    subtypes: tuple[str, ...] = SUBTYPES,
    con: duckdb.DuckDBPyConnection | None = None,
    clean: bool = True,
) -> Path:
    """Pull one country's division areas into a local GeoParquet file.

    `source` may be a local parquet glob (fast, for development) or omitted to
    read the Overture release straight from S3. The bbox filter lets DuckDB
    skip row groups by their min/max statistics, which keeps the S3 scan to a
    couple of minutes and well under Colab's RAM.

    With `clean=True` three fixes the India data needs are applied (each is
    reported, and each is a no-op where it does not apply):

    1. a `region` row without a region code is dropped (Overture ships one
       disputed-area polygon for India from a non-Indian perspective);
    2. a lower-level row without a region code inherits the code of the
       region polygon it overlaps most (19 Arunachal Pradesh districts);
    3. exact duplicates on (subtype, name, region) keep the largest polygon
       (Leh and Kargil appear twice).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    own_con = con is None
    con = con or connect(spatial=True, httpfs=source is None or source.startswith("s3://"))
    if source is None:
        release = release or latest_release(con)
        source = overture_source(release)
        print(f"Overture release: {release}")
    print(f"Reading: {source}")

    minx, miny, maxx, maxy = (float(v) for v in bbox)
    subtype_list = ", ".join(f"'{s}'" for s in subtypes)
    t0 = time.time()
    # GeoParquet readers hand us a GEOMETRY column; a plain parquet reader hands us
    # WKB bytes. Round-trip through WKB either way so the output is one clean type.
    geom_type = con.execute(f"DESCRIBE SELECT geometry FROM read_parquet('{source}')").fetchone()[1]
    geom_expr = "ST_GeomFromWKB(geometry)" if geom_type.upper() == "BLOB" else "ST_GeomFromWKB(ST_AsWKB(geometry))"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _raw AS
        SELECT
            id,
            {_name_expr()} AS name,
            subtype,
            country,
            region,
            admin_level,
            {geom_expr} AS geometry
        FROM read_parquet('{source}')
        WHERE country = '{country}'
          AND subtype IN ({subtype_list})
          AND is_land
          AND bbox.xmin >= {minx} AND bbox.xmax <= {maxx}
          AND bbox.ymin >= {miny} AND bbox.ymax <= {maxy}
        """
    )
    n_raw = con.execute("SELECT count(*) FROM _raw").fetchone()[0]
    print(f"Read {n_raw} rows in {time.time() - t0:.0f}s")

    if clean:
        dropped = con.execute(
            "SELECT list(name) FROM _raw WHERE subtype = 'region' AND region IS NULL"
        ).fetchone()[0] or []
        con.execute("DELETE FROM _raw WHERE subtype = 'region' AND region IS NULL")

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _fix AS
            SELECT c.id,
                   arg_max(r.region, ST_Area(ST_Intersection(c.geometry, r.geometry))) AS region
            FROM _raw AS c
            JOIN _raw AS r
              ON r.subtype = 'region' AND r.region IS NOT NULL
             AND ST_Intersects(ST_Envelope(c.geometry), ST_Envelope(r.geometry))
             AND ST_Intersects(c.geometry, r.geometry)
            WHERE c.subtype != 'country' AND c.region IS NULL
            GROUP BY c.id
            """
        )
        fixed = con.execute("SELECT count(*) FROM _fix").fetchone()[0]
        con.execute(
            "UPDATE _raw SET region = (SELECT f.region FROM _fix f WHERE f.id = _raw.id) "
            "WHERE region IS NULL AND subtype != 'country'"
        )
        still_null = con.execute(
            "SELECT count(*) FROM _raw WHERE subtype != 'country' AND region IS NULL"
        ).fetchone()[0]
        con.execute("DELETE FROM _raw WHERE subtype != 'country' AND region IS NULL")
        n_before = con.execute("SELECT count(*) FROM _raw").fetchone()[0]
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _dedup AS
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY subtype, name, coalesce(region, '')
                    ORDER BY ST_Area(geometry) DESC, id) AS rn
                FROM _raw)
            WHERE rn = 1
            """
        )
        n_after = con.execute("SELECT count(*) FROM _dedup").fetchone()[0]
        print(
            "Cleaning: dropped region rows without a code "
            f"{dropped}; filled region for {fixed} rows by overlap"
            + (f" ({still_null} unresolved, dropped)" if still_null else "")
            + f"; removed {n_before - n_after} duplicate rows"
        )
        table = "_dedup"
    else:
        table = "_raw"

    con.execute(
        f"""
        COPY (
            SELECT id, {{'primary': name}} AS names, subtype, country, region, admin_level, geometry
            FROM {table}
            ORDER BY admin_level, name
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute("DROP TABLE IF EXISTS _raw; DROP TABLE IF EXISTS _fix; DROP TABLE IF EXISTS _dedup;")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    if own_con:
        con.close()
    return out_path


def ensure_geodata(
    path: str | Path | None = None,
    source: str | None = None,
    release: str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> Path:
    """Return the local divisions parquet, downloading it (Overture S3 or `source`) if missing."""
    path = Path(path) if path else default_geodata_path()
    if path.exists():
        print(f"Using cached geodata: {path}")
        return path
    return download_divisions(path, source=source, release=release, con=con)


def summarize(con: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """Row counts and distinct names per subtype."""
    return con.execute(
        """
        SELECT subtype, count(*) AS rows, count(DISTINCT names."primary") AS distinct_names
        FROM read_parquet(?) GROUP BY subtype ORDER BY min(admin_level)
        """,
        [str(path)],
    ).fetchdf()


def duplicate_names(con: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """Names shared by more than one feature; the raw material for disambiguation."""
    return con.execute(
        """
        SELECT names."primary" AS name,
               list(subtype ORDER BY subtype) AS subtypes,
               list(region ORDER BY region) AS regions,
               count(*) AS n
        FROM read_parquet(?)
        GROUP BY 1 HAVING count(*) > 1 ORDER BY n DESC, name
        """,
        [str(path)],
    ).fetchdf()


def inventory(con: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """Compact table used to sample anchors (port of build_inventory.py)."""
    return con.execute(
        """
        SELECT id, names."primary" AS name, subtype, country, region, admin_level,
               ST_Area(geometry) AS area_sq_deg
        FROM read_parquet(?)
        WHERE names."primary" IS NOT NULL AND trim(names."primary") != '' AND geometry IS NOT NULL
        ORDER BY admin_level, name
        """,
        [str(path)],
    ).fetchdf()


def build_adjacency_pairs(con: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """Directed pairs of features that share a border (ST_Touches), both directions.

    gazet keeps only a.id < b.id; the generator needs either side as an anchor,
    so both directions are emitted here.
    """
    return con.execute(
        """
        WITH f AS (
            SELECT id, names."primary" AS name, subtype, region, geometry, ST_Envelope(geometry) AS bbox
            FROM read_parquet(?)
        ),
        pairs AS (
            SELECT a.id AS a_id, a.name AS a_name, a.subtype AS a_subtype, a.region AS a_region,
                   b.id AS b_id, b.name AS b_name, b.subtype AS b_subtype, b.region AS b_region
            FROM f AS a JOIN f AS b
              ON a.id < b.id
             AND ST_Intersects(a.bbox, b.bbox)
             AND ST_Touches(a.geometry, b.geometry)
        )
        SELECT a_id AS anchor_id, a_name AS anchor_name, a_subtype AS anchor_subtype, a_region AS anchor_region,
               b_id AS target_id, b_name AS target_name, b_subtype AS target_subtype, b_region AS target_region
        FROM pairs
        UNION ALL
        SELECT b_id, b_name, b_subtype, b_region, a_id, a_name, a_subtype, a_region FROM pairs
        ORDER BY anchor_name, target_name
        """,
        [str(path)],
    ).fetchdf()


def build_containment_pairs(con: duckdb.DuckDBPyConnection, path: str | Path) -> pd.DataFrame:
    """(container, contained) pairs via ST_Within for country→region, country→county, region→county."""
    frames = []
    for container_st, contained_st in (("country", "region"), ("country", "county"), ("region", "county")):
        df = con.execute(
            f"""
            WITH a AS (
                SELECT id, names."primary" AS name, subtype, region, geometry, ST_Envelope(geometry) AS bbox
                FROM read_parquet(?) WHERE subtype = '{container_st}'
            ),
            b AS (
                SELECT id, names."primary" AS name, subtype, region, geometry, ST_Envelope(geometry) AS bbox
                FROM read_parquet(?) WHERE subtype = '{contained_st}'
            )
            SELECT a.id AS container_id, a.name AS container_name, a.subtype AS container_subtype,
                   a.region AS container_region,
                   b.id AS contained_id, b.name AS contained_name, b.subtype AS contained_subtype,
                   b.region AS contained_region
            FROM a JOIN b
              ON a.id != b.id
             AND ST_Intersects(a.bbox, b.bbox)
             AND ST_Within(b.geometry, a.geometry)
            ORDER BY container_name, contained_name
            """,
            [str(path), str(path)],
        ).fetchdf()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def containment_coverage(pairs: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Share of each contained subtype that found at least one container (sliver check)."""
    rows = []
    for st in ("region", "county"):
        total = int((inv["subtype"] == st).sum())
        covered = pairs.loc[pairs["contained_subtype"] == st, "contained_id"].nunique()
        rows.append({"subtype": st, "features": total, "with_container": covered,
                     "coverage": round(covered / total, 3) if total else None})
    return pd.DataFrame(rows)
