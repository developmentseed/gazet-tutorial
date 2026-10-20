"""Shared fixtures: a tiny synthetic 'divisions_area' parquet built from squares.

Layout (lon/lat degrees):
  country 'Testland' = square [0,0]-[4,4]
  region 'North' = [0,2]-[4,4], region 'South' = [0,0]-[4,2]
  counties: 'Alpha' [0,2]-[2,4], 'Beta' [2,2]-[4,4] (both in North),
            'Gamma' [0,0]-[2,2], 'Alpha' [2,0]-[4,2] (both in South; second Alpha = same-name distractor)
"""

from __future__ import annotations

import pytest

from gazet_tutorial.geodata import connect

ROWS = [
    # id, name, subtype, region, admin_level, minx, miny, maxx, maxy
    ("c0000000-0000-0000-0000-000000000001", "Testland", "country", None, 0, 0, 0, 4, 4),
    ("e0000000-0000-0000-0000-000000000001", "North", "region", "TL-N", 1, 0, 2, 4, 4),
    ("e0000000-0000-0000-0000-000000000002", "South", "region", "TL-S", 1, 0, 0, 4, 2),
    ("a0000000-0000-0000-0000-000000000001", "Alpha", "county", "TL-N", 2, 0, 2, 2, 4),
    ("b0000000-0000-0000-0000-000000000001", "Beta", "county", "TL-N", 2, 2, 2, 4, 4),
    ("d0000000-0000-0000-0000-000000000001", "Gamma", "county", "TL-S", 2, 0, 0, 2, 2),
    ("a0000000-0000-0000-0000-000000000002", "Alpha", "county", "TL-S", 2, 2, 0, 4, 2),
]


@pytest.fixture(scope="session")
def con():
    c = connect(spatial=True)
    yield c
    c.close()


@pytest.fixture(scope="session")
def divisions_path(tmp_path_factory, con):
    path = tmp_path_factory.mktemp("data") / "divisions_area.parquet"
    con.execute("CREATE OR REPLACE TABLE src (id VARCHAR, name VARCHAR, subtype VARCHAR, region VARCHAR, admin_level INTEGER, minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE)")
    con.executemany("INSERT INTO src VALUES (?,?,?,?,?,?,?,?,?)", ROWS)
    con.execute(
        f"""
        COPY (
            SELECT id, {{'primary': name}} AS names, subtype, 'TL' AS country, region, admin_level,
                   ST_MakeEnvelope(minx, miny, maxx, maxy) AS geometry
            FROM src
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    return path
