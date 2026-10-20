"""download_divisions cleaning steps on a synthetic Overture-shaped parquet."""

import duckdb
import pytest

from gazet_tutorial.geodata import download_divisions

# id, name, subtype, region, admin_level, is_land, minx, miny, maxx, maxy
RAW = [
    ("c0000000-0000-0000-0000-000000000001", "Testland", "country", None, 0, True, 0, 0, 4, 4),
    ("c0000000-0000-0000-0000-000000000002", "Testland", "country", None, 0, False, -1, -1, 5, 5),   # territorial, dropped by is_land
    ("e0000000-0000-0000-0000-000000000001", "North", "region", "TL-N", 1, True, 0, 2, 4, 4),
    ("e0000000-0000-0000-0000-000000000002", "South", "region", "TL-S", 1, True, 0, 0, 4, 2),
    ("e0000000-0000-0000-0000-000000000003", "藏南", "region", None, 1, True, 3, 3, 4, 4),            # foreign-perspective row, dropped
    ("a0000000-0000-0000-0000-000000000001", "Alpha district", "county", "TL-N", 2, True, 0, 2, 2, 4),  # suffix stripped
    ("b0000000-0000-0000-0000-000000000001", "Beta", "county", None, 2, True, 2, 2, 4, 4),           # region filled by overlap -> TL-N
    ("d0000000-0000-0000-0000-000000000001", "Gamma", "county", "TL-S", 2, True, 0, 0, 2, 2),
    ("d0000000-0000-0000-0000-000000000002", "Gamma", "county", "TL-S", 2, True, 0, 0, 1, 1),        # duplicate, smaller, dropped
    ("a0000000-0000-0000-0000-000000000002", "Alpha", "county", "TL-S", 2, True, 2, 0, 4, 2),        # same name, other region: kept
    ("f0000000-0000-0000-0000-000000000001", "Far", "county", "TL-S", 2, True, 40, 40, 41, 41),      # outside bbox, dropped
]


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory):
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    path = tmp_path_factory.mktemp("raw") / "division_area.parquet"
    con.execute("CREATE TABLE src (id VARCHAR, name VARCHAR, subtype VARCHAR, region VARCHAR, admin_level INTEGER, is_land BOOLEAN, minx DOUBLE, miny DOUBLE, maxx DOUBLE, maxy DOUBLE)")
    con.executemany("INSERT INTO src VALUES (?,?,?,?,?,?,?,?,?,?)", RAW)
    con.execute(
        f"""
        COPY (
            SELECT id, {{'primary': name}} AS names, subtype, 'TL' AS country, region, admin_level, is_land,
                   {{'xmin': minx, 'xmax': maxx, 'ymin': miny, 'ymax': maxy}} AS bbox,
                   ST_AsWKB(ST_MakeEnvelope(minx, miny, maxx, maxy)) AS geometry
            FROM src
        ) TO '{path}' (FORMAT PARQUET)
        """
    )
    con.close()
    return path


def test_cleaning(raw_path, tmp_path, con):
    out = download_divisions(tmp_path / "clean.parquet", source=str(raw_path), country="TL", bbox=(-2, -2, 10, 10), con=con)
    df = con.execute(f"""SELECT id, names."primary" AS name, subtype, region FROM read_parquet('{out}') ORDER BY subtype, name, region""").fetchdf()
    assert df["subtype"].value_counts().to_dict() == {"county": 4, "region": 2, "country": 1}
    assert "藏南" not in set(df["name"])
    assert df.loc[df["id"] == "b0000000-0000-0000-0000-000000000001", "region"].iloc[0] == "TL-N"
    assert df.loc[df["id"] == "a0000000-0000-0000-0000-000000000001", "name"].iloc[0] == "Alpha"
    assert (df["name"] == "Gamma").sum() == 1 and "d0000000-0000-0000-0000-000000000001" in set(df["id"])
    assert (df["name"] == "Alpha").sum() == 2
    assert "Far" not in set(df["name"])
