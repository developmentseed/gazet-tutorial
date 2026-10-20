from gazet_tutorial.sqlutil import (
    execute,
    format_sql,
    normalize_ws,
    result_ids,
    rewrite_data_paths,
    strip_fences,
    to_feature_collection,
)


def test_rewrite_symbolic_and_hallucinated_paths():
    real = "/tmp/x.parquet"
    assert rewrite_data_paths("SELECT * FROM read_parquet('divisions_area')", real) == f"SELECT * FROM read_parquet('{real}')"
    assert f"read_parquet('{real}')" in rewrite_data_paths('FROM read_parquet("divisions_area")', real)
    assert f"read_parquet('{real}')" in rewrite_data_paths("FROM read_parquet('/data/overture/division_area/*.parquet')", real)
    # unrelated tables untouched
    assert rewrite_data_paths("read_parquet('other.parquet')", real) == "read_parquet('other.parquet')"


def test_strip_fences_variants():
    assert strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"
    assert strip_fences("```\nSELECT 1\n```") == "SELECT 1"
    assert strip_fences("<think>\nhmm\n</think>\n\nSELECT 1") == "SELECT 1"
    assert strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert strip_fences(None) == ""


def test_format_sql_is_idempotent():
    sql = "select id, names.\"primary\" as name from read_parquet('divisions_area') where id = 'x'"
    once = format_sql(sql)
    assert once == format_sql(once)
    assert "SELECT" in once and "FROM" in once
    assert normalize_ws(once) == normalize_ws(format_sql(once))


def test_execute_and_result_ids(con, divisions_path):
    df = execute(con, "SELECT id, subtype FROM read_parquet('divisions_area') WHERE subtype = 'county'", divisions_path)
    assert len(df) == 4
    assert result_ids(df) == sorted(df["id"])
    assert result_ids(df, ordered=True) == df["id"].tolist()


def test_to_feature_collection(con, divisions_path):
    df = execute(con, "SELECT id, names.\"primary\" AS name, ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE subtype='region'", divisions_path)
    fc = to_feature_collection(df)
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    assert "name" in fc["features"][0]["properties"]


def test_rewrite_bare_table_name(tmp_path):
    from gazet_tutorial.sqlutil import rewrite_data_paths
    real = str(tmp_path / "divisions_area.parquet")
    out = rewrite_data_paths("SELECT d.id FROM divisions_area d JOIN divisions_area AS e ON e.id = d.id", real)
    assert out.count(f"read_parquet('{real}')") == 2 and " divisions_area " not in out
    # already-rewritten paths and identifiers containing the name are left alone
    out2 = rewrite_data_paths(f"SELECT * FROM read_parquet('{real}') WHERE divisions_area_id = 1", real)
    assert out2 == f"SELECT * FROM read_parquet('{real}') WHERE divisions_area_id = 1"
    assert rewrite_data_paths("SELECT * FROM read_parquet('divisions_area')", real) == f"SELECT * FROM read_parquet('{real}')"
