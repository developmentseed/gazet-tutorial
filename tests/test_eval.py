from gazet_tutorial.eval import (
    anchor_grounded,
    compare,
    evaluate_places,
    evaluate_sql,
    execution_match,
    parse_places,
    places_prf,
    summarize,
)

GOLD = """SELECT b.id, b.names."primary" AS name, ST_AsGeoJSON(b.geometry) AS geometry
FROM read_parquet('divisions_area') AS b, (SELECT geometry FROM read_parquet('divisions_area') WHERE id = 'e0000000-0000-0000-0000-000000000001') AS a
WHERE b.id != 'e0000000-0000-0000-0000-000000000001' AND b.subtype = 'county' AND ST_Within(b.geometry, a.geometry)"""


def test_execution_match_true_false_error(con, divisions_path):
    assert execution_match(con, GOLD, GOLD, divisions_path)["exec_match"]
    wrong = GOLD.replace("e0000000-0000-0000-0000-000000000001", "e0000000-0000-0000-0000-000000000002")
    res = execution_match(con, wrong, GOLD, divisions_path)
    assert res["sql_valid"] and not res["exec_match"] and res["error"] == "wrong feature set"
    bad = execution_match(con, "SELECT nope FROM read_parquet('divisions_area')", GOLD, divisions_path)
    assert not bad["sql_valid"] and bad["error"]
    assert not execution_match(con, "", GOLD, divisions_path)["sql_valid"]


def test_execution_match_ordered(con, divisions_path):
    top = "SELECT id FROM read_parquet('divisions_area') WHERE subtype='county' ORDER BY ST_Area(geometry) DESC, id LIMIT 2"
    rev = "SELECT id FROM (" + top + ") ORDER BY id DESC"
    assert execution_match(con, top, top, divisions_path, ordered=True)["exec_match"]
    assert not execution_match(con, rev, top, divisions_path, ordered=True)["exec_match"]
    assert execution_match(con, rev, top, divisions_path, ordered=False)["exec_match"]


def test_anchor_grounded():
    assert anchor_grounded(GOLD, GOLD)
    assert not anchor_grounded(GOLD.replace("e0000000", "e1111111"), GOLD)


def test_places_prf_and_parse():
    gold = '{"places": [{"place": "Loja"}, {"place": "Azuay"}]}'
    assert parse_places("```json\n" + gold + "\n```") == ["Loja", "Azuay"]
    assert parse_places("not json") is None
    perfect = places_prf('{"places":[{"place":"azuay"},{"place":"LOJA"}]}', gold)
    assert perfect["exact_set"] and perfect["f1"] == 1.0
    half = places_prf('{"places":[{"place":"Loja"}]}', gold)
    assert half["json_valid"] and not half["exact_set"] and half["recall"] == 0.5 and half["precision"] == 1.0
    assert places_prf("garbage", gold) == {"json_valid": False, "exact_set": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_evaluate_and_summarize(con, divisions_path):
    sample = {"messages": [{"role": "system", "content": "s"},
                           {"role": "user", "content": "<CANDIDATES>\nx\n</CANDIDATES>\n\n<USER_QUERY>\ncounties in North\n</USER_QUERY>"},
                           {"role": "assistant", "content": GOLD}],
              "metadata": {"task_family": "containment", "template_id": "contain_01"}}
    df = evaluate_sql([sample, sample], [GOLD, "SELECT 1"], con, divisions_path)
    assert df["question"].iloc[0] == "counties in North"
    assert df["exec_match"].tolist() == [True, False]
    s = summarize(df)
    assert float(s["exec_match"].iloc[0]) == 0.5
    by = summarize(df, by="task_family")
    assert by.loc["containment", "n"] == 2
    cmp = compare({"base": df, "ft": df})
    assert list(cmp.index) == ["base", "ft"]

    psample = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Loja, Azuay"},
                            {"role": "assistant", "content": '{"places": [{"place": "Loja"}]}'}],
               "metadata": {"task_family": "disambiguation", "template_id": "disambiguate_01"}}
    pdf = evaluate_places([psample], ['{"places": [{"place": "loja"}]}'])
    assert pdf["exact_set"].iloc[0] and pdf["question"].iloc[0] == "Loja, Azuay"


def test_execution_match_falls_back_to_geometry(con, divisions_path):
    from gazet_tutorial.eval import execution_match
    gold = "SELECT id, ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Beta'"
    pred_geom_only = "SELECT ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Beta'"
    pred_wrong = "SELECT ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Gamma'"
    ok = execution_match(con, pred_geom_only, gold, divisions_path)
    assert ok["exec_match"] and ok.get("matched_on") == "geometry"
    assert not execution_match(con, pred_wrong, gold, divisions_path)["exec_match"]
    no_cols = "SELECT names.\"primary\" AS name FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Beta'"
    res = execution_match(con, no_cols, gold, divisions_path)
    assert not res["exec_match"] and "no id or geometry" in res["error"]
