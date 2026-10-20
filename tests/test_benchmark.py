import json
import random

import pandas as pd

from gazet_tutorial.benchmark import (
    COMPOSITIONAL_BY_ID,
    holdout_ids,
    load_subset_ids,
    make_noisy,
    question_in_user,
    select_subset,
    typo,
)
from gazet_tutorial.india import NAME_ALIASES, STATE_ABBREVIATIONS, replace_name, resolve_aliases
from gazet_tutorial.sqlutil import execute, result_ids


def test_holdout_ids_cover_states_and_duplicates():
    inv = pd.DataFrame([
        {"id": "s1", "name": "Kerala", "subtype": "region", "region": "IN-KL"},
        {"id": "s2", "name": "Goa", "subtype": "region", "region": "IN-GA"},
        {"id": "s3", "name": "Punjab", "subtype": "region", "region": "IN-PB"},
        {"id": "d1", "name": "Kollam", "subtype": "county", "region": "IN-KL"},
        {"id": "d2", "name": "North Goa", "subtype": "county", "region": "IN-GA"},
        {"id": "d3", "name": "Bilaspur", "subtype": "county", "region": "IN-HP"},
        {"id": "d4", "name": "Bilaspur", "subtype": "county", "region": "IN-CG"},
        {"id": "d5", "name": "Ludhiana", "subtype": "county", "region": "IN-PB"},
        {"id": "d6", "name": "Patiala", "subtype": "county", "region": "IN-PB"},
    ])
    region_ids, anchor_ids = holdout_ids(inv, {"IN-KL": "Kerala", "IN-GA": "Goa"}, n_random=2, seed=1)
    assert region_ids == {"s1", "s2", "d1", "d2"}
    assert len(anchor_ids) == 2 and anchor_ids & {"d3", "d4"} and not anchor_ids & region_ids
    again = holdout_ids(inv, {"IN-KL": "Kerala", "IN-GA": "Goa"}, n_random=2, seed=1)
    assert again == (region_ids, anchor_ids)


def test_typo_and_replace_name():
    rng = random.Random(0)
    for name in ["Bhopal", "Thiruvananthapuram", "Pune"]:
        bad = typo(name, rng)
        assert bad != name and bad[0] == name[0] and bad[-1] == name[-1]
    assert typo("Goa", rng) == "Goa"
    assert replace_name("districts of UP", "UP", "Uttar Pradesh") == "districts of Uttar Pradesh"
    assert replace_name("show Puri", "Pur", "x") is None


def test_make_noisy_types():
    aliases = {"Mysuru": ["Mysore"]}
    abbr = {"Uttar Pradesh": ["UP"]}
    rng = random.Random(3)
    out = make_noisy("show me Mysuru", "Mysuru", None, rng, aliases, abbr, noise_type="alias")
    assert out["question"] == "show me Mysore" and out["search_strings"] == ["Mysore"] and out["places"] == ["Mysuru"]
    out = make_noisy("Hamirpur, Uttar Pradesh", "Hamirpur", "Uttar Pradesh", rng, aliases, abbr, noise_type="abbreviation")
    assert out["question"] == "Hamirpur, UP" and out["search_strings"] == ["Hamirpur", "UP"] and out["places"] == ["Hamirpur", "Uttar Pradesh"]
    out = make_noisy("Which districts border Bhopal?", "Bhopal", None, rng, aliases, abbr, noise_type="lowercase")
    assert out["question"] == "which districts border bhopal?" and out["places"] == ["bhopal"]
    out = make_noisy("districts of Nagpur", "Nagpur", None, rng, aliases, abbr, noise_type="typo")
    assert out["noise_type"] == "typo" and out["search_strings"][0] != "Nagpur" and out["places"] == out["search_strings"]
    assert make_noisy("goa", "Goa", None, rng, aliases, abbr) is None


def test_select_subset_balanced_and_stable():
    rows = [{"metadata": {"sample_id": f"s{i}", "template_id": f"t{i % 3}"}} for i in range(30)]
    ids = select_subset(rows, 9, seed=1)
    assert len(ids) == 9 == len(set(ids))
    tmpl = [int(i[1:]) % 3 for i in ids]
    assert tmpl.count(0) == tmpl.count(1) == tmpl.count(2) == 3
    assert ids == select_subset(rows, 9, seed=1)


def test_india_tables_shape():
    assert all(isinstance(v, list) and v for v in NAME_ALIASES.values())
    assert all(isinstance(v, list) and v for v in STATE_ABBREVIATIONS.values())
    assert resolve_aliases({"Mysuru", "Nowhere"}) == {"Mysuru": ["Mysore"]}
    assert question_in_user("<CANDIDATES>\nx\n</CANDIDATES>\n\n<USER_QUERY>\nhello there\n</USER_QUERY>") == "hello there"


def test_compositional_reference_sql(con, divisions_path):
    # counties of North that touch South: Alpha(N) and Beta both sit on y=2
    t = COMPOSITIONAL_BY_ID["adj_within"]
    sql = t.sql.format(container_id="e0000000-0000-0000-0000-000000000001", neighbor_id="e0000000-0000-0000-0000-000000000002")
    assert set(execute(con, sql, divisions_path)["name"]) == {"Alpha", "Beta"}
    # largest neighbour of Gamma (touches Alpha(N) and Alpha(S), both 2x2) -> ordered result of length 1
    t = COMPOSITIONAL_BY_ID["agg_adjacent"]
    sql = t.sql.format(anchor_id="d0000000-0000-0000-0000-000000000001", order="DESC", top_n=1)
    assert len(result_ids(execute(con, sql, divisions_path), ordered=True)) == 1
    t = COMPOSITIONAL_BY_ID["buffer_agg"]
    sql = t.sql.format(anchor_id="d0000000-0000-0000-0000-000000000001", buffer_m=50000, top_n=3)
    assert len(execute(con, sql, divisions_path)) == 3
    for t in COMPOSITIONAL_BY_ID.values():
        assert "read_parquet('divisions_area')" in t.sql and "ST_AsGeoJSON" in t.sql and len(t.hints) >= 4


def test_load_subset_ids_missing(tmp_path):
    assert load_subset_ids(tmp_path) == {}
    (tmp_path / "subset_ids.json").write_text(json.dumps({"iid": {"sql": ["a"]}}))
    assert load_subset_ids(tmp_path)["iid"]["sql"] == ["a"]
