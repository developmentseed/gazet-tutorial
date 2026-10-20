from gazet_tutorial.search import (
    CANDIDATE_COLUMNS,
    candidates_csv,
    fetch_by_id,
    fuzzy_search,
    merge_candidates,
    per_place_limit,
)


def test_fuzzy_search_ranks_substring_before_near_miss(con, divisions_path):
    df = fuzzy_search(con, divisions_path, "Alph", limit=5)
    # both 'Alpha' rows are substring matches and come first
    assert df["name"].tolist()[:2] == ["Alpha", "Alpha"]
    assert df["is_substring_match"].iloc[0]


def test_fuzzy_search_exclude_id(con, divisions_path):
    df = fuzzy_search(con, divisions_path, "Alpha", limit=10, exclude_id="a0000000-0000-0000-0000-000000000001")
    assert "a0000000-0000-0000-0000-000000000001" not in df["id"].tolist()
    assert len(df) == 6


def test_fetch_by_id(con, divisions_path):
    df = fetch_by_id(con, divisions_path, "e0000000-0000-0000-0000-000000000001", geometry=True)
    assert df.iloc[0]["name"] == "North" and df.iloc[0]["geometry"].startswith('{"type":"Polygon"')


def test_candidates_csv_columns(con, divisions_path):
    df = fuzzy_search(con, divisions_path, "Beta", limit=3)
    csv = candidates_csv(df)
    header = csv.splitlines()[0].split(",")
    assert header == CANDIDATE_COLUMNS
    assert "similarity" not in header and "country" not in header


def test_merge_and_limit(con, divisions_path):
    a = fuzzy_search(con, divisions_path, "Alpha", limit=3)
    b = fuzzy_search(con, divisions_path, "Alpha", limit=3)
    merged = merge_candidates([a, b])
    assert merged["id"].is_unique
    assert (per_place_limit(1), per_place_limit(2), per_place_limit(3)) == (10, 5, 4)
