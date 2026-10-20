from gazet_tutorial.geodata import (
    build_adjacency_pairs,
    build_containment_pairs,
    containment_coverage,
    duplicate_names,
    inventory,
    summarize,
)


def test_summary_and_inventory(con, divisions_path):
    s = summarize(con, divisions_path).set_index("subtype")
    assert s.loc["country", "rows"] == 1 and s.loc["region", "rows"] == 2 and s.loc["county", "rows"] == 4
    assert s.loc["county", "distinct_names"] == 3  # two Alphas
    inv = inventory(con, divisions_path)
    assert len(inv) == 7 and "area_sq_deg" in inv.columns
    dups = duplicate_names(con, divisions_path)
    assert dups["name"].tolist() == ["Alpha"] and dups["n"].iloc[0] == 2


def test_adjacency_is_symmetric_and_correct(con, divisions_path):
    adj = build_adjacency_pairs(con, divisions_path)
    # Alpha(North) touches Beta, Gamma, Alpha(South) and region South; never the country.
    alpha_n = adj[adj["anchor_id"] == "a0000000-0000-0000-0000-000000000001"]
    assert set(alpha_n["target_name"]) == {"Beta", "Gamma", "Alpha", "South"}
    assert "country" not in set(adj["target_subtype"])
    pairs = set(zip(adj["anchor_id"], adj["target_id"]))
    assert all((b, a) in pairs for a, b in pairs)


def test_containment_pairs_and_coverage(con, divisions_path):
    pairs = build_containment_pairs(con, divisions_path)
    north = pairs[(pairs["container_name"] == "North") & (pairs["contained_subtype"] == "county")]
    assert set(north["contained_name"]) == {"Alpha", "Beta"}
    country = pairs[pairs["container_subtype"] == "country"]
    assert len(country) == 6  # 2 regions + 4 counties
    cov = containment_coverage(pairs, inventory(con, divisions_path)).set_index("subtype")
    assert cov.loc["county", "coverage"] == 1.0 and cov.loc["region", "coverage"] == 1.0
