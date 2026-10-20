"""Execute the `templates` cell of notebook 1 and check every template renders."""

import json
from pathlib import Path

import pytest

NB_SRC = Path(__file__).resolve().parents[1] / "notebooks" / "src" / "01_data_and_dataset.py"


@pytest.fixture(scope="module")
def templates_ns():
    import jupytext
    nb = jupytext.read(NB_SRC)
    cell = next(c for c in nb.cells if "templates" in c.metadata.get("tags", []))
    ns = {}
    exec("from dataclasses import dataclass, field\nfrom gazet_tutorial import TABLE\n", ns)
    exec(cell.source, ns)
    return ns


def test_templates_render(templates_ns):
    templates = templates_ns["TEMPLATES"]
    assert len(templates) == 8
    slots = dict(anchor_id="x", target_subtype="county", buffer_m=1000, top_n=3, order="DESC")
    hint_slots = dict(anchor_name="A", container_name="B", target_label="cantons", target_singular="canton",
                      buffer_label="1 km", top_n=3, superlative="largest")
    for t in templates:
        sql = t.sql.format(**slots)
        assert "read_parquet('divisions_area')" in sql and "ST_AsGeoJSON" in sql, t.template_id
        assert "{" not in sql, t.template_id
        assert len(t.hints) >= 8, t.template_id
        for h in t.hints:
            assert "{" not in h.format(**hint_slots), (t.template_id, h)
    assert {t.family for t in templates} == {"direct_lookup", "disambiguation", "adjacency", "containment", "buffer", "aggregation"}


def test_bundled_dataset_is_consistent():
    from gazet_tutorial import artifacts_dir
    from gazet_tutorial.benchmark import load_benchmark, load_subset_ids
    ds = artifacts_dir() / "dataset"
    stats = json.loads((ds / "stats.json").read_text())
    prompts = json.loads((ds / "prompts.json").read_text())
    assert {"sql_system", "place_system", "user_template", "candidate_columns"} <= prompts.keys()
    assert "IN-" in prompts["sql_system"]
    for split in ("train", "val", "test"):
        n_sql = sum(1 for _ in open(ds / "sql" / f"{split}.jsonl"))
        n_pl = sum(1 for _ in open(ds / "places" / f"{split}.jsonl"))
        assert n_sql == n_pl == stats["splits"][split]["n"]
    bench = load_benchmark(artifacts_dir() / "benchmark")
    assert {"iid", "unseen_region", "unseen_anchor", "noisy", "compositional"} <= bench.keys()
    subset = load_subset_ids(artifacts_dir() / "benchmark")
    for split, tasks in bench.items():
        ids = {r["metadata"]["sample_id"] for r in tasks["sql"]}
        assert len(ids) == len(tasks["sql"]) and ids == {r["metadata"]["sample_id"] for r in tasks["places"]}
        if split in subset:
            assert set(subset[split]["sql"]) <= ids and set(subset[split]["places"]) <= set(subset[split]["sql"])
    # held-out anchors never anchor a training sample
    train_anchors = {json.loads(line)["metadata"]["anchor_id"] for line in open(ds / "sql" / "train.jsonl")}
    for split in ("unseen_region", "unseen_anchor"):
        assert not train_anchors & {r["metadata"]["anchor_id"] for r in bench[split]["sql"]}, split


def test_adjacency_templates_keep_one_level(templates_ns):
    by_id = {t.template_id: t for t in templates_ns["TEMPLATES"]}
    assert "b.subtype = a.subtype" in by_id["adj_01"].sql          # neighbours at the anchor's own level
    assert "b.subtype = '{target_subtype}'" in by_id["adj_02"].sql  # the level the question names


def test_gold_subtypes_match_the_question():
    """'states ...' questions return only states, 'districts ...' only districts, adjacency never mixes levels."""
    import re

    import pandas as pd

    from gazet_tutorial import artifacts_dir
    art = artifacts_dir()
    adj = pd.read_parquet(art / "relations" / "adjacency_pairs.parquet")
    con = pd.read_parquet(art / "relations" / "containment_pairs.parquet")
    subtype = {**dict(zip(adj.anchor_id, adj.anchor_subtype)), **dict(zip(adj.target_id, adj.target_subtype)),
               **dict(zip(con.container_id, con.container_subtype)), **dict(zip(con.contained_id, con.contained_subtype))}
    files = list((art / "benchmark" / "sql").glob("*.jsonl")) + list((art / "dataset" / "sql").glob("*.jsonl"))
    assert files
    bad = []
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            m = r["metadata"]
            q = r["messages"][1]["content"].split("<USER_QUERY>")[-1].split("</USER_QUERY>")[0].lower()
            kinds = {subtype[i] for i in m["gold_ids"] if i in subtype}
            if not kinds:
                continue
            if m["task_family"] == "adjacency" and len(kinds) > 1:
                bad.append((f.name, m["template_id"], q, sorted(kinds)))
            if m["template_id"] == "adj_01" and m["anchor_id"] in subtype and kinds != {subtype[m["anchor_id"]]}:
                bad.append((f.name, m["template_id"], q, sorted(kinds)))
            if re.search(r"\bstates?\b|union territor", q) and not re.search(r"\bdistricts?\b", q) and kinds != {"region"}:
                bad.append((f.name, m["template_id"], q, sorted(kinds)))
            if re.search(r"\bdistricts?\b", q) and not re.search(r"\bstates?\b|union territor|\bin\b|\bof\b|within|inside", q) and kinds != {"county"}:
                bad.append((f.name, m["template_id"], q, sorted(kinds)))
    assert not bad, bad[:10]
