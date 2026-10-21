import json

from gazet_tutorial.baseline import align, load_predictions, prompt_of, run_rows


def test_run_rows_resumable_and_align(tmp_path, monkeypatch):
    import gazet_tutorial.baseline as b

    calls = []

    def fake(messages, **kw):
        calls.append(messages)
        return {"content": "SELECT 1", "finish_reason": "stop", "latency_s": 0.1,
                "prompt_tokens": 10, "completion_tokens": 2, "model": "fake"}

    monkeypatch.setattr(b, "chat_completion", fake)
    rows = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": "gold"}],
             "metadata": {"sample_id": f"id{i}"}} for i in range(3)]
    out = tmp_path / "sql_iid.jsonl"
    res = run_rows(rows, out, task="sql", split="iid", verbose=False)
    assert len(res) == 3 and len(calls) == 3
    assert prompt_of(rows[0]) == rows[0]["messages"][:2]
    # resume: nothing re-queried
    res = run_rows(rows, out, task="sql", split="iid", verbose=False)
    assert len(res) == 3 and len(calls) == 3
    preds = load_predictions(out)
    assert preds == {"id0": "SELECT 1", "id1": "SELECT 1", "id2": "SELECT 1"}
    kept, p = align(rows[1:] + [{"messages": [], "metadata": {"sample_id": "zz"}}], preds)
    assert [r["metadata"]["sample_id"] for r in kept] == ["id1", "id2"] and p == ["SELECT 1"] * 2
    assert json.loads(out.read_text().splitlines()[0])["task"] == "sql"


def test_score_predictions_subset_and_headline(tmp_path):
    from gazet_tutorial.baseline import headline, score_predictions

    def row(sid, gold):
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                             {"role": "assistant", "content": gold}],
                "metadata": {"sample_id": sid, "task_family": "direct_lookup", "template_id": "lookup_01"}}

    gold = '{"places": [{"place": "Nagpur"}]}'
    bench = {"iid": {"places": [row("a", gold), row("b", gold), row("c", gold)]}}
    subset = {"iid": {"places": ["a", "b"]}}          # c is outside the subset gpt-oss was scored on
    pred_dir = tmp_path / "m"
    pred_dir.mkdir()
    (pred_dir / "places_iid.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"sample_id": "a", "pred": gold},
        {"sample_id": "b", "pred": '{"places": [{"place": "Pune"}]}'},
        {"sample_id": "c", "pred": gold},
    ]))
    res = score_predictions(bench, subset, pred_dir, con=None, geodata_path=None, tasks=["places"])
    df = res["iid"]["places"]
    assert df["sample_id"].tolist() == ["a", "b"]      # subset only, benchmark order
    assert df["f1"].tolist() == [1.0, 0.0]

    sql_table, places_table = headline({"m": res, "other": {"iid": {"places": df.iloc[:1]}}})
    assert sql_table.empty
    assert places_table.loc["m", "iid"] == 0.5 and places_table.loc["m", "rows"] == 2
    assert places_table.loc["other", "iid"] == 1.0 and places_table.loc["other", "rows"] == 1
    assert list(places_table.columns) == ["iid", "mean", "rows"]
