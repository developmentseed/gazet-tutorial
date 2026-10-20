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
