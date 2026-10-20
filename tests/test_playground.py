import json

import pandas as pd

import gazet_tutorial.playground as pg
from gazet_tutorial.playground import (
    Endpoint,
    benchmark_rows,
    default_endpoints,
    endpoints_frame,
    endpoints_from_frame,
    geocode,
    mark_candidates,
    places_match,
    replay,
    score,
)

PROMPTS = {"place_system": "P", "sql_system": "S",
           "user_template": "<CANDIDATES>\n{candidates}\n</CANDIDATES>\n\n<USER_QUERY>\n{question}\n</USER_QUERY>"}


def fake_chat_factory(sql_by_question: dict):
    calls = []

    def fake(messages, **kw):
        calls.append(kw)
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == "P":
            name = user.replace("show me ", "").strip()
            content = json.dumps({"places": [{"place": name}]})
        else:
            q = user.split("<USER_QUERY>")[-1].split("</USER_QUERY>")[0].strip()
            content = sql_by_question.get(q, "SELECT nothing")
        return {"content": content, "reasoning": "", "latency_s": 0.01, "prompt_tokens": 1, "completion_tokens": 1,
                "finish_reason": "stop", "model": "fake"}
    return fake, calls


def test_endpoint_defaults_roundtrip(tmp_path):
    eps = default_endpoints()
    assert len(eps) == 3 and [e.kind for e in eps] == ["qwen", "qwen", "gpt-oss"]
    served = tmp_path / "served.json"
    served.write_text(json.dumps([{"name": "reference-q8_0", "port": 8081}, {"name": "short-q8_0", "port": 8083}]))
    multi = default_endpoints(served)
    assert [e.name for e in multi][1:3] == ["reference-q8_0", "short-q8_0"] and multi[2].base_url.endswith(":8083/v1")
    assert len(default_endpoints(tmp_path / "missing.json")) == 3
    assert eps[0].extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}
    assert Endpoint("g", kind="gpt-oss").max_tokens["sql"] > Endpoint("q").max_tokens["sql"]
    back = endpoints_from_frame(endpoints_frame(eps))
    assert [e.name for e in back] == [e.name for e in eps] and all(e.enabled for e in back)
    assert Endpoint("gpt-oss:20b (llama)").slug() == "gpt-oss-20b-llama"
    assert not pg.is_up(Endpoint("dead", "http://127.0.0.1:1/v1"))


def test_geocode_pipeline(con, divisions_path, monkeypatch):
    sql = "SELECT id, names.\"primary\" AS name, ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Beta'"
    fake, calls = fake_chat_factory({"show me Beta": sql})
    monkeypatch.setattr(pg, "chat_completion", fake)
    r = geocode(Endpoint("m", "http://x/v1"), "show me Beta", con, divisions_path, PROMPTS)
    assert r["places"] == ["Beta"] and r["error"] is None and r["result"]["name"].tolist() == ["Beta"]
    assert "Beta" in r["candidates"]["name"].tolist() and len(calls) == 2
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "<CANDIDATES>" in r["prompt_user"] and "show me Beta" in r["prompt_user"]
    beta = r["result"]["id"].iloc[0]
    marked = mark_candidates(r["candidates"], [beta], anchor_id=beta)
    assert marked.columns[0] == "gold" and marked.loc[marked.id == beta, "gold"].iloc[0] == "anchor"
    assert places_match(["beta "], ["Beta"]) and not places_match(["Beta"], ["Beta", "Alpha"])
    assert score(r, [beta]) == {"match": True, "reason": ""}
    assert score(r, [beta, "other"])["reason"].startswith("ids differ (1 missing")
    assert score({"error": "Parser Error", "result": pd.DataFrame()}, [beta])["reason"] == "Parser Error"
    assert score({"error": None, "result": pd.DataFrame({"name": ["x"]})}, [beta])["reason"] == "no id column"
    assert score({"error": None, "result": pd.DataFrame({"id": []})}, [beta])["reason"] == "0 rows"
    two = {"error": None, "result": pd.DataFrame({"id": ["a", "b"]})}
    assert score(two, ["b", "a"])["match"] and score(two, ["b", "a"], ordered=True)["reason"] == "ids differ (order differs)"


def test_benchmark_rows_columns():
    csv = "id,name,subtype,region\na,Alpha,county,X-A\nb,Beta,county,X-B"

    def sample(i, **meta):
        return {"messages": [{"role": "system", "content": "S"},
                             {"role": "user", "content": PROMPTS["user_template"].format(candidates=csv, question=f"q{i}")},
                             {"role": "assistant", "content": "SELECT 1"}],
                "metadata": {"sample_id": f"s{i}", "task_family": "adjacency", "template_id": "adj_01", "gold_ids": ["a"], **meta}}
    place = {"messages": [{"role": "system", "content": "P"}, {"role": "user", "content": "q0"},
                          {"role": "assistant", "content": json.dumps({"places": [{"place": "Alpha"}]})}], "metadata": {"sample_id": "s0"}}
    bench = {"noisy": {"sql": [sample(0, noise_type="alias"), sample(1, ordered=True)], "places": [place]}}
    df = benchmark_rows(bench, "noisy")
    assert df.question.tolist() == ["q0", "q1"] and df.noise_type.tolist()[0] == "alias" and pd.isna(df.noise_type[1])
    assert df.ordered.tolist() == [False, True] and df.gold_ids[0] == ["a"] and df.gold_sql[0] == "SELECT 1"
    assert df.gold_places.tolist() == [["Alpha"], []]
    assert df.gold_candidates[0].name.tolist() == ["Alpha", "Beta"] and list(df.gold_candidates[0].columns) == ["id", "name", "subtype", "region"]
    assert mark_candidates(df.gold_candidates[0], ["a"]).gold.tolist() == ["gold", ""]
    assert benchmark_rows({"iid": {"sql": [], "places": []}}, "iid").empty


def test_replay_recorded_prediction(con, divisions_path, tmp_path):
    gold = "SELECT id, names.\"primary\" AS name, ST_AsGeoJSON(geometry) AS geometry FROM read_parquet('divisions_area') WHERE names.\"primary\" = 'Beta'"
    beta = con.execute(f"SELECT id FROM read_parquet('{divisions_path}') WHERE names.\"primary\" = 'Beta'").fetchone()[0]
    (tmp_path / "sql_iid.jsonl").write_text(json.dumps({"sample_id": "r0", "pred": f"```sql\n{gold}\n```", "latency_s": 0.5}) + "\n")
    (tmp_path / "places_iid.jsonl").write_text(json.dumps({"sample_id": "r0", "pred": json.dumps({"places": [{"place": "Beta"}]}), "latency_s": 0.2}) + "\n")
    rec = pg.load_recorded(tmp_path, "iid")
    assert set(rec) == {"r0"} and set(rec["r0"]) == {"sql", "places"} and pg.load_recorded(tmp_path, "noisy") == {}
    row = {"question": "show me Beta", "gold_candidates": pd.DataFrame({"id": [beta], "name": ["Beta"]}), "prompt_user": "U",
           "gold_ids": [beta], "ordered": False}
    r = replay(row, rec["r0"], con, divisions_path)
    assert r["recorded"] and r["places"] == ["Beta"] and r["error"] is None and r["seconds"] == {"places": 0.2, "sql": 0.5}
    assert r["result"]["name"].tolist() == ["Beta"] and score(r, [beta])["match"]
    assert replay(row, {"places": rec["r0"]["places"]}, con, divisions_path)["error"] == "no recorded SQL prediction"
