"""Engine behind the local playground: any OpenAI-compatible endpoint as a geocoder.

A model is an `Endpoint` (`llama serve`, Ollama, anything that speaks
`/v1/chat/completions`). The pipeline notebook 2 runs on the GPU is run here against
the endpoint: extract places → fuzzy search → write SQL → execute. A benchmark
question goes through the same pipeline and its result ids are compared with the
gold ids the benchmark stores.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .baseline import chat_completion
from .eval import gold_of, parse_places, question_of
from .search import candidates_csv, fetch_by_id, fuzzy_search, merge_candidates, per_place_limit
from .sqlutil import execute, result_ids, strip_fences

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

KINDS = ("qwen", "gpt-oss")
DEFAULT_PORTS = {"base": 8080, "finetuned": 8081, "gpt-oss": 8082}
RECORDED = "gpt-oss:20b (recorded)"     # the committed zero-shot predictions, replayed without a model


@dataclass
class Endpoint:
    name: str
    base_url: str = ""            # OpenAI-compatible server (…/v1)
    model: str = "default"        # llama serve ignores it; Ollama needs the tag
    kind: str = "qwen"            # qwen: non-thinking chat template · gpt-oss: low reasoning
    enabled: bool = True

    @property
    def max_tokens(self) -> dict[str, int]:
        # gpt-oss spends tokens on hidden reasoning before the answer
        return {"sql": 2048, "places": 512} if self.kind == "gpt-oss" else {"sql": 256, "places": 64}

    def extra_body(self) -> dict:
        if self.kind == "qwen":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if self.kind == "gpt-oss":
            return {"chat_template_kwargs": {"reasoning_effort": "low"}}
        return {}

    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")


def default_endpoints(served: str | Path | None = None) -> list[Endpoint]:
    """The servers docs/take-it-home.md starts; edit the URL/model for Ollama.

    `served` is the JSON list scripts/serve_models.sh writes ([{"name", "port"}, ...]): one row per
    fine-tuned GGUF it started, in place of the single default finetuned row."""
    base = Endpoint("base Qwen3.5-0.8B", f"http://127.0.0.1:{DEFAULT_PORTS['base']}/v1", kind="qwen")
    gpt = Endpoint("gpt-oss:20b", f"http://127.0.0.1:{DEFAULT_PORTS['gpt-oss']}/v1", kind="gpt-oss")
    rows = []
    if served and Path(served).exists():
        rows = [Endpoint(r["name"], f"http://127.0.0.1:{int(r['port'])}/v1", kind="qwen") for r in json.loads(Path(served).read_text())]
    if not rows:
        rows = [Endpoint("finetuned Qwen3.5-0.8B", f"http://127.0.0.1:{DEFAULT_PORTS['finetuned']}/v1", kind="qwen")]
    return [base, *rows, gpt]


def endpoints_frame(eps: list[Endpoint]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in eps])


def endpoints_from_frame(df: pd.DataFrame) -> list[Endpoint]:
    out = []
    for r in df.to_dict(orient="records"):
        r = {k: ("" if v is None or (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.items()}
        r["enabled"] = bool(r.get("enabled", True))
        out.append(Endpoint(**{k: r[k] for k in Endpoint.__dataclass_fields__ if k in r}))
    return out


def is_up(ep: Endpoint, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(ep.base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def chat(ep: Endpoint, messages: list[dict], task: str) -> dict:
    return chat_completion(messages, model=ep.model, base_url=ep.base_url, max_tokens=ep.max_tokens[task],
                           reasoning_effort=None, extra_body=ep.extra_body(), retries=1)


# ── free-text geocoding ─────────────────────────────────────────────────────

def geocode(ep: Endpoint, question: str, con: duckdb.DuckDBPyConnection, geodata_path: str | Path, prompts: dict) -> dict:
    """The gazet pipeline against one endpoint; returns everything the UI shows."""
    t0 = time.time()
    out = chat(ep, [{"role": "system", "content": prompts["place_system"]}, {"role": "user", "content": question}], "places")
    places = parse_places(out["content"]) or [question]
    t_places = time.time() - t0
    limit = per_place_limit(len(places))
    candidates = merge_candidates([fuzzy_search(con, geodata_path, p, limit=limit) for p in places])
    user = prompts["user_template"].format(candidates=candidates_csv(candidates), question=question)
    t1 = time.time()
    out_sql = chat(ep, [{"role": "system", "content": prompts["sql_system"]}, {"role": "user", "content": user}], "sql")
    sql = strip_fences(out_sql["content"])
    t_sql = time.time() - t1
    try:
        result = execute(con, sql, geodata_path)
        error = None
    except Exception as exc:
        result, error = pd.DataFrame(), str(exc).splitlines()[0]
    ids = list(dict.fromkeys(_UUID.findall(sql)))
    anchors = pd.concat([fetch_by_id(con, geodata_path, i, geometry=True) for i in ids], ignore_index=True) if ids else pd.DataFrame()
    return {"model": ep.name, "question": question, "places": places, "places_raw": out["content"],
            "candidates": candidates, "prompt_user": user, "sql": sql, "sql_raw": out_sql["content"], "result": result, "anchors": anchors,
            "error": error, "seconds": {"places": round(t_places, 2), "sql": round(t_sql, 2)},
            "reasoning": {"places": out.get("reasoning", ""), "sql": out_sql.get("reasoning", "")}}


# ── benchmark questions ─────────────────────────────────────────────────────

def score(r: dict, gold_ids: list[str], ordered: bool = False) -> dict:
    """Did a `geocode()` run return the benchmark's gold features? (execution match)"""
    if r["error"]:
        return {"match": False, "reason": r["error"]}
    df = r["result"]
    if len(df) == 0:
        return {"match": False, "reason": "0 rows"}
    if "id" not in df.columns:
        return {"match": False, "reason": "no id column"}
    got = result_ids(df, ordered)
    want = [str(g) for g in gold_ids] if ordered else sorted({str(g) for g in gold_ids})
    if got == want:
        return {"match": True, "reason": ""}
    missing, extra = len(set(want) - set(got)), len(set(got) - set(want))
    what = "order differs" if set(got) == set(want) else f"{missing} missing, {extra} extra"
    return {"match": False, "reason": f"ids differ ({what})"}


def candidates_of(sample: dict) -> pd.DataFrame:
    """The candidate table inside a SQL sample's user prompt, as the model saw it."""
    user = sample["messages"][-2]["content"]
    if "<CANDIDATES>" not in user:
        return pd.DataFrame(columns=["id", "name", "subtype", "region"])
    csv = user.split("<CANDIDATES>")[-1].split("</CANDIDATES>")[0].strip()
    return pd.read_csv(io.StringIO(csv), dtype=str, keep_default_na=False)


def benchmark_rows(bench: dict, split: str) -> pd.DataFrame:
    """One row per SQL benchmark question: the question, the prompt it was asked with and the gold answer."""
    places = {}
    for s in bench[split].get("places", []):
        places[s["metadata"]["sample_id"]] = parse_places(gold_of(s)) or []
    recs = []
    for s in bench[split]["sql"]:
        m = s["metadata"]
        recs.append({"sample_id": m["sample_id"], "question": question_of(s), "task_family": m.get("task_family"),
                     "template_id": m.get("template_id"), "noise_type": m.get("noise_type"), "anchor_name": m.get("anchor_name"),
                     "anchor_id": m.get("anchor_id"), "gold_ids": list(m.get("gold_ids", [])), "ordered": bool(m.get("ordered", False)),
                     "gold_places": places.get(m["sample_id"], []), "gold_candidates": candidates_of(s), "gold_sql": gold_of(s),
                     "prompt_user": s["messages"][-2]["content"]})
    cols = ["sample_id", "question", "task_family", "template_id", "noise_type", "anchor_name", "anchor_id", "gold_ids", "ordered",
            "gold_places", "gold_candidates", "gold_sql", "prompt_user"]
    return pd.DataFrame(recs, columns=cols)


def _norm(names) -> list[str]:
    return sorted({str(n).strip().lower() for n in names})


def places_match(pred: list[str], gold: list[str]) -> bool:
    """Same set of place names, ignoring case and whitespace."""
    return _norm(pred) == _norm(gold)


def mark_candidates(cands: pd.DataFrame, gold_ids: list[str], anchor_id: str | None = None) -> pd.DataFrame:
    """Candidate table with a `gold` column: 'anchor' for the place asked about, 'gold' for an answer row."""
    out = cands.copy()
    gold = {str(g) for g in gold_ids}
    out.insert(0, "gold", ["anchor" if str(i) == anchor_id else ("gold" if str(i) in gold else "") for i in out["id"]])
    return out


# ── recorded predictions (no model needed) ──────────────────────────────────

def load_recorded(baseline_dir: str | Path, split: str) -> dict[str, dict]:
    """{sample_id: {"sql": record, "places": record}} from a baseline folder's `{task}_{split}.jsonl` files."""
    out: dict[str, dict] = {}
    for task in ("sql", "places"):
        path = Path(baseline_dir) / f"{task}_{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rec["sample_id"], {})[task] = rec
    return out


def replay(row: dict, rec: dict, con: duckdb.DuckDBPyConnection, geodata_path: str | Path, model: str = RECORDED) -> dict:
    """A recorded prediction in the shape `geocode()` returns, so the UI can show it the same way.

    The recorded run answered the benchmark prompt, so its candidates are the benchmark's
    (`row["gold_candidates"]`), not a fresh fuzzy search.
    """
    places_rec, sql_rec = rec.get("places"), rec.get("sql")
    places = (parse_places(places_rec["pred"]) if places_rec else None) or []
    sql = strip_fences(sql_rec["pred"]) if sql_rec else ""
    try:
        result = execute(con, sql, geodata_path) if sql else pd.DataFrame()
        error = None if sql else "no recorded SQL prediction"
    except Exception as exc:
        result, error = pd.DataFrame(), str(exc).splitlines()[0]
    ids = list(dict.fromkeys(_UUID.findall(sql)))
    anchors = pd.concat([fetch_by_id(con, geodata_path, i, geometry=True) for i in ids], ignore_index=True) if ids else pd.DataFrame()
    return {"model": model, "question": row["question"], "places": places, "places_raw": places_rec["pred"] if places_rec else "",
            "candidates": row["gold_candidates"], "prompt_user": row["prompt_user"], "sql": sql,
            "sql_raw": sql_rec["pred"] if sql_rec else "", "result": result, "anchors": anchors, "error": error,
            "seconds": {"places": (places_rec or {}).get("latency_s", 0), "sql": (sql_rec or {}).get("latency_s", 0)},
            "reasoning": {"places": "", "sql": ""}, "recorded": True}
