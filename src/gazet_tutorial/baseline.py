"""Zero-shot baseline against any OpenAI-compatible chat endpoint.

Used offline (see scripts/run_baseline.py) to score gpt-oss:20b, the model
gazet runs in production, on the *same prompts and candidate tables* the
fine-tuned 0.8B model sees. Predictions are stored as JSONL and scored in the
notebooks with the same execution-accuracy code; nothing here runs in Colab.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # local Ollama (proxies "*-cloud" models when signed in)
DEFAULT_MODEL = "gpt-oss:20b-cloud"


def chat_completion(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    reasoning_effort: str | None = "low",
    timeout: float = 180.0,
    api_key: str | None = None,
    retries: int = 3,
    extra_body: dict | None = None,
) -> dict:
    """One chat completion; returns content, latency and token counts.

    `extra_body` is merged into the request, e.g. llama.cpp's
    ``{"chat_template_kwargs": {"enable_thinking": False}}``.
    """
    body: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if extra_body:
        body.update(extra_body)
    headers = {"Content-Type": "application/json"}
    api_key = api_key or os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            latency = time.time() - t0
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            msg = choice.get("message") or {}
            return {
                "content": msg.get("content") or "",
                # gpt-oss returns its chain of thought separately; keep it for inspection.
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"),
                "latency_s": round(latency, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "model": data.get("model", model),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code in (400, 401, 403, 404):
                break
        except Exception as exc:  # network blips, timeouts
            last_exc = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat completion failed after {retries} attempts: {last_exc}")


def prompt_of(row: dict) -> list[dict]:
    return [m for m in row["messages"] if m["role"] != "assistant"]


def run_rows(
    rows: list[dict],
    out_path: str | Path,
    task: str,
    split: str,
    max_tokens: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> list[dict]:
    """Query the endpoint for every row not already in `out_path` (resumable)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["sample_id"]] = rec
    # reasoning tokens count against max_tokens, so leave generous room
    max_tokens = max_tokens or (2048 if task == "sql" else 512)
    results = list(done.values())
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            sid = row["metadata"]["sample_id"]
            if sid in done:
                continue
            res = chat_completion(prompt_of(row), max_tokens=max_tokens, **kwargs)
            rec = {"sample_id": sid, "task": task, "split": split, "pred": res["content"],
                   **{k: res[k] for k in ("latency_s", "prompt_tokens", "completion_tokens", "finish_reason", "model")},
                   "reasoning_chars": len(res.get("reasoning", ""))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            results.append(rec)
            if verbose:
                print(f"[{task}/{split}] {len(results)}/{len(rows)}  {res['latency_s']}s  {res['content'][:70]!r}")
    return results


def load_predictions(path: str | Path) -> dict[str, str]:
    """{sample_id: prediction} from a baseline JSONL (empty if missing)."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["sample_id"]] = rec["pred"]
    return out


def align(rows: list[dict], preds: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Rows that have a prediction, with predictions in the same order."""
    kept = [r for r in rows if r["metadata"]["sample_id"] in preds]
    return kept, [preds[r["metadata"]["sample_id"]] for r in kept]


# ── scoring prediction folders (scripts/benchmark_gguf.py) ──────────────────

def score_predictions(
    bench: dict,
    subset: dict,
    pred_dir: str | Path,
    con,
    geodata_path: str | Path,
    splits: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict]:
    """{split: {task: scored DataFrame}} for the predictions in `pred_dir`.

    Only rows in `subset` (the fixed 40 SQL + 20 extraction questions per split that
    gpt-oss was scored on) are kept, so every model is compared on the same questions.
    Scoring is execution accuracy for SQL and set P/R/F1 for extraction, the same code
    the experiments use.
    """
    from .benchmark import SPLITS, TASKS
    from .eval import evaluate_places, evaluate_sql

    pred_dir = Path(pred_dir)
    out: dict[str, dict] = {}
    for split in splits or SPLITS:
        for task in tasks or TASKS:
            rows = bench.get(split, {}).get(task)
            if not rows:
                continue
            ids = subset.get(split, {}).get(task)
            if ids is not None:
                keep = set(ids)
                rows = [r for r in rows if r["metadata"]["sample_id"] in keep]
            rows, preds = align(rows, load_predictions(pred_dir / f"{task}_{split}.jsonl"))
            if not rows:
                continue
            df = evaluate_sql(rows, preds, con, geodata_path) if task == "sql" else evaluate_places(rows, preds)
            out.setdefault(split, {})[task] = df
    return out


def headline(results: dict[str, dict], splits: list[str] | None = None):
    """(SQL execution accuracy, extraction F1) tables: one row per model, one column per split.

    `results` is {model_name: score_predictions(...)}. The `rows` column counts the
    questions scored, so a partial run (``--limit``) is visible next to a full one.
    """
    import pandas as pd

    from .benchmark import SPLITS

    def table(task: str, metric: str) -> pd.DataFrame:
        cells: dict[str, dict] = {}
        for model_name, per_split in results.items():
            for split, per_task in per_split.items():
                df = per_task.get(task)
                if df is not None and len(df):
                    cells.setdefault(model_name, {})[split] = df[metric].mean()
                    cells[model_name]["rows"] = cells[model_name].get("rows", 0) + len(df)
        out = pd.DataFrame(cells).T
        cols = [s for s in (splits or SPLITS) if s in out.columns]
        out = out.reindex(columns=cols + ["rows"])
        out.insert(len(cols), "mean", out[cols].mean(axis=1))
        out["rows"] = out["rows"].astype(int)
        return out

    return table("sql", "exec_match"), table("places", "f1")
