"""Evaluation: execution accuracy for SQL, set/F1 for place extraction.

Generation is separated from scoring so the scoring functions can be tested
with canned predictions and reused for any model backend.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from .sqlutil import execute, normalize_ws, result_ids, strip_fences

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def question_of(sample: dict) -> str:
    """The natural-language question inside a chat sample (either task)."""
    user = sample["messages"][-2]["content"] if sample["messages"][-1]["role"] == "assistant" else sample["messages"][-1]["content"]
    if "<USER_QUERY>" in user:
        return user.split("<USER_QUERY>")[-1].split("</USER_QUERY>")[0].strip()
    return user.strip()


def prompt_messages(sample: dict) -> list[dict]:
    """Messages without the assistant turn, ready for generation."""
    return [m for m in sample["messages"] if m["role"] != "assistant"]


def gold_of(sample: dict) -> str:
    return sample["messages"][-1]["content"]


# ── generation ──────────────────────────────────────────────────────────────

def generate_batch(
    model,
    tokenizer,
    messages_list: list[list[dict]],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    enable_thinking: bool = False,
    progress: bool = True,
) -> list[str]:
    """Greedy batched generation with a HF/Unsloth causal LM and chat template."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[str] = []
    rng = range(0, len(messages_list), batch_size)
    try:
        from tqdm.auto import tqdm
        rng = tqdm(rng, disable=not progress, desc="generate")
    except ImportError:
        pass
    for start in rng:
        batch = messages_list[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            for m in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


# ── SQL scoring ─────────────────────────────────────────────────────────────

def execution_match(
    con: duckdb.DuckDBPyConnection,
    pred_sql: str,
    gold_sql: str,
    path: str | Path,
    ordered: bool = False,
) -> dict:
    """Run both queries; match when they return the same feature ids.

    `ordered=True` compares id order too (for top-N aggregation queries).
    """
    out = {"sql_valid": False, "sql_nonempty": False, "exec_match": False, "error": ""}
    try:
        gold_df = execute(con, gold_sql, path)
    except Exception as exc:
        out["error"] = f"gold failed: {exc}"
        return out
    gold_ids = result_ids(gold_df, ordered=ordered)
    pred_sql = strip_fences(pred_sql)
    if not pred_sql:
        out["error"] = "empty prediction"
        return out
    try:
        pred_df = execute(con, pred_sql, path)
    except Exception as exc:
        out["error"] = str(exc).splitlines()[0][:200]
        return out
    out["sql_valid"] = True
    out["sql_nonempty"] = len(pred_df) > 0
    if "id" in pred_df.columns:
        out["exec_match"] = result_ids(pred_df, ordered=ordered) == gold_ids and bool(gold_ids)
    elif "geometry" in pred_df.columns and "geometry" in gold_df.columns:
        # No id column (a zero-shot model often returns only the geometry): fall back
        # to comparing the returned geometries themselves.
        out["exec_match"] = _geometry_keys(pred_df, ordered) == _geometry_keys(gold_df, ordered) and len(gold_df) > 0
        out["matched_on"] = "geometry"
    else:
        out["error"] = "no id or geometry column in result"
        return out
    if not out["exec_match"] and not out["error"]:
        out["error"] = "empty result" if not out["sql_nonempty"] else "wrong feature set"
    return out


def _geometry_keys(df: pd.DataFrame, ordered: bool):
    keys = [hashlib.md5(str(g).encode()).hexdigest() for g in df["geometry"].tolist()]
    return keys if ordered else set(keys)


def anchor_grounded(pred_sql: str, gold_sql: str) -> bool:
    """Did the prediction copy every id the gold SQL filters on?"""
    gold_ids = set(_UUID.findall(gold_sql))
    return bool(gold_ids) and gold_ids.issubset(set(_UUID.findall(pred_sql)))


def evaluate_sql(
    samples: list[dict],
    predictions: list[str],
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    ordered_families: Iterable[str] = ("aggregation",),
) -> pd.DataFrame:
    rows = []
    ordered_families = set(ordered_families)
    for i, (s, raw) in enumerate(zip(samples, predictions)):
        meta = s.get("metadata", {})
        family = meta.get("task_family", "unknown")
        gold = gold_of(s)
        pred = strip_fences(raw)
        ordered = bool(meta.get("ordered", family in ordered_families))
        res = execution_match(con, pred, gold, path, ordered=ordered)
        rows.append(
            {
                "index": i,
                "task_family": family,
                "template_id": meta.get("template_id"),
                "sample_id": meta.get("sample_id"),
                "question": question_of(s),
                "gold": gold,
                "pred": pred,
                "sql_valid": res["sql_valid"],
                "sql_nonempty": res["sql_nonempty"],
                "exec_match": res["exec_match"],
                "exact_match": normalize_ws(pred) == normalize_ws(gold),
                "anchor_grounded": anchor_grounded(pred, gold),
                "error": res["error"],
            }
        )
    return pd.DataFrame(rows)


# ── places scoring ──────────────────────────────────────────────────────────

def parse_places(text: str) -> list[str] | None:
    """Parse {"places":[{"place":...}]} → list of names; None when not valid JSON."""
    cleaned = strip_fences(text)
    try:
        data = json.loads(cleaned)
        return [str(p["place"]) for p in data["places"]]
    except Exception:
        return None


def _norm(names: Iterable[str]) -> set[str]:
    return {n.strip().lower() for n in names if n and n.strip()}


def places_prf(pred_text: str, gold_text: str) -> dict:
    gold = _norm(parse_places(gold_text) or [])
    pred_list = parse_places(pred_text)
    pred = _norm(pred_list or [])
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "json_valid": pred_list is not None,
        "exact_set": pred == gold and bool(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_places(samples: list[dict], predictions: list[str]) -> pd.DataFrame:
    rows = []
    for i, (s, raw) in enumerate(zip(samples, predictions)):
        meta = s.get("metadata", {})
        gold = gold_of(s)
        scores = places_prf(raw, gold)
        rows.append(
            {
                "index": i,
                "task_family": meta.get("task_family", "unknown"),
                "template_id": meta.get("template_id"),
                "sample_id": meta.get("sample_id"),
                "question": question_of(s),
                "gold": gold,
                "pred": strip_fences(raw),
                **scores,
            }
        )
    return pd.DataFrame(rows)


# ── aggregation ─────────────────────────────────────────────────────────────

SQL_METRICS = ["sql_valid", "sql_nonempty", "exec_match", "exact_match", "anchor_grounded"]
PLACES_METRICS = ["json_valid", "exact_set", "precision", "recall", "f1"]


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean of every boolean/float metric column, overall or grouped."""
    metrics = [c for c in SQL_METRICS + PLACES_METRICS if c in df.columns]
    if by:
        out = df.groupby(by)[metrics].mean()
        out.insert(0, "n", df.groupby(by).size())
        return out
    out = df[metrics].mean().to_frame("value").T
    out.insert(0, "n", len(df))
    return out


def compare(results: dict[str, pd.DataFrame], by: str | None = None) -> pd.DataFrame:
    """Side-by-side summary for several models, e.g. {"base": df1, "finetuned": df2}."""
    parts = []
    for label, df in results.items():
        s = summarize(df, by=by)
        s.insert(0, "model", label)
        parts.append(s)
    out = pd.concat(parts)
    return out.set_index("model", append=bool(by)) if by else out.set_index("model")


def failures(df: pd.DataFrame, metric: str = "exec_match", n: int = 5) -> pd.DataFrame:
    """Worst cases to show in an error gallery."""
    cols = [c for c in ("task_family", "question", "error", "gold", "pred") if c in df.columns]
    return df.loc[~df[metric].astype(bool), cols].head(n)
