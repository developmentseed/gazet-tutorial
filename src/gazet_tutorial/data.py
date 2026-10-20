"""Dataset IO, splitting and token statistics.

Ported from gazet's dataset/scripts/export_training_data.py (stratified_split,
_shuffle_candidates_for_export) and finetune/check_token_lengths.py.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd


def save_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    key: Callable[[dict], str] = lambda s: s["metadata"]["template_id"],
) -> tuple[list[dict], list[dict], list[dict]]:
    """80/10/10 split stratified by template_id so every template is in every split."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)

    train, val, test = [], [], []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def shuffle_candidates(sample: dict) -> dict:
    """Copy of sample with candidate rows shuffled deterministically (seeded by sample id).

    Removes the positional shortcut where the true anchor is always row 1.
    Candidate ids are unchanged, so selected_candidates stay valid.
    """
    out = copy.deepcopy(sample)
    cands = out.get("candidates", [])
    if len(cands) > 1:
        random.Random(out.get("id", "sample")).shuffle(cands)
    return out


def count_by(records: list[dict], field: str = "task_family") -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.get("metadata", {}).get(field, "unknown")] += 1
    return dict(sorted(counts.items()))


def token_length_stats(rows: list[dict], tokenizer, label: str = "") -> pd.Series:
    """Token lengths of fully rendered chat samples (system+user+assistant)."""
    lengths = []
    for r in rows:
        ids = tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        lengths.append(len(ids))
    s = pd.Series(lengths, name=label or "tokens")
    return s.describe(percentiles=[0.5, 0.95, 0.99])


def label_mask_preview(tokenizer, input_ids, labels, ignore_index: int = -100) -> str:
    """Decode only the tokens that contribute to the loss (labels != ignore_index)."""
    kept = [tok for tok, lab in zip(input_ids, labels) if lab != ignore_index]
    return tokenizer.decode(kept)
