"""Preference pairs from execution feedback (notebook 3, scripts/mine_pairs.py).

A pair is a prompt the model answers *inconsistently*: sample it several times, run
every answer, and when some pass and some fail, the passing one is `chosen` and a
failing one is `rejected`. DPO then raises the margin between the two. Prompts the
model always gets right or always gets wrong yield nothing, which is the whole point:
preference tuning sharpens what a model can already do; it does not add knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path

RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"   # non-thinking Qwen3.5 reply, as in SFT


def choose_pair(samples: list[str], verdicts: list[dict]) -> tuple[str, str] | None:
    """(chosen, rejected) from K sampled answers and their execution verdicts, or None.

    `verdicts[i]` has a boolean ``pass`` and, for SQL, ``sql_valid``. Chosen is the
    shortest passing answer (no reason to prefer verbosity); rejected is a failing
    answer that is still valid SQL when one exists, because "right shape, wrong rows"
    teaches more than a syntax error. Identical strings never form a pair.
    """
    passing = [s for s, v in zip(samples, verdicts) if v["pass"]]
    failing = [(s, v) for s, v in zip(samples, verdicts) if not v["pass"]]
    if not passing or not failing:
        return None
    chosen = min(passing, key=len)
    failing.sort(key=lambda sv: (not sv[1].get("sql_valid", True), len(sv[0])))
    for s, _ in failing:
        if s.strip() != chosen.strip():
            return chosen, s
    return None


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """The prompt string exactly as inference sees it, ending in the non-thinking marker.

    Qwen3.5's template with ``enable_thinking=False`` already appends the empty think
    block; a template that stops at ``<|im_start|>assistant\n`` gets it appended here so
    DPO trains on the same prefix SFT did.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head = "<|im_start|>assistant\n"
    if not text.endswith(RESPONSE_PART) and text.endswith(head):
        text += RESPONSE_PART[len(head):]
    return text


def load_pairs(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
