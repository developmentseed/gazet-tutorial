"""Benchmark splits beyond the in-distribution test set.

The training set shares anchors and templates with its own test split, so a
fine-tuned small model looks near-perfect there. These helpers build the harder
splits that show what it actually learned:

- ``unseen_region``   whole states held out (state + every district)
- ``unseen_anchor``   individual districts held out
- ``noisy``           abbreviations, old names, typos, casing in the question
- ``compositional``   templates that combine two operators, never used in SFT
- ``paraphrase``      LLM rewrites of test questions (built offline by scripts/paraphrase.py)

Reference SQL for the compositional templates lives here, not in the notebook,
so it stays out of the training templates by construction.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import TABLE
from .india import HOLDOUT_STATES, replace_name

SPLITS = ["iid", "unseen_region", "unseen_anchor", "noisy", "paraphrase", "compositional"]
TASKS = ["sql", "places"]

_T = f"read_parquet('{TABLE}')"
_SELECT_B = f'SELECT b.id, b.names."primary" AS name, b.subtype, ST_AsGeoJSON(b.geometry) AS geometry FROM {_T} AS b'


@dataclass
class RefTemplate:
    template_id: str
    sql: str
    hints: list
    ordered: bool = False
    family: str = "compositional"


COMPOSITIONAL = [
    # containment ∩ adjacency: two anchors (a state and a neighbouring state)
    RefTemplate(
        template_id="adj_within",
        sql=(f"WITH a AS (SELECT geometry FROM {_T} WHERE id = '{{container_id}}'), "
             f"n AS (SELECT geometry FROM {_T} WHERE id = '{{neighbor_id}}') "
             f"{_SELECT_B}, a, n WHERE b.subtype = 'county' AND ST_Within(b.geometry, a.geometry) "
             f"AND ST_Touches(b.geometry, n.geometry)"),
        hints=["districts of {container_name} that border {neighbor_name}",
               "which districts in {container_name} touch {neighbor_name}?",
               "{container_name} districts along the {neighbor_name} border",
               "districts of {container_name} adjacent to {neighbor_name}",
               "list the districts of {container_name} next to {neighbor_name}"],
    ),
    # adjacency + ordering: the largest / smallest neighbour of a district
    RefTemplate(
        template_id="agg_adjacent", ordered=True,
        sql=(f"WITH a AS (SELECT geometry FROM {_T} WHERE id = '{{anchor_id}}') "
             f"SELECT b.id, b.names.\"primary\" AS name, b.subtype, ST_Area(b.geometry) AS area, "
             f"ST_AsGeoJSON(b.geometry) AS geometry FROM {_T} AS b, a "
             f"WHERE b.id != '{{anchor_id}}' AND b.subtype = 'county' AND ST_Touches(a.geometry, b.geometry) "
             f"ORDER BY area {{order}} LIMIT {{top_n}}"),
        hints=["the {superlative} district bordering {anchor_name}",
               "which is the {superlative} district next to {anchor_name}?",
               "{superlative} neighbour of {anchor_name}",
               "top {top_n} {superlative} districts bordering {anchor_name}",
               "{top_n} {superlative} districts that touch {anchor_name}"],
    ),
    # buffer + ordering: the largest districts within a radius
    RefTemplate(
        template_id="buffer_agg", ordered=True,
        sql=(f"WITH a AS (SELECT ST_Buffer(geometry, {{buffer_m}} / 111320.0) AS geom FROM {_T} WHERE id = '{{anchor_id}}') "
             f"SELECT b.id, b.names.\"primary\" AS name, b.subtype, ST_Area(b.geometry) AS area, "
             f"ST_AsGeoJSON(b.geometry) AS geometry FROM {_T} AS b, a "
             f"WHERE b.id != '{{anchor_id}}' AND b.subtype = 'county' AND ST_Intersects(b.geometry, a.geom) "
             f"ORDER BY area DESC LIMIT {{top_n}}"),
        hints=["{top_n} largest districts within {buffer_label} of {anchor_name}",
               "the largest district within {buffer_label} of {anchor_name}",
               "biggest districts no more than {buffer_label} from {anchor_name}, top {top_n}",
               "which are the {top_n} biggest districts within {buffer_label} of {anchor_name}?"],
    ),
]
COMPOSITIONAL_BY_ID = {t.template_id: t for t in COMPOSITIONAL}


# ── holdouts ────────────────────────────────────────────────────────────────

def holdout_ids(
    inv: pd.DataFrame,
    holdout_regions: dict[str, str] | None = None,
    n_random: int = 30,
    seed: int = 108,
    must_cover_duplicates: bool = True,
) -> tuple[set[str], set[str]]:
    """(unseen_region_ids, unseen_anchor_ids) drawn from the inventory.

    `unseen_region_ids` = the held-out states plus every district whose region
    code belongs to them. `unseen_anchor_ids` = `n_random` districts from the
    remaining states; when `must_cover_duplicates` is set, one member of every
    same-name district pair is included so disambiguation is tested on an
    anchor the model never trained on.
    """
    holdout_regions = HOLDOUT_STATES if holdout_regions is None else holdout_regions
    rng = random.Random(seed)
    region_ids = set(inv.loc[inv["region"].isin(holdout_regions), "id"])
    rest = inv[(inv["subtype"] == "county") & ~inv["id"].isin(region_ids)]
    chosen: list[str] = []
    if must_cover_duplicates:
        dup = rest.groupby("name").filter(lambda g: len(g) > 1)
        for _, grp in dup.sort_values("name").groupby("name", sort=True):
            ids = sorted(grp["id"].tolist())
            chosen.append(rng.choice(ids))
    pool = sorted(set(rest["id"]) - set(chosen))
    rng.shuffle(pool)
    chosen.extend(pool[: max(0, n_random - len(chosen))])
    return region_ids, set(chosen)


# ── noisy questions ─────────────────────────────────────────────────────────

NOISE_TYPES = ("alias", "abbreviation", "typo", "lowercase")


def typo(name: str, rng: random.Random) -> str:
    """One keyboard-style edit that keeps the first and last characters."""
    letters = [i for i, ch in enumerate(name) if ch.isalpha()]
    inner = [i for i in letters if 0 < i < len(name) - 1]
    if len(inner) < 2:
        return name
    kind = rng.choice(["swap", "drop", "double"])
    i = rng.choice(inner[:-1] if kind == "swap" else inner)
    s = list(name)
    if kind == "swap" and i + 1 < len(s) and s[i + 1].isalpha():
        s[i], s[i + 1] = s[i + 1], s[i]
    elif kind == "drop":
        del s[i]
    else:
        s.insert(i, s[i])
    out = "".join(s)
    return out if out != name else name[:i] + name[i + 1:]


def make_noisy(
    question: str,
    anchor_name: str,
    container_name: str | None,
    rng: random.Random,
    aliases: dict[str, list[str]],
    abbreviations: dict[str, list[str]],
    noise_type: str | None = None,
) -> dict | None:
    """Rewrite a question with one kind of real-world noise.

    Returns a dict with the new question, the noise type, the *search strings*
    an extractor would plausibly emit for it (what fuzzy search will receive)
    and the gold places. Gold places are the canonical names for alias and
    abbreviation noise (the model is expected to normalise), and the surface
    form for typos and casing (fuzzy search absorbs those).
    """
    options = []
    if anchor_name in aliases and replace_name(question, anchor_name, "x"):
        options.append("alias")
    for nm in (container_name, anchor_name):
        if nm and nm in abbreviations and replace_name(question, nm, "x"):
            options.append("abbreviation")
            break
    if replace_name(question, anchor_name, "x") and len(anchor_name) >= 5:
        options.append("typo")
    if question != question.lower():
        options.append("lowercase")
    if noise_type:
        options = [o for o in options if o == noise_type]
    if not options:
        return None
    kind = rng.choice(options)
    search = [anchor_name] + ([container_name] if container_name else [])
    places = list(search)

    if kind == "alias":
        alias = rng.choice(aliases[anchor_name])
        q = replace_name(question, anchor_name, alias)
        search[0] = alias
    elif kind == "abbreviation":
        target = container_name if (container_name and container_name in abbreviations and replace_name(question, container_name, "x")) else anchor_name
        abbr = rng.choice(abbreviations[target])
        q = replace_name(question, target, abbr)
        search = [abbr if s == target else s for s in search]
    elif kind == "typo":
        bad = typo(anchor_name, rng)
        q = replace_name(question, anchor_name, bad)
        search[0] = bad
        places[0] = bad
    else:
        q = question.lower()
        search = [s.lower() for s in search]
        places = [p.lower() for p in places]
    if not q or q == question:
        return None
    return {"question": q, "noise_type": kind, "search_strings": search, "places": places}


# ── fixed subset for the expensive baseline ─────────────────────────────────

def select_subset(rows: list[dict], n: int, seed: int = 108, key: str = "template_id") -> list[str]:
    """Deterministic, template-balanced pick of `n` sample ids from a split."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["metadata"].get(key, "?"), []).append(r["metadata"]["sample_id"])
    for ids in groups.values():
        rng.shuffle(ids)
    out: list[str] = []
    while len(out) < n and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


# ── IO ──────────────────────────────────────────────────────────────────────

def load_benchmark(bench_dir: str | Path) -> dict[str, dict[str, list[dict]]]:
    """{split: {task: rows}} for every split file that exists."""
    from .data import load_jsonl

    bench_dir = Path(bench_dir)
    out: dict[str, dict[str, list[dict]]] = {}
    for split in SPLITS:
        for task in TASKS:
            p = bench_dir / task / f"{split}.jsonl"
            if p.exists():
                out.setdefault(split, {})[task] = load_jsonl(p)
    return out


def load_subset_ids(bench_dir: str | Path) -> dict[str, dict[str, list[str]]]:
    p = Path(bench_dir) / "subset_ids.json"
    return json.loads(p.read_text()) if p.exists() else {}


def question_in_user(user_content: str) -> str:
    m = re.search(r"<USER_QUERY>\s*(.*?)\s*</USER_QUERY>", user_content, flags=re.S)
    return m.group(1) if m else user_content.strip()
