# %% [markdown]
# # Notebook 1 · From a gazetteer to a training set (no LLM required)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/01_data_and_dataset.ipynb)
#
# This is part 1 of a two-notebook tutorial that teaches you to fine-tune a small
# language model (Qwen3.5-0.8B) to do **natural-language geocoding**: turning a
# question like *"which districts border Nagpur?"* into a spatial SQL query that
# returns real boundary polygons.
#
# It is a simplified reproduction of [gazet](https://github.com/developmentseed/gazet),
# cut down to one country (**India**), one table (Overture administrative
# divisions: country, states and union territories, districts) and eight query
# templates so everything fits in a free Colab session.
#
# **In this notebook (CPU only, ~30 minutes):**
#
# 1. Pull India's administrative boundaries from Overture Maps into a local parquet file
#    and clean the handful of quirks real data has.
# 2. Precompute two spatial relations (who touches whom, who contains whom).
# 3. Write SQL templates paired with question phrasings.
# 4. Decide **what the model never sees**: whole states and individual districts held
#    out of training, so we can later tell memorisation from skill.
# 5. Generate ~3,600 verified `(question, candidates, SQL)` triples and derive a second
#    task (place-name extraction) from the same samples.
# 6. Export the training set, plus a **benchmark** of harder test splits (unseen places,
#    noisy names, compositions of operators) in chat format for notebook 2.
#
# Nothing here calls an LLM. The training data is **synthetic, templated, and
# verified by execution**, which is exactly how the full gazet dataset (43k rows,
# 248 countries) was built.

# %% [markdown]
# ## 0. Setup
#
# The only install is the tutorial's helper package. It carries the boilerplate
# (DuckDB setup, fuzzy search, IO, the benchmark helpers) so the cells below can
# focus on the ideas.

# %%
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("gazet_tutorial") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/developmentseed/gazet-tutorial"]
    )

# %% tags=["parameters"]
# Knobs. In Colab you can leave these alone.
OUT_DIR = None          # where to write the dataset; None -> <work dir>/artifacts
GEODATA_SOURCE = None   # None -> download from Overture S3; or a local parquet glob for development
TARGET_SCALE = 1.0      # 0.1 for a quick smoke run
N_UNSEEN_ANCHORS = 30   # individual districts held out of training (on top of the held-out states)
SEED = 108

# %%
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from gazet_tutorial import data_dir, TABLE
from gazet_tutorial.geodata import (
    connect, ensure_geodata, summarize, duplicate_names, inventory,
    build_adjacency_pairs, build_containment_pairs, containment_coverage,
)
from gazet_tutorial.search import fuzzy_search, fetch_by_id, candidates_csv, merge_candidates, per_place_limit, CANDIDATE_COLUMNS
from gazet_tutorial.sqlutil import execute, format_sql, result_ids
from gazet_tutorial.data import stratified_split, shuffle_candidates, save_jsonl, count_by, token_length_stats
from gazet_tutorial.india import HOLDOUT_STATES, resolve_aliases, resolve_abbreviations
from gazet_tutorial.benchmark import COMPOSITIONAL, NOISE_TYPES, holdout_ids, make_noisy, select_subset
from gazet_tutorial.viz import DEFAULT_TILES

random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

OUT_DIR = Path(OUT_DIR) if OUT_DIR else data_dir() / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Outputs ->", OUT_DIR)

# %% [markdown]
# ## 1. The gazetteer: Overture divisions for India
#
# [Overture Maps](https://overturemaps.org) publishes administrative boundaries
# as GeoParquet on S3. DuckDB can read them in place: the `country` filter and the
# bounding-box filter let it skip whole row groups using column statistics, so we
# download ~60 MB out of a multi-GB theme. Expect 2–3 minutes.
#
# We keep three admin levels (`country`, `region` = state or union territory,
# `county` = district) and `is_land = true` (Overture also ships a water-inclusive
# "territorial" polygon per country).
#
# Real data has quirks, and the download applies three small fixes. Read the printed
# report: one state-level polygon drawn from a non-Indian perspective is dropped,
# 19 Arunachal Pradesh districts that arrive without a state code get one from the
# state polygon they overlap, and two districts that appear twice are deduplicated.
# A trailing " district" is stripped from ~35 names so questions read naturally.

# %%
con = connect(spatial=True)
geodata_path = ensure_geodata(source=GEODATA_SOURCE, con=connect(spatial=True, httpfs=GEODATA_SOURCE is None))
summarize(con, geodata_path)

# %% [markdown]
# Expect 1 country, 36 states and union territories, and 784 districts. Note the
# schema: DuckDB's spatial extension gives us a real `GEOMETRY` column, and names
# live in a struct, so you access them as `names."primary"` (the quotes matter,
# `primary` is a keyword).

# %%
con.execute(f"DESCRIBE SELECT * FROM read_parquet('{geodata_path}')").fetchdf()

# %%
con.execute(
    f"""SELECT id, names."primary" AS name, subtype, region, admin_level
        FROM read_parquet('{geodata_path}') WHERE subtype = 'region' ORDER BY name"""
).fetchdf()

# %% [markdown]
# ### Same name, different place
#
# A geocoder's central problem is **ambiguity**. India has two Bilaspurs (Himachal
# Pradesh and Chhattisgarh), two Hamirpurs, two Pratapgarhs, two Balrampurs, and
# three union territories whose single district carries the territory's name. The
# model we train will have to read a table of candidates and pick the right row; it
# cannot guess from the name.

# %%
duplicate_names(con, geodata_path)

# %%
import geopandas as gpd

gdf = gpd.read_parquet(geodata_path)
gdf["name"] = [n["primary"] for n in gdf["names"]]   # the struct column arrives as dicts
states = gdf[gdf.subtype == "region"].drop(columns=["names"]).copy()
states["geometry"] = states.geometry.simplify(0.02)   # keep the map light
states.explore(column="name", legend=False, tooltip=["name", "region"], tiles=DEFAULT_TILES)  # OSM errors out in Colab, CartoDB wants a key

# %% [markdown]
# ## 2. Spatial relations
#
# To write a correct *"which districts border X?"* example we need to know, for each X,
# that at least one district actually touches it. So we precompute two relation tables
# with DuckDB spatial self-joins. The bounding-box pre-filter (`ST_Intersects` on
# envelopes) is what keeps an 821 × 821 join down to seconds.

# %%
t0 = time.time()
inv = inventory(con, geodata_path)
adjacency = build_adjacency_pairs(con, geodata_path)
containment = build_containment_pairs(con, geodata_path)
print(f"{len(inv)} features, {len(adjacency)} directed adjacency pairs, "
      f"{len(containment)} containment pairs ({time.time() - t0:.1f}s)")
adjacency.head()

# %%
# Sanity check: states that touch Madhya Pradesh.
adjacency.query("anchor_name == 'Madhya Pradesh' and target_subtype == 'region'")[["target_name", "target_subtype"]]

# %% [markdown]
# `ST_Within` is strict: a district whose coastline differs from the state's by a
# sliver is *not* within it. Real boundary data is never perfectly nested, so we
# report coverage instead of assuming it is 100%. A few misses are fine for
# training data; the verification step later only keeps queries that return rows.
# (For district → state containment the `region` code column would give you the
# answer without any geometry. We use the spatial predicate because that is the
# operator we want the model to learn.)

# %%
containment_coverage(containment, inv)

# %%
rel_dir = OUT_DIR / "relations"
rel_dir.mkdir(exist_ok=True)
adjacency.to_parquet(rel_dir / "adjacency_pairs.parquet", index=False)
containment.to_parquet(rel_dir / "containment_pairs.parquet", index=False)

# %% [markdown]
# ## 3. Templates: SQL skeletons plus ways of asking
#
# A template is a SQL string with placeholders and a list of **question hints**,
# natural-language phrasings with the same placeholders. The full gazet has 81
# templates in 15 families; we keep eight that cover the core spatial vocabulary:
#
# | family | template | what the model learns |
# |---|---|---|
# | direct_lookup | `lookup_01` | copy the right id out of the candidate table |
# | disambiguation | `disambiguate_01` | "Bilaspur, Himachal Pradesh": use the container to pick among same-name rows |
# | adjacency | `adj_01`, `adj_02` | `ST_Touches`; `adj_01` keeps neighbours at the anchor's own level, `adj_02` filters to the level the question names |
# | containment | `contain_01`, `contain_02` | `ST_Within` and its flip, `ST_Contains` |
# | buffer | `buffer_01` | `ST_Buffer` distance queries |
# | aggregation | `agg_01` | `ST_Area` + `ORDER BY` + `LIMIT` |
#
# Every SQL string refers to the table symbolically as `read_parquet('divisions_area')`;
# the real file path is substituted at execution time. That keeps the training data
# portable across machines.

# %% tags=["templates"]
@dataclass
class SQLTemplate:
    template_id: str
    family: str
    difficulty: str
    sql: str                       # placeholders: {anchor_id}, {target_subtype}, {buffer_m}, {top_n}, {order}
    hints: list                    # placeholders: {anchor_name}, {container_name}, {target_label}, {buffer_label}, ...
    anchor_from: str               # "inventory" | "adjacency" | "containment" | "contained"
    target_subtypes: tuple = ()    # subtypes this template can ask for (chosen at generation time)


# Vocabulary. Overture's "region" is a state or union territory, "county" is a district.
ALIASES = {"region": ["states", "states and union territories"], "county": ["districts"]}
SINGULAR = {"region": ["state"], "county": ["district"]}

T = f"read_parquet('{TABLE}')"
SELECT_B = f'SELECT b.id, b.names."primary" AS name, b.subtype, ST_AsGeoJSON(b.geometry) AS geometry FROM {T} AS b, a'

TEMPLATES = [
    SQLTemplate(
        template_id="lookup_01", family="direct_lookup", difficulty="easy", anchor_from="inventory",
        sql=f"SELECT id, names.\"primary\" AS name, subtype, ST_AsGeoJSON(geometry) AS geometry FROM {T} WHERE id = '{{anchor_id}}'",
        hints=["show me {anchor_name}", "where is {anchor_name}?", "map of {anchor_name}", "boundary of {anchor_name}",
               "find {anchor_name}", "outline of {anchor_name}", "{anchor_name}", "pull up {anchor_name}",
               "locate {anchor_name}", "what does {anchor_name} look like?", "get the shape of {anchor_name}"],
    ),
    SQLTemplate(
        template_id="disambiguate_01", family="disambiguation", difficulty="easy", anchor_from="contained",
        sql=f"SELECT id, names.\"primary\" AS name, subtype, region, ST_AsGeoJSON(geometry) AS geometry FROM {T} WHERE id = '{{anchor_id}}'",
        hints=["{anchor_name}, {container_name}", "{anchor_name} in {container_name}", "show me {anchor_name}, {container_name}",
               "the {anchor_name} that's in {container_name}", "map of {anchor_name} ({container_name})",
               "find {anchor_name} in {container_name}", "{anchor_name} ({container_name})",
               "locate {anchor_name} within {container_name}", "{anchor_name} district in {container_name}",
               "where is {anchor_name}, {container_name}?", "{anchor_name} {container_name}"],
    ),
    SQLTemplate(
        template_id="adj_01", family="adjacency", difficulty="medium", anchor_from="adjacency",
        # neighbours *at the same level*: a state's neighbours are states, a district's are districts
        sql=f"WITH a AS (SELECT geometry, subtype FROM {T} WHERE id = '{{anchor_id}}') {SELECT_B} WHERE b.id != '{{anchor_id}}' AND b.subtype = a.subtype AND ST_Touches(a.geometry, b.geometry)",
        hints=["what touches {anchor_name}?", "neighbours of {anchor_name}", "what shares a border with {anchor_name}",
               "everything bordering {anchor_name}", "what's next to {anchor_name}?", "places adjacent to {anchor_name}",
               "what surrounds {anchor_name}?", "areas touching {anchor_name}", "find the neighbours of {anchor_name}",
               "list everything that borders {anchor_name}"],
    ),
    SQLTemplate(
        template_id="adj_02", family="adjacency", difficulty="medium", anchor_from="adjacency", target_subtypes=("region", "county"),
        sql=f"WITH a AS (SELECT geometry FROM {T} WHERE id = '{{anchor_id}}') {SELECT_B} WHERE b.id != '{{anchor_id}}' AND b.subtype = '{{target_subtype}}' AND ST_Touches(a.geometry, b.geometry)",
        hints=["which {target_label} border {anchor_name}?", "{target_label} that touch {anchor_name}",
               "neighbouring {target_label} of {anchor_name}", "{target_label} adjacent to {anchor_name}",
               "which {target_label} share a border with {anchor_name}?", "list the {target_label} next to {anchor_name}",
               "{target_label} along the {anchor_name} border", "show {target_label} bordering {anchor_name}",
               "what {target_label} are next to {anchor_name}?"],
    ),
    SQLTemplate(
        template_id="contain_01", family="containment", difficulty="medium", anchor_from="containment", target_subtypes=("county", "region"),
        sql=f"WITH a AS (SELECT geometry FROM {T} WHERE id = '{{anchor_id}}') {SELECT_B} WHERE b.id != '{{anchor_id}}' AND b.subtype = '{{target_subtype}}' AND ST_Within(b.geometry, a.geometry)",
        hints=["{target_label} of {anchor_name}", "what {target_label} are in {anchor_name}?", "list all {target_label} inside {anchor_name}",
               "{target_label} in {anchor_name}", "which {target_label} fall within {anchor_name}?", "all {target_label} within {anchor_name}",
               "{anchor_name} {target_label}", "give me the {target_label} in {anchor_name}", "which {target_label} does {anchor_name} contain?",
               "show every one of the {target_label} in {anchor_name}"],
    ),
    SQLTemplate(
        template_id="contain_02", family="containment", difficulty="medium", anchor_from="contained", target_subtypes=("region",),
        sql=f"WITH a AS (SELECT geometry FROM {T} WHERE id = '{{anchor_id}}') {SELECT_B} WHERE b.subtype = '{{target_subtype}}' AND ST_Contains(b.geometry, a.geometry)",
        hints=["which {target_singular} is {anchor_name} in?", "what {target_singular} contains {anchor_name}?",
               "{anchor_name} belongs to which {target_singular}?", "{target_singular} of {anchor_name}",
               "find the {target_singular} that contains {anchor_name}", "which {target_singular} does {anchor_name} fall in?",
               "what {target_singular} is {anchor_name} part of?", "tell me the {target_singular} for {anchor_name}"],
    ),
    SQLTemplate(
        template_id="buffer_01", family="buffer", difficulty="hard", anchor_from="inventory", target_subtypes=("county", "region"),
        sql=f"WITH a AS (SELECT ST_Buffer(geometry, {{buffer_m}} / 111320.0) AS geom FROM {T} WHERE id = '{{anchor_id}}') "
            f"SELECT b.id, b.names.\"primary\" AS name, b.subtype, ST_AsGeoJSON(b.geometry) AS geometry FROM {T} AS b, a "
            f"WHERE b.id != '{{anchor_id}}' AND b.subtype = '{{target_subtype}}' AND ST_Intersects(b.geometry, a.geom)",
        hints=["which {target_label} are within {buffer_label} of {anchor_name}?", "{target_label} within {buffer_label} of {anchor_name}",
               "find {target_label} within a {buffer_label} radius of {anchor_name}", "{target_label} within {buffer_label} around {anchor_name}",
               "what {target_label} fall within {buffer_label} of {anchor_name}?", "show {target_label} up to {buffer_label} from {anchor_name}",
               "{target_label} near {anchor_name}, within {buffer_label}", "{target_label} no more than {buffer_label} from {anchor_name}"],
    ),
    SQLTemplate(
        template_id="agg_01", family="aggregation", difficulty="hard", anchor_from="containment", target_subtypes=("county", "region"),
        sql=f"WITH a AS (SELECT geometry FROM {T} WHERE id = '{{anchor_id}}') "
            f"SELECT b.id, b.names.\"primary\" AS name, b.subtype, ST_Area(b.geometry) AS area, ST_AsGeoJSON(b.geometry) AS geometry FROM {T} AS b, a "
            f"WHERE b.id != '{{anchor_id}}' AND b.subtype = '{{target_subtype}}' AND ST_Within(b.geometry, a.geometry) ORDER BY area {{order}} LIMIT {{top_n}}",
        hints=["top {top_n} {superlative} {target_label} in {anchor_name}", "{top_n} {superlative} {target_label} of {anchor_name}",
               "the {top_n} {superlative} {target_label} in {anchor_name}", "which are the {top_n} {superlative} {target_label} in {anchor_name}?",
               "{anchor_name}'s {top_n} {superlative} {target_label}",
               # singular forms, used when top_n == 1
               "the {superlative} {target_singular} in {anchor_name}", "which is the {superlative} {target_singular} of {anchor_name}?",
               "{superlative} {target_singular} in {anchor_name}", "what is the {superlative} {target_singular} of {anchor_name}?"],
    ),
]

BUFFER_CHOICES = [(10000, "10 km"), (25000, "25 km"), (50000, "50 km"), (100000, "100 km"), (200000, "200 km")]
TOP_N_CHOICES = [1, 3, 5]
SUPERLATIVES = {"DESC": ["largest", "biggest"], "ASC": ["smallest"]}

TEMPLATE_BY_ID = {t.template_id: t for t in TEMPLATES}
print(f"{len(TEMPLATES)} templates, {sum(len(t.hints) for t in TEMPLATES)} hints")

# %% [markdown]
# Why `/ 111320.0`? `ST_Buffer` works in the units of the coordinates, which are
# degrees. One degree of latitude is about 111 km everywhere; one degree of
# longitude is 111 km at the equator but only ~95 km at Delhi's latitude. So a
# "100 km" buffer here is slightly oval. That is fine for training data (the gold
# answer is whatever the query returns, and the model learns the query), but it
# is the kind of simplification you would replace with a projected CRS in production.

# %% [markdown]
# ## 4. What the model never sees
#
# A small model trained on a few hundred place names will partly **memorise them**.
# If the test set reuses the same names, it looks better than it is. So before
# generating anything we split the *places* (not just the samples):
#
# - **Held-out states** (Kerala, Odisha, Goa): neither the state nor any of its
#   districts is ever a training anchor → benchmark split `unseen_region`.
# - **Held-out districts**: 30 more districts from other states, including one of
#   each same-name pair (Bilaspur, Hamirpur, …) → split `unseen_anchor`.
# - Everything else is the **training anchor pool**.
#
# Held-out places can still *appear* in training data, as neighbours in a result set
# or as look-alike rows in a candidate table. What they never do is anchor a question.

# %%
unseen_region_ids, unseen_anchor_ids = holdout_ids(inv, HOLDOUT_STATES, n_random=N_UNSEEN_ANCHORS, seed=SEED)
TRAIN_IDS = set(inv["id"]) - unseen_region_ids - unseen_anchor_ids
name_of = dict(zip(inv["id"], inv["name"]))
print(f"held-out states {list(HOLDOUT_STATES.values())}: {len(unseen_region_ids)} features")
print(f"held-out districts ({len(unseen_anchor_ids)}):", sorted(name_of[i] for i in unseen_anchor_ids))
print(f"training anchor pool: {len(TRAIN_IDS)} features")

# %% [markdown]
# ## 5. Generating one sample
#
# A sample has four parts:
#
# - a **question**, rendered from a hint with real place names;
# - a **candidate table**: the true place plus 9 look-alike rows found by fuzzy
#   name search (what the inference pipeline will see);
# - the **SQL**, rendered from the template with the true id;
# - the **target places**, the names an extractor should pull from the question.
#
# The SQL is executed, and samples whose query errors or returns nothing are
# discarded. That execution check is what makes templated data trustworthy.

# %%
# Which names are ambiguous within the pool a template samples from? A sample like
# "show me Bilaspur" would have two equally valid answers, so we only use ambiguous
# names where the template itself provides the disambiguating context.
name_counts = inv.groupby("name").size()
AMBIGUOUS_ALL = set(name_counts[name_counts > 1].index)
counts_by_subtype = inv.groupby(["subtype", "name"]).size()
def ambiguous_within(subtype: str) -> set:
    s = counts_by_subtype.loc[subtype]
    return set(s[s > 1].index)
AMBIGUOUS_WITHIN = {st: ambiguous_within(st) for st in ("region", "county")}
print("ambiguous anywhere:", sorted(AMBIGUOUS_ALL))
print("ambiguous among districts:", sorted(AMBIGUOUS_WITHIN["county"]))

# %%
def to_candidate_rows(rows: list) -> list:
    return [{"candidate_id": f"c{i}", **{k: (None if pd.isna(r.get(k)) else r.get(k)) for k in ("id", "name", "subtype", "region")}}
            for i, r in enumerate(rows, 1)]


def build_candidates(anchor_id: str, anchor_name: str, must_include: list | None = None, n: int = 10) -> list:
    """True anchor + fuzzy-name distractors, as the model will see them at inference.

    Rows the query *mentions* (a disambiguation container, a second anchor) are
    forced in so the model can ground every name in the question.
    """
    anchor = fetch_by_id(con, geodata_path, anchor_id).iloc[0].to_dict()
    distractors = fuzzy_search(con, geodata_path, anchor_name, limit=n - 1, exclude_id=anchor_id)
    rows = [anchor] + distractors.to_dict(orient="records")
    for extra_id in must_include or []:
        if extra_id not in {r["id"] for r in rows}:
            rows = rows[:-1] + [fetch_by_id(con, geodata_path, extra_id).iloc[0].to_dict()]
    return to_candidate_rows(rows)


def pick_anchor(template: SQLTemplate, rng: random.Random, allowed: set) -> dict | None:
    """Sample an anchor from `allowed` (and any extra slots) so the SQL is non-trivial."""
    t = template
    if t.anchor_from == "inventory":
        pool = inv[(inv.subtype != "country") & ~inv.name.isin(AMBIGUOUS_ALL) & inv.id.isin(allowed)]
        if pool.empty:
            return None
        row = pool.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        return {"anchor_id": row.id, "anchor_name": row["name"], "target_subtype": rng.choice(t.target_subtypes) if t.target_subtypes else None}

    if t.anchor_from == "adjacency":
        pool = adjacency[~adjacency.anchor_name.isin(AMBIGUOUS_ALL) & adjacency.anchor_id.isin(allowed)]
        target_subtype = rng.choice(t.target_subtypes) if t.target_subtypes else None
        if target_subtype:
            pool = pool[pool.target_subtype == target_subtype]
        if pool.empty:
            return None
        row = pool.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        return {"anchor_id": row.anchor_id, "anchor_name": row.anchor_name, "target_subtype": target_subtype}

    if t.anchor_from == "containment":
        # anchor = container; the container's subtype decides which children exist
        target_subtype = rng.choice(t.target_subtypes)
        pool = containment[(containment.contained_subtype == target_subtype) & containment.container_id.isin(allowed)]
        pool = pool[pool.container_subtype == ("country" if target_subtype == "region" else "region")]
        pool = pool[~pool.container_name.isin(AMBIGUOUS_WITHIN.get("region", set()))]
        if pool.empty:
            return None
        row = pool.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        return {"anchor_id": row.container_id, "anchor_name": row.container_name, "target_subtype": target_subtype}

    if t.anchor_from == "contained":
        # anchor = a district; container = its state
        pool = containment[(containment.container_subtype == "region") & (containment.contained_subtype == "county")
                           & containment.contained_id.isin(allowed)]
        if t.template_id == "disambiguate_01":
            amb = pool[pool.contained_name.isin(AMBIGUOUS_ALL)]       # favour the genuinely ambiguous ones
            if rng.random() < 0.5 and len(amb):
                pool = amb
        else:
            pool = pool[~pool.contained_name.isin(AMBIGUOUS_WITHIN["county"])]
        if pool.empty:
            return None
        row = pool.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        return {"anchor_id": row.contained_id, "anchor_name": row.contained_name,
                "container_id": row.container_id, "container_name": row.container_name,
                "target_subtype": rng.choice(t.target_subtypes) if t.target_subtypes else None}
    raise ValueError(t.anchor_from)


def render_question(template: SQLTemplate, slots: dict, rng: random.Random) -> str:
    hints = template.hints
    if template.template_id == "agg_01":
        hints = [h for h in hints if ("{top_n}" in h) == (slots["top_n"] > 1)]
    return rng.choice(hints).format(**slots)


def generate_sample(template: SQLTemplate, rng: random.Random, sample_id: str, allowed: set) -> dict | None:
    anchor = pick_anchor(template, rng, allowed)
    if anchor is None:
        return None
    ts = anchor.get("target_subtype")
    slots = {
        "anchor_name": anchor["anchor_name"],
        "container_name": anchor.get("container_name"),
        "target_label": rng.choice(ALIASES[ts]) if ts else None,
        "target_singular": rng.choice(SINGULAR[ts]) if ts else None,
    }
    sql_slots = {"anchor_id": anchor["anchor_id"], "target_subtype": ts}
    if template.template_id == "buffer_01":
        sql_slots["buffer_m"], slots["buffer_label"] = rng.choice(BUFFER_CHOICES)
    if template.template_id == "agg_01":
        sql_slots["top_n"] = slots["top_n"] = rng.choice(TOP_N_CHOICES)
        sql_slots["order"] = rng.choice(list(SUPERLATIVES))
        slots["superlative"] = rng.choice(SUPERLATIVES[sql_slots["order"]])

    sql = template.sql.format(**sql_slots)
    try:
        result = execute(con, sql, geodata_path)
    except Exception:
        return None
    if result.empty:
        return None

    question = render_question(template, slots, rng)
    places = [anchor["anchor_name"]] + ([anchor["container_name"]] if template.template_id == "disambiguate_01" else [])
    must_include = [anchor["container_id"]] if template.template_id == "disambiguate_01" else []
    candidates = build_candidates(anchor["anchor_id"], anchor["anchor_name"], must_include=must_include)

    return {
        "id": sample_id,
        "question": question,
        "candidates": candidates,
        "target": {"selected_candidates": ["c1"], "sql": sql, "places": places},
        "metadata": {
            "sample_id": sample_id,
            "task_family": template.family, "template_id": template.template_id, "sql_difficulty": template.difficulty,
            "anchor_id": anchor["anchor_id"], "anchor_name": anchor["anchor_name"],
            "container_name": anchor.get("container_name"),
            "gold_ids": result_ids(result, ordered=template.family == "aggregation"),
            "ordered": template.family == "aggregation",
            "sql_verified": True,
        },
    }


demo_rng = random.Random(1)
for t in TEMPLATES:
    s = generate_sample(t, demo_rng, "demo", TRAIN_IDS)
    print(f"[{t.template_id}] {s['question']!r}  ->  {len(s['metadata']['gold_ids'])} feature(s); places={s['target']['places']}")

# %% [markdown]
# Here is one complete sample. Look at the candidate table: the true answer is
# `c1`, and the other rows are what a fuzzy name search returns for the same name.

# %%
s = generate_sample(TEMPLATE_BY_ID["disambiguate_01"], random.Random(7), "demo", TRAIN_IDS)
print(s["question"])
print(candidates_csv(s["candidates"]))
print(format_sql(s["target"]["sql"]))
print(s["target"]["places"])

# %% [markdown]
# ### Light paraphrasing
#
# Eight templates with ten hints each is not much surface variety. We apply a few
# shallow, meaning-preserving rewrites (British/American spelling, "show me" →
# "display", "within" → "inside") so the model sees more ways of asking the same thing.
# These are deliberately local: the SQL intent must not change.

# %%
GENERIC_RULES = [
    (r"\bneighbouring\b", ["neighboring"]), (r"\bneighbours\b", ["neighbors"]), (r"\bwhat's\b", ["what is"]),
    (r"\bshow me\b", ["show", "display"]), (r"\bgive me\b", ["show", "list"]), (r"\bpull up\b", ["show", "display"]),
    (r"\bfind\b", ["show", "locate"]),
]
FAMILY_RULES = {
    "adjacency": [(r"\bbordering\b", ["next to", "adjacent to"]), (r"\btouching\b", ["next to"]),
                  (r"share a border with", ["are adjacent to", "are next to"]), (r"adjacent to", ["next to"])],
    "containment": [(r"\bwithin\b", ["inside", "in"]), (r"\binside\b", ["in"]), (r"\bcontain\b", ["have"])],
    "buffer": [(r"within ([0-9]+\s*(?:km|m)) of", [r"up to \1 from", r"at a distance of \1 from"]), (r"close to", ["near"])],
    "aggregation": [(r"\blargest\b", ["biggest"]), (r"\bsmallest\b", ["tiniest"])],
    "direct_lookup": [(r"\bwhere is\b", ["show", "locate"]), (r"\bmap of\b", ["outline of"])],
    "disambiguation": [(r"\bshow me\b", ["find", "show"])],
}


def diversify_question(question: str, family: str, rng: random.Random) -> str:
    if rng.random() < 0.35:            # keep a third of questions canonical
        return question
    rules = GENERIC_RULES + FAMILY_RULES.get(family, [])
    for _ in range(2 if rng.random() < 0.5 else 1):
        matches = [(p, r) for p, reps in rules for r in reps if re.search(p, question, flags=re.I)]
        if not matches:
            break
        pattern, replacement = rng.choice(matches)
        question = re.sub(r"\s+", " ", re.sub(pattern, replacement, question, count=1, flags=re.I)).strip()
    return question


print(diversify_question("show me the neighbouring districts within Karnataka", "containment", random.Random(3)))

# %% [markdown]
# ## 6. Generate the training set
#
# Per-template targets below give ~3,600 samples from the training anchor pool.
# We deduplicate on `(template, SQL, question)` and cap how many samples any one
# anchor contributes (states are containers for two templates, so they get a
# higher cap than districts).

# %%
TARGETS = {
    "lookup_01": 450, "disambiguate_01": 450, "adj_01": 450, "adj_02": 500,
    "contain_01": 400, "contain_02": 400, "buffer_01": 500, "agg_01": 450,
}
TARGETS = {k: max(10, int(v * TARGET_SCALE)) for k, v in TARGETS.items()}
RETRY_MULTIPLIER = 3
MAX_PER_ANCHOR = {"containment": 30, "aggregation": 30}     # per (template, anchor); default below
DEFAULT_MAX_PER_ANCHOR = 8


def sample_id(prefix: str, template_id: str, anchor_id: str, question: str) -> str:
    """Stable id from *what was asked*, not from position: regenerating the data with a changed
    gold SQL keeps the id (and any cached model predictions) for every prompt that did not change."""
    return f"{prefix}_{hashlib.sha1(f'{template_id}|{anchor_id}|{question}'.encode()).hexdigest()[:10]}"


def generate_split(targets: dict, allowed: set, rng: random.Random, prefix: str, max_per_anchor: dict | None = None) -> list:
    out, seen, per_anchor = [], set(), Counter()
    caps = max_per_anchor or {}
    for template in TEMPLATES:
        made, target = 0, targets.get(template.template_id, 0)
        cap = caps.get(template.family, DEFAULT_MAX_PER_ANCHOR)
        for _ in range(target * RETRY_MULTIPLIER):
            if made >= target:
                break
            s = generate_sample(template, rng, "pending", allowed)
            if s is None:
                continue
            s["question"] = diversify_question(s["question"], template.family, rng)
            s["id"] = s["metadata"]["sample_id"] = sample_id(prefix, template.template_id, s["metadata"]["anchor_id"], s["question"])
            key = (template.template_id, s["target"]["sql"], s["question"].lower())
            anchor_key = (template.template_id, s["metadata"]["anchor_id"])
            if key in seen or per_anchor[anchor_key] >= cap:
                continue
            seen.add(key)
            per_anchor[anchor_key] += 1
            out.append(s)
            made += 1
        print(f"  {template.template_id:16s} {made:4d}")
    return out


t0 = time.time()
rng = random.Random(SEED)
print("training pool")
samples = generate_split(TARGETS, TRAIN_IDS, rng, "train", MAX_PER_ANCHOR)
print(f"{len(samples)} samples from {len({s['metadata']['anchor_id'] for s in samples})} anchors in {time.time() - t0:.0f}s")

# %%
pd.Series(count_by(samples)).plot.barh(title="training samples per family")

# %% [markdown]
# ### Validate
#
# Every SQL was already executed once during generation. We still re-check the
# structural invariants (candidate ids resolve, no duplicate candidate rows, no
# unfilled `{placeholders}`, no held-out anchor) and re-execute a random 10%, the
# way gazet's `validate_dataset.py` does.

# %%
def validate(sample: dict, allowed: set | None = None) -> list:
    issues = []
    cids = {c["candidate_id"] for c in sample["candidates"]}
    if not set(sample["target"]["selected_candidates"]) <= cids:
        issues.append("selected candidate missing")
    ids = [c["id"] for c in sample["candidates"]]
    if len(ids) != len(set(ids)):
        issues.append("duplicate candidate")
    if re.search(r"\{[a-z_]+\}", sample["question"]) or re.search(r"\{[a-z_]+\}", sample["target"]["sql"]):
        issues.append("unfilled placeholder")
    if not sample["target"]["places"]:
        issues.append("no places")
    if allowed is not None and sample["metadata"]["anchor_id"] not in allowed:
        issues.append("anchor outside the allowed pool")
    return issues

bad = {s["id"]: validate(s, TRAIN_IDS) for s in samples if validate(s, TRAIN_IDS)}
recheck = random.Random(0).sample(samples, k=max(1, len(samples) // 10))
reexec_fail = [s["id"] for s in recheck if execute(con, s["target"]["sql"], geodata_path).empty]
print(f"structural issues: {len(bad)}   re-executed {len(recheck)}, empty results: {len(reexec_fail)}")
samples = [s for s in samples if s["id"] not in bad and s["id"] not in reexec_fail]
print(f"{len(samples)} samples kept")

# %% [markdown]
# ## 7. Export two tasks in chat format
#
# One synthetic sample yields two training examples, both as `messages` lists
# (system / user / assistant) so the tokenizer's chat template can render them:
#
# 1. **SQL generation** — system: schema + rules; user: candidate CSV + question; assistant: SQL.
# 2. **Place extraction** — system: rules + a few examples; user: question; assistant: JSON list of names.
#
# At inference the extractor runs first, its names drive the fuzzy search that builds
# the candidate table, and the SQL model runs second. One small model learns both.
#
# Two details that matter more than they look:
#
# - Candidate rows are **shuffled** before export so the answer is not always `c1`.
# - The SQL is **pretty-printed** so the model learns one consistent style.

# %%
SQL_SYSTEM = f"""You are a text-to-SQL translator for natural-language geocoding.

One DuckDB table is available via {T} with columns:
  id VARCHAR, names STRUCT("primary" VARCHAR), subtype VARCHAR (country | region | county),
  country VARCHAR (ISO-2), region VARCHAR (ISO 3166-2 code, e.g. 'IN-MH'), geometry GEOMETRY
The spatial extension is loaded; access names with names."primary".
Pick the matching candidate id(s) from <CANDIDATES> and return only SQL
that outputs id, names."primary" AS name and ST_AsGeoJSON(geometry) AS geometry."""

PLACE_SYSTEM = """You are a geographic entity extractor. Extract the place names the user is asking about and return valid JSON only.

OUTPUT FORMAT:
{"places": [{"place": "<name>"}]}

RULES:
- Extract every place name that is explicitly mentioned, in the order it appears.
- When a place is followed by its containing state as context ("Raigarh, Chhattisgarh"), extract both the place and the state.
- Do not expand category nouns like "districts" or "states" into place names ("districts of Karnataka" -> "Karnataka" only).
- Ignore distances, counts and superlatives.
- No duplicates.

EXAMPLES:
Query: "districts of Karnataka"
-> {"places": [{"place": "Karnataka"}]}

Query: "Raigarh, Chhattisgarh"
-> {"places": [{"place": "Raigarh"}, {"place": "Chhattisgarh"}]}

Query: "which states border Madhya Pradesh?"
-> {"places": [{"place": "Madhya Pradesh"}]}

Query: "districts within 50 km of Nagpur"
-> {"places": [{"place": "Nagpur"}]}

Query: "top 3 largest districts in Rajasthan"
-> {"places": [{"place": "Rajasthan"}]}

Query: "which state is Bhopal in?"
-> {"places": [{"place": "Bhopal"}]}"""

USER_TEMPLATE = "<CANDIDATES>\n{candidates}\n</CANDIDATES>\n\n<USER_QUERY>\n{question}\n</USER_QUERY>"

# The system prompt is part of every training example, so its examples must not
# name a held-out place.
unseen_names = {name_of[i] for i in unseen_region_ids | unseen_anchor_ids}
leaks = [n for n in ("Karnataka", "Raigarh", "Chhattisgarh", "Madhya Pradesh", "Nagpur", "Rajasthan", "Bhopal") if n in unseen_names]
assert not leaks, f"PLACE_SYSTEM mentions held-out places {leaks}; swap the examples"


def to_sql_example(sample: dict) -> dict:
    user = USER_TEMPLATE.format(candidates=candidates_csv(sample["candidates"]), question=sample["question"])
    return {"messages": [{"role": "system", "content": SQL_SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": format_sql(sample["target"]["sql"])}],
            "metadata": sample["metadata"]}


def to_places_example(sample: dict) -> dict:
    answer = json.dumps({"places": [{"place": p} for p in sample["target"]["places"]]}, ensure_ascii=False)
    return {"messages": [{"role": "system", "content": PLACE_SYSTEM},
                         {"role": "user", "content": sample["question"]},
                         {"role": "assistant", "content": answer}],
            "metadata": sample["metadata"]}


def finalize(samples: list, shuffle: bool = True) -> list:
    out = [shuffle_candidates(s) if shuffle else s for s in samples]
    for s in out:   # selected candidate is wherever the anchor row landed
        s["target"]["selected_candidates"] = [c["candidate_id"] for c in s["candidates"] if c["id"] == s["metadata"]["anchor_id"]]
    return out


shuffled = finalize(samples)
train, val, test = stratified_split(shuffled, seed=42)
print(f"train {len(train)}  val {len(val)}  test {len(test)}")

ds_dir = OUT_DIR / "dataset"
stats = {"total_samples": len(shuffled), "targets": TARGETS, "holdout_states": HOLDOUT_STATES,
         "n_unseen_region_features": len(unseen_region_ids), "n_unseen_anchor_features": len(unseen_anchor_ids),
         "splits": {}}
for split_name, rows in (("train", train), ("val", val), ("test", test)):
    save_jsonl([to_sql_example(s) for s in rows], ds_dir / "sql" / f"{split_name}.jsonl")
    save_jsonl([to_places_example(s) for s in rows], ds_dir / "places" / f"{split_name}.jsonl")
    stats["splits"][split_name] = {"n": len(rows), "by_family": count_by(rows)}
save_jsonl(shuffled, ds_dir / "raw" / "samples.jsonl")
(ds_dir / "prompts.json").write_text(json.dumps({"sql_system": SQL_SYSTEM, "place_system": PLACE_SYSTEM,
                                                 "user_template": USER_TEMPLATE, "candidate_columns": CANDIDATE_COLUMNS}, indent=2, ensure_ascii=False))
(ds_dir / "templates.json").write_text(json.dumps([asdict(t) for t in TEMPLATES], indent=2, ensure_ascii=False))
(ds_dir / "stats.json").write_text(json.dumps(stats, indent=2))
print("wrote", sorted(str(p.relative_to(OUT_DIR)) for p in ds_dir.rglob("*") if p.is_file()))

# %%
example = to_sql_example(train[0])
print(example["messages"][1]["content"])
print("\n--- assistant ---\n" + example["messages"][2]["content"])
print("\n--- places task ---\n" + to_places_example(train[0])["messages"][2]["content"])

# %% [markdown]
# ## 8. The benchmark: harder test splits
#
# `test` above is drawn from the same anchors and templates as `train`; call it
# `iid`. It tells you whether training worked, not whether the model can geocode.
# We add five splits, each isolating one kind of generalisation:
#
# | split | built from | what a drop here means |
# |---|---|---|
# | `iid` | the 10% test split | regression / sanity |
# | `unseen_region` | held-out states + their districts, same templates | the model memorised anchors |
# | `unseen_anchor` | 30 held-out districts, same templates | same, at district granularity |
# | `noisy` | test questions with abbreviations (UP), old names (Bangalore), typos, lowercase; candidates re-derived by fuzzy search on the noisy string | brittle name handling, or fuzzy search failed (we report both) |
# | `compositional` | 3 templates that combine two operators, never used in training | the model learned templates, not operators |
# | `paraphrase` | LLM rewrites of test questions (built offline with `scripts/paraphrase.py`) | brittle phrasing |
#
# The same metric (execution accuracy) is used everywhere, and notebook 2 also scores
# gpt-oss:20b, the much larger model gazet runs in production, on a fixed subset.

# %%
UNSEEN_TARGETS = {t.template_id: int(40 * TARGET_SCALE) or 5 for t in TEMPLATES}
rng = random.Random(SEED + 1)
print("unseen_region (held-out states)")
unseen_region = finalize(generate_split(UNSEEN_TARGETS, unseen_region_ids, rng, "ureg", MAX_PER_ANCHOR))
print("unseen_anchor (held-out districts)")
unseen_anchor = finalize(generate_split(UNSEEN_TARGETS, unseen_anchor_ids, rng, "uanc", MAX_PER_ANCHOR))
bench = {"iid": test, "unseen_region": unseen_region, "unseen_anchor": unseen_anchor}
for k, v in bench.items():
    assert not {s["id"]: validate(s) for s in v if validate(s)}, k
print({k: len(v) for k, v in bench.items()})

# %% [markdown]
# ### Noisy names
#
# People write "UP", "Bangalore" and "Thiruvanantapuram". The aliases and
# abbreviations live in `gazet_tutorial.india` (hand-written, worth a look). Only
# names present in the data are used. For each noisy question we **re-run the fuzzy
# search on the noisy string**, exactly as inference would, so the candidate table
# may no longer contain the right row; the share where it still does is the
# *retrieval ceiling* for that noise type.

# %%
present = set(inv["name"])
aliases, abbreviations = resolve_aliases(present), resolve_abbreviations(present)
print(f"{len(aliases)} aliases and {len(abbreviations)} abbreviations apply to this data")


def candidates_from_search(search_strings: list, n: int = 10) -> list:
    limit = per_place_limit(len(search_strings))
    merged = merge_candidates([fuzzy_search(con, geodata_path, s, limit=limit) for s in search_strings]).head(n)
    return to_candidate_rows(merged.to_dict(orient="records"))


def noisy_version(sample: dict, rng: random.Random, noise_type: str | None = None) -> dict | None:
    m = sample["metadata"]
    out = make_noisy(sample["question"], m["anchor_name"], m.get("container_name"), rng, aliases, abbreviations, noise_type)
    if out is None:
        return None
    s = json.loads(json.dumps(sample))
    s["id"] = s["metadata"]["sample_id"] = f"{sample['id']}_{out['noise_type']}"
    s["question"] = out["question"]
    s["target"]["places"] = out["places"]
    s["candidates"] = candidates_from_search(out["search_strings"])
    s["target"]["selected_candidates"] = [c["candidate_id"] for c in s["candidates"] if c["id"] == m["anchor_id"]]
    s["metadata"].update({"noise_type": out["noise_type"], "search_strings": out["search_strings"],
                          "anchor_in_candidates": bool(s["target"]["selected_candidates"]), "noisy_from": sample["id"]})
    return s


rng = random.Random(SEED + 2)
pool = test + unseen_region + unseen_anchor
noisy, per_type = [], Counter()
target_per_type = int(50 * TARGET_SCALE) or 5
for noise_type in NOISE_TYPES:
    for s in rng.sample(pool, len(pool)):
        if per_type[noise_type] >= target_per_type:
            break
        v = noisy_version(s, rng, noise_type)
        if v:
            noisy.append(v)
            per_type[noise_type] += 1
bench["noisy"] = noisy
noisy_df = pd.DataFrame([{"noise_type": s["metadata"]["noise_type"], "anchor_in_candidates": s["metadata"]["anchor_in_candidates"]} for s in noisy])
print(noisy_df.groupby("noise_type")["anchor_in_candidates"].agg(["count", "mean"]).rename(columns={"mean": "retrieval_ceiling"}))
for s in noisy[::max(1, len(noisy) // 6)]:
    print(f"  [{s['metadata']['noise_type']:12s}] {s['question']!r}  search={s['metadata']['search_strings']}  anchor found={s['metadata']['anchor_in_candidates']}")

# %% [markdown]
# ### Compositional templates
#
# Three templates the model never trained on, each a combination of two operators
# it did see: districts *of* A that *border* B; the *largest* district *bordering* X;
# the largest districts *within* a radius of X. The reference SQL lives in
# `gazet_tutorial.benchmark` so it cannot leak into the training templates. Gold
# answers come from executing it; the model only ever sees the question.

# %%
def generate_compositional(t, rng: random.Random, sample_id: str) -> dict | None:
    train_regions = adjacency[(adjacency.anchor_subtype == "region") & (adjacency.target_subtype == "region")
                              & adjacency.anchor_id.isin(TRAIN_IDS) & adjacency.target_id.isin(TRAIN_IDS)
                              & ~adjacency.anchor_name.isin(AMBIGUOUS_ALL) & ~adjacency.target_name.isin(AMBIGUOUS_ALL)]
    districts = inv[(inv.subtype == "county") & inv.id.isin(TRAIN_IDS) & ~inv.name.isin(AMBIGUOUS_ALL)]
    slots, must_include, places = {}, [], []
    if t.template_id == "adj_within":
        row = train_regions.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        slots = {"container_id": row.anchor_id, "container_name": row.anchor_name, "neighbor_id": row.target_id, "neighbor_name": row.target_name}
        anchor_id, anchor_name = row.anchor_id, row.anchor_name
        must_include, places = [row.target_id], [row.anchor_name, row.target_name]
        hints = t.hints
    else:
        row = districts.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        anchor_id, anchor_name = row.id, row["name"]
        places = [anchor_name]
        top_n = rng.choice([1, 3])
        slots = {"anchor_id": anchor_id, "anchor_name": anchor_name, "top_n": top_n}
        if t.template_id == "agg_adjacent":
            slots["order"] = rng.choice(list(SUPERLATIVES))
            slots["superlative"] = rng.choice(SUPERLATIVES[slots["order"]])
        else:
            slots["buffer_m"], slots["buffer_label"] = rng.choice(BUFFER_CHOICES[2:4])
        hints = [h for h in t.hints if ("{top_n}" in h) == (top_n > 1)]
    sql = t.sql.format(**slots)
    try:
        result = execute(con, sql, geodata_path)
    except Exception:
        return None
    if result.empty or len(result) < slots.get("top_n", 1):
        return None
    cands = build_candidates(anchor_id, anchor_name, must_include=must_include)
    if must_include:   # two anchors: give the second one its own look-alikes, 5 + 5 like inference would
        cands = to_candidate_rows(
            merge_candidates([fuzzy_search(con, geodata_path, places[0], limit=5), fuzzy_search(con, geodata_path, places[1], limit=5)]).to_dict(orient="records"))
        if not {anchor_id, must_include[0]} <= {c["id"] for c in cands}:
            return None
    return {
        "id": sample_id, "question": rng.choice(hints).format(**slots), "candidates": cands,
        "target": {"selected_candidates": [c["candidate_id"] for c in cands if c["id"] == anchor_id], "sql": sql, "places": places},
        "metadata": {"sample_id": sample_id, "task_family": t.family, "template_id": t.template_id, "sql_difficulty": "hard",
                     "anchor_id": anchor_id, "anchor_name": anchor_name, "container_name": None,
                     "gold_ids": result_ids(result, ordered=t.ordered), "ordered": t.ordered, "sql_verified": True},
    }


rng = random.Random(SEED + 3)
compositional, seen = [], set()
per_template = int(50 * TARGET_SCALE) or 5
for t in COMPOSITIONAL:
    made = 0
    for _ in range(per_template * RETRY_MULTIPLIER):
        if made >= per_template:
            break
        s = generate_compositional(t, rng, "pending")
        if s is not None:
            s["id"] = s["metadata"]["sample_id"] = sample_id("comp", t.template_id, s["metadata"]["anchor_id"], s["question"])
        if s is None or (t.template_id, s["target"]["sql"], s["question"]) in seen:
            continue
        seen.add((t.template_id, s["target"]["sql"], s["question"]))
        compositional.append(s)
        made += 1
    print(f"  {t.template_id:14s} {made:4d}   e.g. {s['question']!r} -> {len(s['metadata']['gold_ids'])} feature(s)")
bench["compositional"] = finalize(compositional)

# %% [markdown]
# ### Write the benchmark
#
# Each split is exported in the same chat format as the training data (so the same
# evaluation code scores everything), for both tasks. `subset_ids.json` fixes a
# template-balanced subset per split (40 SQL, 20 places questions) that the
# expensive gpt-oss:20b baseline is run on, so the comparison is apples to apples.

# %%
bench_dir = OUT_DIR / "benchmark"
bench_stats, subset_ids = {}, {}
for split_name, rows in bench.items():
    save_jsonl([to_sql_example(s) for s in rows], bench_dir / "sql" / f"{split_name}.jsonl")
    save_jsonl([to_places_example(s) for s in rows], bench_dir / "places" / f"{split_name}.jsonl")
    sql_ids = select_subset([to_sql_example(s) for s in rows], 40, seed=SEED)
    subset_ids[split_name] = {"sql": sql_ids, "places": sql_ids[:20]}
    bench_stats[split_name] = {"n": len(rows), "by_template": dict(Counter(s["metadata"]["template_id"] for s in rows))}
bench_stats["noisy"]["retrieval_ceiling"] = noisy_df.groupby("noise_type")["anchor_in_candidates"].mean().round(3).to_dict()
(bench_dir / "subset_ids.json").write_text(json.dumps(subset_ids, indent=1))
(bench_dir / "stats.json").write_text(json.dumps(bench_stats, indent=2))
pd.DataFrame({k: {"n": v["n"], **v["by_template"]} for k, v in bench_stats.items()}).fillna(0).astype(int)

# %% [markdown]
# ### How long are these in tokens?
#
# Fine-tuning needs a `max_seq_length`. The SQL examples are the long ones: the
# candidate table alone is ~300 tokens because each UUID costs ~25 tokens. We check
# with the actual Qwen3.5 tokenizer (only the tokenizer, no model download).

# %%
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-0.8B")
    sql_examples = [to_sql_example(s) for s in train[:300]]
    place_examples = [to_places_example(s) for s in train[:300]]
    lengths = pd.concat([token_length_stats(sql_examples, tok, "sql"), token_length_stats(place_examples, tok, "places")], axis=1)
    display(lengths.round(0))
    print("=> max_seq_length = 1024 covers the 99th percentile" if lengths.loc["99%"].max() < 1024 else "=> raise max_seq_length!")
except Exception as exc:
    print("tokenizer check skipped:", exc)

# %% [markdown]
# ## Done
#
# You now have, under `OUT_DIR/`:
#
# - `dataset/sql/{train,val,test}.jsonl`, `dataset/places/{train,val,test}.jsonl` — chat-format training data
# - `dataset/prompts.json` — the two system prompts (notebook 2 reuses them byte-for-byte at inference)
# - `benchmark/{sql,places}/{iid,unseen_region,unseen_anchor,noisy,compositional}.jsonl`, `benchmark/subset_ids.json`
# - `dataset/raw/samples.jsonl`, `templates.json`, `stats.json`, `relations/` — provenance
#
# A copy of exactly this output is bundled inside the package you installed, so
# **notebook 2 works even if you skipped this one**. If you changed the templates and want to train on
# your own data, run the next cell to download a zip and upload it in notebook 2.
#
# **Things to try:** add a template (e.g. `ST_Difference`: "Maharashtra without Pune"),
# hold out a different set of states, add your own state's abbreviations and old
# names to `gazet_tutorial.india`, or swap the country code and bbox for another country.

# %%
# Optional: download your dataset (Colab only).
# import shutil; from google.colab import files
# zip_path = shutil.make_archive(str(data_dir() / "gazet-in-dataset"), "zip", OUT_DIR)
# files.download(zip_path)
