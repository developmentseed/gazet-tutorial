"""gazet-tutorial playground: compare geocoders side by side, locally.

    streamlit run app/playground.py

Models are OpenAI-compatible endpoints (`llama serve`, Ollama). See
docs/take-it-home.md for how to serve the base model, your fine-tuned GGUF and
gpt-oss:20b with llama.cpp.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from gazet_tutorial import artifacts_dir, data_dir
from gazet_tutorial.benchmark import SPLITS, load_benchmark
from gazet_tutorial.geodata import connect, ensure_geodata
from gazet_tutorial.playground import (
    KINDS,
    RECORDED,
    benchmark_rows,
    default_endpoints,
    endpoints_frame,
    endpoints_from_frame,
    geocode,
    is_up,
    load_recorded,
    mark_candidates,
    places_match,
    replay,
    score,
)
from gazet_tutorial.search import fetch_by_id
from gazet_tutorial.sqlutil import format_sql
from gazet_tutorial.viz import map_features

st.set_page_config(page_title="gazet playground", page_icon="🗺️", layout="wide")

EXAMPLES = [
    "which states border Madhya Pradesh?", "show me Nagpur", "Bilaspur, Himachal Pradesh", "districts of Karnataka",
    "districts within 50 km of Pune", "3 largest districts in Rajasthan", "which state is Bhopal in?",
    "districts of Kerala", "districts of UP", "show me Bangalore Urban", "districts of Karnataka that border Tamil Nadu",
    "largest district bordering Nagpur",
]
MAP_HEIGHT = 340


@st.cache_resource(show_spinner="Loading geodata and benchmark …")
def resources():
    art = artifacts_dir()
    con = connect(spatial=True)
    geodata_path = ensure_geodata(con=connect(spatial=True, httpfs=True))
    prompts = json.loads((art / "dataset" / "prompts.json").read_text())
    bench = load_benchmark(art / "benchmark")
    return con, geodata_path, prompts, bench, art / "baselines" / "gpt-oss-20b-cloud" / "zero"


con, geodata_path, prompts, bench, baseline_dir = resources()

# ── sidebar: models ─────────────────────────────────────────────────────────
st.sidebar.title("Models")
st.sidebar.caption("Any OpenAI-compatible endpoint. A row is used when *enabled* is ticked and the server answers.")
if "endpoints" not in st.session_state:
    st.session_state.endpoints = endpoints_frame(default_endpoints(data_dir() / "served.json"))   # rows from scripts/serve_models.sh
edited = st.sidebar.data_editor(
    st.session_state.endpoints, num_rows="dynamic", hide_index=True, key="endpoint_editor",
    column_config={
        "kind": st.column_config.SelectboxColumn(options=list(KINDS), required=True),
        "enabled": st.column_config.CheckboxColumn(),
        "base_url": st.column_config.TextColumn(width="medium"),
        "model": st.column_config.TextColumn(width="small", help="llama serve ignores it; Ollama needs the tag, e.g. gpt-oss:20b"),
    },
)
st.session_state.endpoints = edited
endpoints = [e for e in endpoints_from_frame(edited) if e.enabled and e.name]
status = {e.name: is_up(e) for e in endpoints}
for e in endpoints:
    st.sidebar.write(("🟢 " if status[e.name] else "🔴 ") + e.name)
live = [e for e in endpoints if status[e.name]]
st.sidebar.divider()
st.sidebar.caption("Start the servers with `llama serve` (docs/take-it-home.md) or `scripts/serve_models.sh start`.")


CAND_COLS = ["gold", "id", "name", "subtype", "region"]


def candidate_table(cands: pd.DataFrame, gold: dict | None, height: int = 200):
    """The candidate CSV as a table; with a gold row, mark the anchor and the answer rows."""
    if gold is not None:
        cands = mark_candidates(cands, gold["gold_ids"], gold.get("anchor_id"))
        n_in = int(cands.gold.eq("gold").sum())
        anchor = "yes" if cands.gold.eq("anchor").any() else "no"
        st.caption(f"anchor among candidates: {anchor} · answer rows among candidates: {n_in}/{len(gold['gold_ids'])} "
                   "(only lookups expect the answer itself here)")
    cols = [c for c in CAND_COLS if c in cands.columns]
    st.dataframe(cands[cols], hide_index=True, height=height,
                 column_config={"id": st.column_config.TextColumn(width="small"), "gold": st.column_config.TextColumn(width="small")})


def show_run(r: dict, verdict: dict | None = None, gold: dict | None = None):
    """One model's answer: places, candidates, SQL, rows and the map. `gold` is a benchmark row (as dict)."""
    if verdict is not None:
        st.markdown("✅ **matches gold**" if verdict["match"] else f"❌ **{verdict['reason']}**")
    if r.get("recorded"):
        st.caption("replayed from the committed zero-shot predictions: it answered the benchmark prompt, so the candidates below are the benchmark's")
    if gold is not None and gold.get("gold_places"):
        ok = places_match(r["places"], gold["gold_places"])
        st.write(f"**places** {'✅' if ok else '❌'} `{r['places']}`" + ("" if ok else f" · gold `{gold['gold_places']}`") + f" · {r['seconds']['places']}s")
    else:
        st.write(f"**places** `{r['places']}` · {r['seconds']['places']}s")
    st.caption("candidates from fuzzy search (what the SQL prompt contained)")
    candidate_table(r["candidates"], gold)
    st.code(format_sql(r["sql"]) if r["sql"] else r["sql_raw"], language="sql")
    with st.expander("exact prompt sent for the SQL step"):
        st.code(r["prompt_user"], language="text")
    if r["reasoning"]["sql"]:
        with st.expander("model reasoning"):
            st.write(r["reasoning"]["sql"])
    if r["error"]:
        st.error(r["error"])
        return
    names = r["result"]["name"].astype(str).tolist() if "name" in r["result"] else []
    st.write(f"**{len(r['result'])} feature(s)** · {r['seconds']['sql']}s" + (": " + ", ".join(names[:15]) if names else ""))
    if len(r["result"]) and "geometry" in r["result"]:
        st_folium(map_features(r["result"], anchors=r["anchors"]), height=MAP_HEIGHT, use_container_width=True, returned_objects=[])


def show_gold(row: dict, con, geodata_path):
    """The benchmark's side of the question: gold places, the candidates it was asked with, gold SQL, gold features."""
    if row["gold_places"]:
        st.write(f"**places** `{row['gold_places']}`")
    st.caption("candidates in the benchmark prompt")
    candidate_table(row["gold_candidates"], row)
    st.code(format_sql(row["gold_sql"]), language="sql")
    gold_df = pd.concat([fetch_by_id(con, geodata_path, i, geometry=True) for i in row["gold_ids"]], ignore_index=True) if row["gold_ids"] else pd.DataFrame()
    names = gold_df["name"].astype(str).tolist() if "name" in gold_df else []
    st.write(f"**{len(gold_df)} feature(s)**" + (": " + ", ".join(names[:15]) if names else ""))
    if len(gold_df):
        st_folium(map_features(gold_df), height=MAP_HEIGHT, use_container_width=True, returned_objects=[])


st.title("gazet playground")
st.caption("One question, several models, one map each. The pipeline is the one gazet serves: extract places → fuzzy search → write SQL → execute in DuckDB.")

tab_ask, tab_bench = st.tabs(["Ask", "Benchmark"])

# ── Ask ─────────────────────────────────────────────────────────────────────
with tab_ask:
    c1, c2 = st.columns([3, 2])
    question = c1.text_input("Question", value=st.session_state.get("question", EXAMPLES[0]))
    pick = c2.selectbox("…or pick an example", ["(examples)"] + EXAMPLES, index=0)
    if pick != "(examples)":
        question = pick
    st.session_state.question = question
    run = st.button("Geocode", type="primary", disabled=not live)
    if not live:
        st.info("No live endpoint. Start `llama serve` (see docs/take-it-home.md).")
    if run and live:
        cols = st.columns(len(live))
        for col, ep in zip(cols, live):
            with col:
                st.subheader(ep.name)
                with st.spinner("asking …"):
                    try:
                        r = geocode(ep, question, con, geodata_path, prompts)
                    except Exception as exc:
                        st.error(str(exc)[:300])
                        continue
                show_run(r)

# ── Benchmark ───────────────────────────────────────────────────────────────
with tab_bench:
    runs = st.session_state.setdefault("runs", {})       # (split, sample_id, model) -> {"r": ..., "verdict": ...}

    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 2])
    split = f1.selectbox("split", [s for s in SPLITS if s in bench])
    table = benchmark_rows(bench, split)
    recorded = load_recorded(baseline_dir, split)
    # The benchmark view is the subset gpt-oss was scored on (40 questions per split): its committed
    # zero-shot predictions are replayed here without a model, so a live gpt-oss endpoint is not run.
    use_recorded = bool(recorded)
    sub = table[table.sample_id.isin(recorded)] if use_recorded else table
    if use_recorded and not any(k[0] == split and k[2] == RECORDED for k in runs):
        with st.spinner(f"replaying {len(recorded)} recorded gpt-oss predictions …"):
            for row in sub.to_dict(orient="records"):
                r = replay(row, recorded[row["sample_id"]], con, geodata_path)
                runs[(split, row["sample_id"], RECORDED)] = {"r": r, "verdict": score(r, row["gold_ids"], row["ordered"])}
    runners = [e for e in live if not (use_recorded and e.kind == "gpt-oss")]
    families = f2.multiselect("family", sorted(sub.task_family.dropna().unique()))
    if families:
        sub = sub[sub.task_family.isin(families)]
    templates = f3.multiselect("template", sorted(sub.template_id.dropna().unique()))
    if templates:
        sub = sub[sub.template_id.isin(templates)]
    if split == "noisy":
        noise = f4.multiselect("noise", sorted(sub.noise_type.dropna().unique()))
        if noise:
            sub = sub[sub.noise_type.isin(noise)]
    else:
        f4.empty()
    text = f5.text_input("question contains", "")
    if text:
        sub = sub[sub.question.str.contains(text, case=False, regex=False)]

    # verdict columns: live models first, then anything else already run on this split
    models = [e.name for e in runners] + ([RECORDED] if use_recorded else [])
    models += sorted({k[2] for k in runs if k[0] == split} - set(models) - ({RECORDED} if not use_recorded else set()))
    view = sub[["question", "task_family", "template_id", "noise_type", "anchor_name"]].copy()
    if split != "noisy":
        view = view.drop(columns=["noise_type"])
    for name in models:
        view[name] = [("✅" if runs[(split, sid, name)]["verdict"]["match"] else "❌") if (split, sid, name) in runs else ""
                      for sid in sub.sample_id]
    ran = [m for m in models if (view[m] != "").any()]
    if ran:
        show = st.radio("show", ["all", "any model failed", "all models failed", "all passed"] + [f"{m} failed" for m in ran],
                        horizontal=True)
        marks = view[ran]
        if show == "any model failed":
            view = view[(marks == "❌").any(axis=1)]
        elif show == "all models failed":
            view = view[((marks == "❌") | (marks == "")).all(axis=1) & (marks == "❌").any(axis=1)]
        elif show == "all passed":
            view = view[(marks == "✅").all(axis=1)]
        elif show.endswith(" failed"):
            view = view[view[show[:-len(" failed")]] == "❌"]
    sub = sub.loc[view.index]

    c1, c2, c3 = st.columns([1, 2, 4])
    n = c1.number_input("rows to run", min_value=1, max_value=max(1, len(view)), value=min(10, max(1, len(view))))
    go = c2.button("Run through all models", type="primary", disabled=not runners or not len(view))
    c3.caption(f"{len(view)} question(s) in {split}" + (" (the subset gpt-oss was scored on; its predictions are replayed, not re-run)" if use_recorded else "")
               + ". Select rows to run exactly those, or run the first *n*. Each run goes through the full pipeline and is kept for this session.")
    event = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="multi-row", key=f"table_{split}",
                         width="stretch", height=min(420, 40 + 35 * max(1, len(view))))
    picked = list(event.selection.rows)
    chosen = sub.iloc[picked] if picked else sub.head(int(n))

    if go and runners:
        for ep in runners:
            todo = [row for row in chosen.itertuples() if (split, row.sample_id, ep.name) not in runs]
            bar = st.progress(0.0, text=ep.name)
            for i, row in enumerate(todo, 1):
                try:
                    r = geocode(ep, row.question, con, geodata_path, prompts)
                    verdict = score(r, row.gold_ids, row.ordered)
                except Exception as exc:
                    r, verdict = None, {"match": False, "reason": str(exc)[:200]}
                runs[(split, row.sample_id, ep.name)] = {"r": r, "verdict": verdict}
                bar.progress(i / len(todo), text=f"{ep.name}: {i}/{len(todo)}")
            bar.empty()
        st.rerun()

    # detail: one question, gold next to every model that ran it
    with_runs = sub[[any((split, sid, m) in runs for m in models) for sid in sub.sample_id]]
    if len(with_runs):
        default = 0
        if len(picked) == 1 and sub.iloc[picked[0]].sample_id in set(with_runs.sample_id):
            default = with_runs.sample_id.tolist().index(sub.iloc[picked[0]].sample_id)
        which = st.selectbox("inspect", with_runs.question.tolist(), index=default)
        row = with_runs[with_runs.question == which].iloc[0]
        st.markdown(f"**{row.task_family} · {row.template_id}**" + (f" · {row.noise_type}" if pd.notna(row.noise_type) else ""))
        ran_here = [m for m in models if (split, row.sample_id, m) in runs]
        cols = st.columns(1 + len(ran_here))
        gold = row.to_dict()
        with cols[0]:
            st.subheader("gold")
            show_gold(gold, con, geodata_path)
        for col, name in zip(cols[1:], ran_here):
            with col:
                st.subheader(name)
                hit = runs[(split, row.sample_id, name)]
                if hit["r"] is None:
                    st.error(hit["verdict"]["reason"])
                else:
                    show_run(hit["r"], hit["verdict"], gold)
