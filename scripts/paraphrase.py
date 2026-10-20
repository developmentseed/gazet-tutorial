#!/usr/bin/env python
"""Build the `paraphrase` benchmark split with an LLM (offline, once).

Takes questions from the iid / unseen_region / unseen_anchor SQL splits (the
baseline subset by default), asks the model to rewrite each one in different
words while keeping every place name verbatim, and keeps a rewrite only if it
differs from the original and still contains all the gold place names. Gold
SQL, candidates and ids are unchanged, so the split measures surface
generalisation only. Spot-check the output before committing it.

    python scripts/paraphrase.py --limit 3     # smoke
    python scripts/paraphrase.py               # subset (≈40 per source split)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gazet_tutorial.baseline import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion  # noqa: E402
from gazet_tutorial.benchmark import load_benchmark, load_subset_ids, question_in_user  # noqa: E402
from gazet_tutorial.data import save_jsonl  # noqa: E402

SYSTEM = (
    "You rewrite short geographic questions. Keep the meaning identical, keep every place name, "
    "number and unit exactly as written, and change the wording and sentence shape as much as you can. "
    "Reply with the rewritten question only."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", default=str(ROOT / "artifacts" / "benchmark"))
    ap.add_argument("--sources", nargs="*", default=["iid", "unseen_region", "unseen_anchor"])
    ap.add_argument("--per-split", type=int, default=40)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=108)
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    bench = load_benchmark(bench_dir)
    subset = load_subset_ids(bench_dir)
    rng = random.Random(args.seed)
    sql_out, places_out, rejected = [], [], 0

    for split in args.sources:
        sql_rows = bench[split]["sql"]
        places_by_id = {r["metadata"]["sample_id"]: r for r in bench[split]["places"]}
        ids = subset.get(split, {}).get("sql") or [r["metadata"]["sample_id"] for r in sql_rows]
        rows = [r for r in sql_rows if r["metadata"]["sample_id"] in set(ids)]
        rng.shuffle(rows)
        rows = rows[: args.limit or args.per_split]
        for r in rows:
            user = r["messages"][1]["content"]
            q = question_in_user(user)
            gold_places = [p["place"] for p in json.loads(places_by_id[r["metadata"]["sample_id"]]["messages"][2]["content"])["places"]]
            res = chat_completion([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                  model=args.model, base_url=args.base_url, max_tokens=80)
            new_q = res["content"].strip().strip('"').splitlines()[0].strip() if res["content"].strip() else ""
            ok = (new_q and new_q.lower() != q.lower()
                  and all(re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", new_q) for p in gold_places)
                  and sorted(re.findall(r"\d+", new_q)) == sorted(re.findall(r"\d+", q)))
            print(f"[{split}] {q!r} -> {new_q!r}  {'ok' if ok else 'REJECTED'}")
            if not ok:
                rejected += 1
                continue
            sid = f"{r['metadata']['sample_id']}_para"
            s = copy.deepcopy(r)
            s["messages"][1]["content"] = user.replace(q, new_q)
            s["metadata"].update({"sample_id": sid, "paraphrase_of": r["metadata"]["sample_id"],
                                  "source_split": split, "original_question": q})
            sql_out.append(s)
            p = copy.deepcopy(places_by_id[r["metadata"]["sample_id"]])
            p["messages"][1]["content"] = new_q
            p["metadata"].update({"sample_id": sid, "paraphrase_of": r["metadata"]["sample_id"],
                                  "source_split": split, "original_question": q})
            places_out.append(p)

    save_jsonl(sql_out, bench_dir / "sql" / "paraphrase.jsonl")
    save_jsonl(places_out, bench_dir / "places" / "paraphrase.jsonl")
    ids_path = bench_dir / "subset_ids.json"
    subset = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    ids = [s["metadata"]["sample_id"] for s in sql_out]
    subset["paraphrase"] = {"sql": ids[:40], "places": ids[:20]}   # same subset sizes as the other splits
    ids_path.write_text(json.dumps(subset, indent=1))
    print(f"kept {len(sql_out)} paraphrases, rejected {rejected}; wrote benchmark/{{sql,places}}/paraphrase.jsonl")


if __name__ == "__main__":
    main()
