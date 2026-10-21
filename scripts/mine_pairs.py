#!/usr/bin/env python
"""Mine DPO preference pairs from the fine-tuned model's own samples, verified by execution.

Each prompt is sampled K times at a temperature; every SQL answer is executed and every
extraction parsed. Prompts with at least one passing and one failing answer become a pair
(see gazet_tutorial.dpo.choose_pair). Output goes to artifacts/dpo/ and ships in the wheel,
so notebook 3 never has to do this on a T4:

    pairs.jsonl   prompt messages + chosen + rejected
    stats.json    how many prompts were sampled and what the yield was

Where the prompts come from matters more than anything else here. On its own training
distribution the SFT model is consistent: `--source dataset` (validation rows plus a sample
of training rows) yields *no* pairs even at temperature 1.3, every sample passes. The model
is only inconsistent where it is weak, so the default `--source benchmark` samples the
`noisy` and `compositional` rows that are NOT in the fixed 40-question subset the benchmark
scores. The subset stays untouched as the before/after test; the pairs come from the same
kind of question, which is exactly what the notebook then discusses.

Examples
    python scripts/mine_pairs.py                                   # published reference GGUF, K=6, T=0.8
    python scripts/mine_pairs.py work/gguf/reference-q8_0.gguf --k 8
    python scripts/mine_pairs.py --source dataset --limit 40       # see the yield problem yourself
    python scripts/mine_pairs.py --limit 20                        # smoke test

Samples are cached in work/dpo/samples.jsonl so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_gguf import NO_THINKING, default_name, serve  # noqa: E402

from gazet_tutorial import artifacts_dir, data_dir  # noqa: E402
from gazet_tutorial.baseline import chat_completion, prompt_of  # noqa: E402
from gazet_tutorial.benchmark import load_benchmark, load_subset_ids  # noqa: E402
from gazet_tutorial.data import load_jsonl  # noqa: E402
from gazet_tutorial.dpo import choose_pair  # noqa: E402
from gazet_tutorial.eval import execution_match, gold_of, places_prf, strip_fences  # noqa: E402
from gazet_tutorial.geodata import connect, ensure_geodata  # noqa: E402

DEFAULT_MODEL = "developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF:Q8_0"
MAX_TOKENS = {"sql": 256, "places": 64}


def verdict(row: dict, answer: str, task: str, con, geo) -> dict:
    gold = gold_of(row)
    if task == "sql":
        meta = row["metadata"]
        ordered = bool(meta.get("ordered", meta.get("task_family") == "aggregation"))
        r = execution_match(con, strip_fences(answer), gold, geo, ordered=ordered)
        return {"pass": bool(r["exec_match"]), "sql_valid": bool(r["sql_valid"]), "error": r["error"]}
    return {"pass": places_prf(answer, gold)["f1"] == 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="GGUF path or Hugging Face spec")
    ap.add_argument("--base-url", help="an already running server instead of starting llama serve")
    ap.add_argument("--k", type=int, default=6, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--source", choices=["benchmark", "dataset"], default="benchmark",
                    help="benchmark: noisy + compositional rows outside the scored subset (default); dataset: val + train rows")
    ap.add_argument("--n-train", type=int, default=350, help="--source dataset: training rows per task added to the validation rows")
    ap.add_argument("--dataset-dir", default=str(artifacts_dir() / "dataset"))
    ap.add_argument("--bench-dir", default=str(artifacts_dir() / "benchmark"))
    ap.add_argument("--out-dir", default=str(artifacts_dir() / "dpo"))
    ap.add_argument("--geodata")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, help="prompts per task; for smoke tests")
    ap.add_argument("--seed", type=int, default=108)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rng = random.Random(args.seed)
    prompts: list[tuple[str, dict]] = []
    if args.source == "dataset":
        ds = Path(args.dataset_dir)
        for task in ("sql", "places"):
            val = load_jsonl(ds / task / "val.jsonl")
            train = load_jsonl(ds / task / "train.jsonl")
            rng.shuffle(train)
            rows = val + train[: args.n_train]
            prompts += [(task, r) for r in rows[: args.limit] if args.limit] if args.limit else [(task, r) for r in rows]
    else:
        bench, subset = load_benchmark(args.bench_dir), load_subset_ids(args.bench_dir)
        for split in ("noisy", "compositional"):
            for task in ("sql", "places"):
                scored = set(subset.get(split, {}).get(task, []))
                rows = [r for r in bench[split][task] if r["metadata"]["sample_id"] not in scored]
                for r in rows[: args.limit] if args.limit else rows:
                    r["metadata"]["split"] = split
                    prompts.append((task, r))
    print(f"source={args.source}: {len(prompts)} prompts × {args.k} samples at T={args.temperature}")

    work = data_dir() / "dpo"
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "samples.jsonl"
    cache: dict[tuple[str, str], list[str]] = {}   # (task, sample_id): both tasks share sample ids
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["k"] >= args.k and rec["temperature"] == args.temperature:
                    cache[(rec["task"], rec["sample_id"])] = rec["samples"][: args.k]
    todo = [(t, r) for t, r in prompts if (t, r["metadata"]["sample_id"]) not in cache]
    print(f"{len(cache)} cached, {len(todo)} to sample")

    if todo:
        name = default_name(args.model)
        base_url = args.base_url or serve(args.model, name, args.port, 4096, args.parallel, work, 900)

        def sample(item):
            task, row = item
            msgs = prompt_of(row)
            return [chat_completion(msgs, model=name, base_url=base_url, temperature=args.temperature,
                                    max_tokens=MAX_TOKENS[task], reasoning_effort=None, extra_body=NO_THINKING)["content"]
                    for _ in range(args.k)]

        t0 = time.time()
        with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.parallel) as ex:
            for i, ((task, row), outs) in enumerate(zip(todo, ex.map(sample, todo)), 1):
                sid = row["metadata"]["sample_id"]
                cache[(task, sid)] = outs
                f.write(json.dumps({"sample_id": sid, "task": task, "k": args.k, "temperature": args.temperature, "samples": outs}, ensure_ascii=False) + "\n")
                f.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  sampled {i}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)")

    con = connect(spatial=True)
    geo = ensure_geodata(path=args.geodata, con=connect(spatial=True, httpfs=True))
    pairs, stats = [], {"prompts": {}, "mixed": {}, "always_pass": {}, "always_fail": {}, "by_family": {}}
    for task, row in prompts:
        sid = row["metadata"]["sample_id"]
        outs = cache[(task, sid)]
        vs = [verdict(row, o, task, con, geo) for o in outs]
        n_pass = sum(v["pass"] for v in vs)
        fam = row["metadata"].get("task_family", "?")
        stats["prompts"][task] = stats["prompts"].get(task, 0) + 1
        key = "mixed" if 0 < n_pass < len(vs) else ("always_pass" if n_pass == len(vs) else "always_fail")
        stats[key][task] = stats[key].get(task, 0) + 1
        pair = choose_pair(outs, vs)
        if pair is None:
            continue
        stats["by_family"].setdefault(task, {}).setdefault(fam, 0)
        stats["by_family"][task][fam] += 1
        pairs.append({"sample_id": sid, "task": task, "task_family": fam, "template_id": row["metadata"].get("template_id"),
                      "split": row["metadata"].get("split", "?"), "noise_type": row["metadata"].get("noise_type"),
                      "prompt": prompt_of(row), "chosen": pair[0], "rejected": pair[1], "gold": gold_of(row),
                      "rejected_error": next((v.get("error") for o, v in zip(outs, vs) if o == pair[1]), None),
                      "n_pass": n_pass, "k": len(vs)})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    stats.update({"source": args.source, "model": args.model, "k": args.k, "temperature": args.temperature, "pairs": len(pairs)})
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"\n{len(pairs)} pairs -> {out / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
