#!/usr/bin/env python
"""Zero-shot gpt-oss:20b (or any OpenAI-compatible model) on the benchmark subset.

Runs offline, once, by the maintainer; the predictions are committed under
artifacts/baselines/<model>/ and scored in notebook 2 with the same code as the
fine-tuned model. Resumable: existing predictions are skipped.

Examples
    # local Ollama signed in to Ollama cloud (default endpoint and model)
    ollama signin
    python scripts/run_baseline.py --limit 5            # smoke test
    python scripts/run_baseline.py                      # full subset

    # direct Ollama cloud endpoint
    OLLAMA_API_KEY=... python scripts/run_baseline.py --base-url https://ollama.com/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gazet_tutorial.baseline import DEFAULT_BASE_URL, DEFAULT_MODEL, run_rows  # noqa: E402
from gazet_tutorial.benchmark import SPLITS, TASKS, load_benchmark, load_subset_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", default=str(ROOT / "artifacts" / "benchmark"))
    ap.add_argument("--out-dir", default=None, help="default: artifacts/baselines/<model-slug>/zero")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--reasoning", default="low", help="reasoning_effort for gpt-oss ('' to omit)")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--all", action="store_true", help="ignore subset_ids.json and run every row")
    ap.add_argument("--limit", type=int, default=None, help="rows per (task, split); for smoke tests")
    args = ap.parse_args()

    bench = load_benchmark(args.bench_dir)
    subset = {} if args.all else load_subset_ids(args.bench_dir)
    slug = args.model.replace(":", "-").replace("/", "-")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "baselines" / slug / "zero"
    print(f"model={args.model} endpoint={args.base_url} -> {out_dir}")

    for split in args.splits:
        for task in args.tasks:
            rows = bench.get(split, {}).get(task)
            if not rows:
                print(f"[{task}/{split}] no rows, skipped")
                continue
            ids = subset.get(split, {}).get(task)
            if ids is not None:
                keep = set(ids)
                rows = [r for r in rows if r["metadata"]["sample_id"] in keep]
            if args.limit:
                rows = rows[: args.limit]
            run_rows(rows, out_dir / f"{task}_{split}.jsonl", task=task, split=split,
                     model=args.model, base_url=args.base_url,
                     reasoning_effort=args.reasoning or None)
    print("done")


if __name__ == "__main__":
    main()
