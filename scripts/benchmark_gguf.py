#!/usr/bin/env python
"""Score a GGUF on the benchmark subset and compare it with gpt-oss:20b and the base model.

Takes the q8_0 GGUF that notebook 2 exports, serves it with `llama serve` (llama.app CLI),
asks it every question of the fixed benchmark subset (40 SQL + 20 extraction per split,
the same rows the committed gpt-oss:20b predictions cover), scores the answers by
*executing* the SQL against the real table, and prints one table per task with every
model found under artifacts/baselines/ and work/benchmark/.

Examples
    python scripts/benchmark_gguf.py gazet-in-q8_0.gguf                 # serve, predict, score, compare
    python scripts/benchmark_gguf.py gazet-in-q8_0.gguf --limit 3       # smoke test (3 rows per split/task)
    python scripts/benchmark_gguf.py unsloth/Qwen3.5-0.8B-GGUF:Q8_0     # a Hugging Face GGUF (llama downloads it)
    python scripts/benchmark_gguf.py --base-url http://localhost:8081/v1 --name gazet-in   # a server you started
    python scripts/benchmark_gguf.py --compare-only                     # just print the tables again

Predictions are written to work/benchmark/<name>/<task>_<split>.jsonl and are resumable:
re-running after an interruption only asks the questions that are missing. The geodata
(India divisions, ~60 MB) is downloaded from Overture on first use unless --geodata points
at a local copy.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gazet_tutorial import artifacts_dir, data_dir  # noqa: E402
from gazet_tutorial.baseline import headline, run_rows, score_predictions  # noqa: E402
from gazet_tutorial.benchmark import SPLITS, TASKS, load_benchmark, load_subset_ids  # noqa: E402
from gazet_tutorial.eval import failures  # noqa: E402
from gazet_tutorial.geodata import connect, ensure_geodata  # noqa: E402

MAX_TOKENS = {"sql": 256, "places": 64}                      # what the notebook used for generation
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}   # Qwen3.5: same marker as training


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()


def default_name(model: str) -> str:
    p = Path(model)
    if p.exists():
        return slug(p.stem)
    repo, _, quant = model.partition(":")          # unsloth/Qwen3.5-0.8B-GGUF:Q8_0 -> qwen3.5-0.8b-q8_0
    base = repo.split("/")[-1]
    base = re.sub(r"-?gguf$", "", base, flags=re.I)
    return slug(f"{base}-{quant}" if quant else base)


def wait_for(url: str, proc: subprocess.Popen, timeout: float, log: Path) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"llama serve exited with code {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    sys.exit(f"llama serve did not answer on {url} within {timeout:.0f}s; see {log}")


def serve(model: str, name: str, port: int, ctx: int, parallel: int, out_dir: Path, timeout: float) -> str:
    """Start `llama serve` for a local GGUF or a Hugging Face spec; stopped automatically at exit."""
    if not shutil_which("llama"):
        sys.exit("llama not found: install it from https://llama.app/docs/installation")
    cmd = ["llama", "serve", "--host", "127.0.0.1", "--port", str(port), "--jinja", "-ngl", "99",
           "-c", str(ctx), "-np", str(parallel), "-a", name]
    cmd += ["-m", model] if Path(model).exists() else ["-hf", model]
    log = out_dir / "llama-serve.log"
    print("$", " ".join(cmd), f"\n  (log: {log})")
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)

    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)

    atexit.register(stop)
    base_url = f"http://127.0.0.1:{port}/v1"
    wait_for(base_url + "/models", proc, timeout, log)
    return base_url


def shutil_which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


def predict(bench: dict, subset: dict, out_dir: Path, name: str, base_url: str,
            splits: list[str], limit: int | None, verbose: bool) -> None:
    for split in splits:
        for task in TASKS:
            rows = bench.get(split, {}).get(task)
            if not rows:
                continue
            ids = subset.get(split, {}).get(task, [])
            by_id = {r["metadata"]["sample_id"]: r for r in rows}
            rows = [by_id[i] for i in ids if i in by_id]
            if limit:
                rows = rows[:limit]
            t0 = time.time()
            recs = run_rows(rows, out_dir / f"{task}_{split}.jsonl", task=task, split=split,
                            model=name, base_url=base_url, reasoning_effort=None,
                            max_tokens=MAX_TOKENS[task], extra_body=NO_THINKING, verbose=verbose)
            lat = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
            print(f"[{task:6s}/{split:14s}] {len(recs):3d} predictions  ({time.time()-t0:5.1f}s this run, "
                  f"mean {sum(lat)/max(len(lat),1):.2f}s per call)")


def prediction_dirs(extra: Path | None) -> dict[str, Path]:
    """Every model we can score: committed baselines, then local runs under work/benchmark."""
    dirs: dict[str, Path] = {}
    for d in sorted((artifacts_dir() / "baselines").glob("*/zero")):
        dirs[d.parent.name + " zero-shot"] = d
    local = data_dir() / "benchmark"
    for d in sorted(p for p in local.glob("*") if p.is_dir()):
        dirs[d.name] = d
    if extra is not None and extra not in dirs.values():
        dirs[extra.name] = extra
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="path to a .gguf, or a Hugging Face spec like unsloth/Qwen3.5-0.8B-GGUF:Q8_0")
    ap.add_argument("--name", help="label for the tables and the output folder (default: from the file / spec)")
    ap.add_argument("--base-url", help="an already running OpenAI-compatible server; no llama serve is started")
    ap.add_argument("--out-dir", help="where predictions go (default: work/benchmark/<name>)")
    ap.add_argument("--bench-dir", default=str(artifacts_dir() / "benchmark"))
    ap.add_argument("--geodata", help="local divisions parquet (default: work/, downloaded from Overture if missing)")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--limit", type=int, help="rows per (task, split); for smoke tests")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--parallel", type=int, default=2, help="llama serve -np")
    ap.add_argument("--startup-timeout", type=float, default=900, help="seconds to wait for the server (HF downloads take a while)")
    ap.add_argument("--failures", type=int, default=2, help="wrong SQL answers to print per split for the scored model")
    ap.add_argument("--compare-only", action="store_true", help="skip prediction; score and tabulate what is on disk")
    ap.add_argument("--verbose", action="store_true", help="print every prediction as it arrives")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # progress shows up when piped to a log

    if not args.compare_only and not args.model and not args.base_url:
        ap.error("give a GGUF path / HF spec, or --base-url for a running server, or --compare-only")
    if args.model and args.base_url:
        ap.error("--base-url means the server is already running; do not also pass a model")

    scored: Path | None = None
    if not args.compare_only:
        name = args.name or (default_name(args.model) if args.model else None)
        if not name:
            ap.error("--name is required with --base-url")
        run_dir = data_dir() / "benchmark" / name          # log, per-question scores, tables
        scored = Path(args.out_dir) if args.out_dir else run_dir   # predictions
        run_dir.mkdir(parents=True, exist_ok=True)
        scored.mkdir(parents=True, exist_ok=True)
        bench = load_benchmark(args.bench_dir)
        subset = load_subset_ids(args.bench_dir)
        base_url = args.base_url or serve(args.model, name, args.port, args.ctx, args.parallel, run_dir, args.startup_timeout)
        print(f"model={name} endpoint={base_url} -> {scored}")
        predict(bench, subset, scored, name, base_url, args.splits, args.limit, args.verbose)
    else:
        bench = load_benchmark(args.bench_dir)
        subset = load_subset_ids(args.bench_dir)

    con = connect(spatial=True)
    geodata_path = ensure_geodata(path=args.geodata, con=connect(spatial=True, httpfs=True))

    dirs = prediction_dirs(scored)
    results = {}
    for label, d in dirs.items():
        r = score_predictions(bench, subset, d, con, geodata_path, splits=args.splits)
        if r:
            results[label] = r
    if not results:
        sys.exit("nothing to score")

    pd.set_option("display.width", 160)
    sql_table, places_table = headline(results, args.splits)
    print("\nSQL: execution accuracy (predicted and reference SQL return the same feature ids)")
    print(sql_table.round(3).to_string())
    print("\nPlace extraction: F1")
    print(places_table.round(3).to_string())

    if scored is not None:
        label = next((k for k, v in dirs.items() if v.resolve() == scored.resolve()), scored.name)
        mine = results.get(label, {})
        if args.failures:
            print(f"\nWrong SQL answers from {label}:")
            for split, per_task in mine.items():
                if "sql" not in per_task:
                    continue
                for _, row in failures(per_task["sql"], "exec_match", n=args.failures).iterrows():
                    print(f"\n  [{split}/{row.task_family}] {row.question}   -> {row.error}")
                    print("    gold:", re.sub(r"\s+", " ", row.gold)[:160])
                    print("    pred:", re.sub(r"\s+", " ", row.pred)[:160])
        for split, per_task in mine.items():
            for task, df in per_task.items():
                df.to_json(run_dir / f"scores_{task}_{split}.json", orient="records", force_ascii=False, indent=1)
        sql_table.to_csv(run_dir / "headline_sql.csv")
        places_table.to_csv(run_dir / "headline_places.csv")
        summary = {"sql_exec_match": sql_table.loc[label].to_dict() if label in sql_table.index else {},
                   "places_f1": places_table.loc[label].to_dict() if label in places_table.index else {}}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\nper-question scores and the tables are in {run_dir}")


if __name__ == "__main__":
    main()
