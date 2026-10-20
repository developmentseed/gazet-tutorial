"""Tabulate experiment results.

    python experiments/summarize.py experiments/multirun/2026-10-19/14-35-08      # a Hydra run/multirun dir
    python experiments/summarize.py --volume reference regularized wide-slow      # straight from the Modal volume
    python experiments/summarize.py --volume                                      # every run on the volume

The volume form fetches runs/<name>/{eval,config}.json into work/runs/, which is what you
need when the laptop lost its connection mid-sweep (the detached app finishes anyway).
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["iid", "unseen_region", "unseen_anchor", "noisy", "paraphrase", "compositional"]


def from_volume(names: list[str]) -> list[dict]:
    modal = str(ROOT / ".venv" / "bin" / "modal")
    if not names:
        out = subprocess.run([modal, "volume", "ls", "gazet-tutorial", "runs"], capture_output=True, text=True, check=True).stdout
        names = [line.strip().split("/")[-1] for line in out.splitlines() if line.strip().startswith("runs/")]
    rows = []
    for name in names:
        dest = ROOT / "work" / "runs" / name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("eval.json", "config.json"):
            subprocess.run([modal, "volume", "get", "--force", "gazet-tutorial", f"runs/{name}/{f}", str(dest)], capture_output=True)
        if not (dest / "eval.json").exists():
            print(f"{name}: no eval.json yet (unfinished or killed)")
            continue
        ev = json.loads((dest / "eval.json").read_text())
        cfg = json.loads((dest / "config.json").read_text())
        rows.append({"name": name, "min": ev.get("train_minutes"), "gpu": ev.get("gpu"), "frac": cfg.get("data_fraction"),
                     **{k: cfg.get(k) for k in ("lr", "epochs", "lora_r", "lora_alpha", "lora_dropout", "weight_decay")},
                     **{s: round(ev["final"].get(f"{s}/sql_exec", float("nan")), 2) for s in SPLITS},
                     **{f"{s}_f1": round(ev["final"].get(f"{s}/places_f1", float("nan")), 2) for s in SPLITS}})
    return rows


def from_hydra(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.rglob("result.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        rows.append({"name": r["name"], "min": r["train_minutes"], "gpu": c["run"]["gpu"], "frac": c["data"]["fraction"],
                     **{k: c["train"][k] for k in ("lr", "epochs", "lora_r", "lora_alpha", "lora_dropout", "weight_decay")},
                     **{s: round(r["final"].get(f"{s}/sql_exec", float("nan")), 2) for s in SPLITS},
                     **{f"{s}_f1": round(r["final"].get(f"{s}/places_f1", float("nan")), 2) for s in SPLITS}})
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--volume":
        rows, out = from_volume(args[1:]), ROOT / "work" / "runs" / "summary.csv"
    else:
        root = Path(args[0]) if args else ROOT / "experiments" / "multirun"
        rows, out = from_hydra(root), root / "summary.csv"
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False) if len(df) else "no finished runs")
    if len(df):
        df.to_csv(out, index=False)
        print(f"\nwritten {out}")
