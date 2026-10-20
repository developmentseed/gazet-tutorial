"""The Modal side of the experiments: one GPU function that trains, benchmarks and saves a run.

Driven by experiments/train.py (Hydra). Direct use is possible too:

    modal run experiments/modal_train.py          # smoke test with defaults (20 steps)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

DEFAULT_GPU = os.environ.get("GAZET_GPU", "T4")   # T4 mirrors Colab; train.py overrides per run via with_options
VOL = "/mnt/gt"
PROJECT = "gazet-tutorial"

app = modal.App(PROJECT)
vol = modal.Volume.from_name("gazet-tutorial", create_if_missing=True)

# Same pins as notebook 2's install cell, so what we learn here transfers to Colab.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "triton>=3.3.0", "torchvision", "torchaudio==2.8.0", "bitsandbytes", "xformers==0.0.32.post2",
        # the notebook installs these from git HEAD; the zip archive of main is the same code and needs no git in the image
        "unsloth_zoo[base] @ https://github.com/unslothai/unsloth-zoo/archive/refs/heads/main.zip",
        "unsloth[base] @ https://github.com/unslothai/unsloth/archive/refs/heads/main.zip",
    )
    .pip_install("tokenizers>=0.22.0,<=0.23.0", "trl==0.22.2", extra_options="--no-deps")
    .pip_install("transformers==5.2.0", "trackio>=0.42", "datasets",
                 "duckdb>=1.4", "pandas>=2.2", "pyarrow>=17", "sqlparse>=0.5", "shapely>=2.0", "geopandas>=1.0")
    .run_commands("unsloth install-kernels || true")
    .env({
        "GAZET_TUTORIAL_DATA": f"{VOL}/work",     # geodata parquet, cached on the volume after the first pull
        "HF_HOME": f"{VOL}/hf",                   # base-model weights, cached on the volume
        "TRACKIO_DIR": f"{VOL}/trackio",          # local SQLite logs; no Space, no Hub
    })
    .add_local_python_source("gazet_tutorial")
)


def upload_artifacts(local: Path, name: str) -> str:
    """Put a notebook-1 artifacts folder on the volume as artifacts/<name>; returns the remote path."""
    remote = f"artifacts/{name}"
    with vol.batch_upload(force=True) as batch:
        for sub in ("dataset", "benchmark"):
            batch.put_directory(str(local / sub), f"{remote}/{sub}")
    return f"{VOL}/{remote}"


@app.function(image=image, gpu=DEFAULT_GPU, timeout=4 * 3600, volumes={VOL: vol})
def train(cfg_dict: dict, artifacts: str = f"{VOL}/artifacts/bundled", project: str = PROJECT, gpu: str = DEFAULT_GPU) -> dict:
    os.environ["GAZET_TUTORIAL_ARTIFACTS"] = artifacts
    from gazet_tutorial.benchmark import SPLITS, load_benchmark, load_subset_ids
    from gazet_tutorial.geodata import connect, ensure_geodata
    from gazet_tutorial.train import (
        TrainConfig,
        benchmark_eval,
        build_model,
        build_trainer,
        load_train_val,
        make_benchmark_callback,
        save_run,
    )

    cfg = TrainConfig.from_dict(cfg_dict)
    cfg = cfg.with_overrides(name=cfg.run_name())
    name = cfg.name
    art = Path(artifacts)
    run_dir = Path(VOL) / "runs" / name
    print(f"[{name}] gpu={gpu} artifacts={artifacts} config={cfg.to_dict()}")

    con = connect(spatial=True)
    geodata_path = ensure_geodata(con=connect(spatial=True, httpfs=True))
    vol.commit()
    bench = load_benchmark(art / "benchmark")
    subset = load_subset_ids(art / "benchmark")

    model, tokenizer = build_model(cfg)
    train_ds, val_ds = load_train_val(art / "dataset", tokenizer, cfg)
    print(f"[{name}] train {len(train_ds)} val {len(val_ds)}")

    def evaluate(m, t):
        return benchmark_eval(m, t, bench, con, geodata_path, cfg.bench_n, cfg.bench_splits, subset, gen_batch=cfg.gen_batch)

    cb = make_benchmark_callback(evaluate, every=cfg.eval_every)
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg, run_dir / "trainer", report_to="trackio",
                            project=project, callbacks=[cb])
    cb.trainer = trainer
    t0 = time.time()
    trainer.train()
    train_min = round((time.time() - t0) / 60, 1)

    # save the adapter *before* the final benchmark: if the run is cancelled during eval, the model survives
    save_run(run_dir, model, tokenizer, cfg, {}, trainer.state.log_history,
             extra={"during_training": cb.history, "train_minutes": train_min, "gpu": gpu, "artifacts": artifacts, "final_pending": True})
    vol.commit()
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    final = benchmark_eval(model, tokenizer, bench, con, geodata_path, cfg.final_n, SPLITS, subset, gen_batch=cfg.gen_batch)
    save_run(run_dir, model, tokenizer, cfg, final, trainer.state.log_history,
             extra={"during_training": cb.history, "train_minutes": train_min, "gpu": gpu, "artifacts": artifacts})
    vol.commit()
    if cfg.save_merged:
        model.save_pretrained_merged(str(run_dir / "merged"), tokenizer, save_method="merged_16bit")
        vol.commit()
        print(f"[{name}] merged 16-bit weights saved to {run_dir / 'merged'}")
    print(f"[{name}] done in {train_min} min: " + " ".join(f"{k}={v:.2f}" for k, v in final.items() if k.endswith("sql_exec")))
    return {"name": name, "final": final, "train_minutes": train_min, "run_dir": str(run_dir)}


@app.local_entrypoint()
def main():
    """Smoke test without Hydra: 20 steps on the bundled artifacts."""
    from gazet_tutorial.train import TrainConfig

    root = Path(__file__).resolve().parents[1]
    remote = upload_artifacts(root / "artifacts", "bundled")
    cfg = TrainConfig(max_steps=20, warmup_steps=2, eval_every=10, bench_n=5, val_cap=32, name="smoke")
    print(train.remote(cfg.to_dict(), artifacts=remote))
