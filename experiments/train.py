"""Hydra entrypoint: compose a config, run it on Modal, keep the result next to the config.

    python experiments/train.py                                   # notebook-2 defaults on a T4
    python experiments/train.py +experiment=smoke                 # 20 steps, 5 eval rows
    python experiments/train.py train.lr=1e-4 train.epochs=2 run.name=lr1e-4-ep2
    python experiments/train.py -m train.lr=5e-5,1e-4,2e-4 train.lora_dropout=0,0.05
    python experiments/train.py -m data=bundled,half run.gpu=L4
    python experiments/train.py +experiment=smoke run.dry_run=true   # just print the config

Every job writes result.json (final benchmark scores, train minutes) into its Hydra output
dir; `python experiments/summarize.py experiments/multirun/<date>/<time>` tabulates a sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gazet_tutorial.train import TrainConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))


def to_train_config(cfg: DictConfig) -> TrainConfig:
    flat = {**OmegaConf.to_container(cfg.train, resolve=True), **OmegaConf.to_container(cfg.eval, resolve=True)}
    flat["data_fraction"] = float(cfg.data.fraction)
    flat["name"] = cfg.run.name
    return TrainConfig.from_dict(flat)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    tc = to_train_config(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("TrainConfig:", tc)
    if cfg.run.dry_run:
        return

    import modal
    from modal_train import DEFAULT_GPU, app, train, upload_artifacts

    # detach=True: the Modal app keeps running (and writes runs/<name>/ to the volume) even if this
    # laptop loses its connection; `summarize.py --volume` collects results afterwards.
    with modal.enable_output(), app.run(detach=True):
        remote_artifacts = upload_artifacts(Path(cfg.data.artifacts), cfg.data.name)
        fn = train.with_options(gpu=cfg.run.gpu) if cfg.run.gpu != DEFAULT_GPU else train
        result = fn.remote(tc.to_dict(), artifacts=remote_artifacts, project=cfg.run.project, gpu=cfg.run.gpu)

    (out_dir / "result.json").write_text(json.dumps({"config": OmegaConf.to_container(cfg, resolve=True), **result}, indent=1))
    final = result["final"]
    print(f"\n{result['name']}: {result['train_minutes']} min on {cfg.run.gpu}")
    for split in sorted({k.split('/')[0] for k in final}):
        print(f"  {split:14s} sql_exec {final.get(f'{split}/sql_exec', float('nan')):.2f}   places_f1 {final.get(f'{split}/places_f1', float('nan')):.2f}")
    print(f"result -> {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
