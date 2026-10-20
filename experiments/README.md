# Experiments

Notebook 2 is the tutorial. This folder is where we find out *what to put in it*: the same
training, run on a Modal GPU from the laptop, configured with Hydra, every run logged to a
local trackio database, the benchmark scored during training so over-fitting is visible as
a curve. Nothing here is needed in the workshop, but a Modal run is also the fallback if
Colab is down (see the end).

Training code is shared with the notebook through `gazet_tutorial.train`
(`TrainConfig`, `build_model`, `build_trainer`, `benchmark_eval`). The notebook reads
`TrainConfig()` defaults; `conf/config.yaml` mirrors them. When a study settles a better
value, change both.

## Setup (once)

```bash
uv pip install --python .venv/bin/python -e ".[experiments]"   # modal, hydra, trackio
.venv/bin/modal setup                                          # browser login -> ~/.modal.toml
```

The first run builds the Modal image from notebook 2's exact pins (a few minutes, then
cached), uploads `artifacts/` to the `gazet-tutorial` volume, and pulls India's boundaries
and the base-model weights into the volume once.

## Run

```bash
.venv/bin/python experiments/train.py +experiment=smoke                 # 20 steps, 5 eval rows: does everything work?
.venv/bin/python experiments/train.py                                   # notebook defaults on a T4 (~25 min)
.venv/bin/python experiments/train.py train.lr=1e-4 train.epochs=2 run.name=lr1e-4-ep2
.venv/bin/python experiments/train.py +experiment=regularized run.gpu=L4
.venv/bin/python experiments/train.py +experiment=smoke run.dry_run=true   # print the resolved config only
```

Sweeps are Hydra multiruns: every comma-separated value multiplies out, and the joblib
launcher sends the jobs to Modal in parallel (6 at a time by default):

```bash
.venv/bin/python experiments/train.py -m train.lr=5e-5,1e-4,2e-4 train.epochs=1,2
.venv/bin/python experiments/train.py -m train.lora_r=8,16,32 train.lora_dropout=0,0.05
.venv/bin/python experiments/train.py -m train.weight_decay=0.01,0.1 train.lora_alpha=16,32
.venv/bin/python experiments/train.py -m data=bundled,half                # half the training rows
.venv/bin/python experiments/train.py -m +experiment=smoke train.lr=1e-4,2e-4   # smoke-sized sweep
```

What you can vary (see `conf/config.yaml`):

| group | keys |
|---|---|
| `train` | `lr`, `epochs`, `max_steps`, `lora_r`, `lora_alpha` (null = 2×r), `lora_dropout`, `weight_decay`, `batch_size`, `grad_accum`, `warmup_steps`, `scheduler`, `max_seq_length`, `base_model`, `seed` |
| `eval` | `eval_every`, `bench_n`, `bench_splits`, `val_cap` |
| `data` | `data=bundled` or `data=half`; or `data.fraction=0.25`; `data.artifacts=/path` for a folder another notebook-1 run produced (different `TARGET_SCALE`, templates, holdouts) |
| `run` | `name`, `gpu` (T4, L4, A10G, A100), `project`, `dry_run` |

Named presets live in `conf/experiment/` (`smoke`, `reference`); add one per study. Memory note for big GPUs: the loss over a batch materialises `batch × seq × vocab`
logits in float32, about 20 GB for 32 × 1024 tokens with Qwen's 150k vocabulary, so on a 44 GB
L40S keep `train.batch_size` at 16 and use `train.grad_accum` for a larger effective batch.

Each job logs, every `eval.eval_every` steps: `eval/loss` on validation rows and
`bench/<split>/sql_exec`, `bench/<split>/places_f1` on `eval.bench_n` benchmark rows per
split. At the end it scores `2 × bench_n` rows of every split, including `paraphrase` and
`compositional`, saves the adapter to the volume under `runs/<name>/` and writes
`result.json` into its Hydra output dir (`experiments/outputs/...` or
`experiments/multirun/.../<job>/`).

Cost: a T4 is about $0.59/hour on Modal, so a default run is about $0.25; a 6-job sweep of
one-epoch runs about $1.50. `run.gpu=L4` is roughly twice as fast for a little more per hour.

## Look at the results

```bash
.venv/bin/python experiments/summarize.py experiments/multirun/<date>/<time>   # one table for a sweep
.venv/bin/python experiments/summarize.py --volume                              # same table from the volume (survives a dropped connection)
scripts/trackio_pull.sh               # pull the trackio DB from the volume, open the local dashboard
scripts/trackio_pull.sh lr1e-4-ep2    # also fetch that run's adapter into work/runs/lr1e-4-ep2/
```

The dashboard is trackio's local Gradio app over SQLite files under `work/trackio`;
nothing goes to Hugging Face. Compare runs on `train/loss`, `eval/loss` and the `bench/*`
curves: over-fitting shows as `iid` still climbing while `unseen_region` and `noisy` flatten
or drop. Per-run numbers are also in `runs/<name>/eval.json` on the volume.

## Studies

1. Over-fitting: `train.lr`, `train.epochs`, `train.lora_r`, `train.lora_dropout`,
   `train.weight_decay`. Goal: a default that holds `unseen_region` and `noisy`.
2. Data scale: `data=half`, `data.fraction=0.25`, or an artifacts folder from notebook 1 run
   with a different `TARGET_SCALE`. Is over-fitting about volume or about diversity?

Record what moved here when a study is done, then change the `TrainConfig` default (and
`conf/config.yaml`) and rebuild the notebooks.

### 2026-10-20: reference vs short vs regularized (L40S, 16 x 2, final score on 60 rows per split)

| run | steps | min | iid | unseen_region | unseen_anchor | noisy | paraphrase | compositional |
|---|---|---|---|---|---|---|---|---|
| reference (lr 2e-4, r 16, 1 epoch) | 177 | 12.4 | 1.00 | 1.00 | 1.00 | 0.75 | 0.97 | 0.15 |
| short (same, stopped at 90) | 90 | 8.1 | 1.00 | 0.98 | 0.97 | 0.75 | 0.97 | 0.25 |
| regularized (lr 1e-4, r 8, dropout 0.05, wd 0.1, 2 epochs) | 354 | 23.2 | 1.00 | 1.00 | 1.00 | 0.72 | 0.92 | 0.22 |

SQL execution match; places F1 is 1.00 everywhere except noisy (0.21 for all three: the gold there
is the canonical name, e.g. *Thiruvananthapuram* for *Trivandrum*, which no recipe can learn from
SFT data that never contains aliases). gpt-oss:20b zero-shot on the same splits: 0.53 / 0.55 / 0.55 /
0.25 / 0.57 / 0.50.

What it says: the recipe barely matters for the splits SFT covers; every run saturates iid and the
held-out states within ~50 steps and the differences after that are within 1-3 rows of 60.
Regularisation buys nothing (and costs paraphrase). The two splits with headroom are *noisy*
(0.75: the remaining misses are aliases that fuzzy search cannot bridge, a retrieval and
name-normalisation problem) and *compositional* (0.15-0.25 vs gpt-oss 0.50: templates the model never
saw). Those are the targets for the post-training roadmap, not for more SFT sweeps. For the
workshop, 90 steps (8 min on an L40S, ~15 min on a T4) is as good as a full epoch. Only the
`reference` run and preset were kept; the other two were deleted from the volume.

## Compare runs in the playground app

Every run leaves `runs/<name>/` on the volume: `adapter/` (final LoRA), `trainer/checkpoint-N/`
(adapter checkpoints when `eval.save_every` is set), `merged/` (16-bit merged weights when
`eval.save_merged` is set), `eval.json`, `config.json`, `log_history.json`. To put a run next to the
base model and gpt-oss in the app:

```bash
.venv/bin/modal volume get --force gazet-tutorial runs/reference/merged work/runs/reference/   # final model
# or any checkpoint, merged on the laptop (CPU, a minute):
.venv/bin/modal volume get --force gazet-tutorial runs/reference/trainer/checkpoint-50 work/runs/reference/trainer/
uv run --with "transformers>=5" --with torch --with peft --with sentencepiece \
    python experiments/merge_adapter.py work/runs/reference/trainer/checkpoint-50 work/merged/reference-step50
# convert + quantize exactly as docs/take-it-home.md steps 4-5 (converter at the tag matching `llama version`):
uv run --with "transformers>=5" --with torch --with gguf --with sentencepiece --with protobuf --with "numpy<3" \
    python ~/llama.cpp/convert_hf_to_gguf.py work/runs/reference/merged --outtype bf16 --outfile work/gguf/reference-bf16.gguf
llama quantize work/gguf/reference-bf16.gguf work/gguf/reference-q8_0.gguf Q8_0
# serve any number of them; each becomes a sidebar row named after the file
scripts/serve_models.sh start work/gguf/reference-q8_0.gguf work/gguf/reference-step50-q8_0.gguf work/gguf/short-q8_0.gguf
.venv/bin/streamlit run app/playground.py
```

Serving step 50 next to the end of the same run is the most direct way to show a room what
over-fitting does and does not change.

## If Colab is down during the workshop

```bash
.venv/bin/python experiments/train.py run.name=workshop
scripts/trackio_pull.sh workshop
```

Then either open notebook 2 on any GPU machine with `ADAPTER_DIR=work/runs/workshop/adapter`
(training is skipped, evaluation and the maps run), or merge + convert the adapter to GGUF
per `docs/take-it-home.md` and demo from the playground app with no GPU at all.
