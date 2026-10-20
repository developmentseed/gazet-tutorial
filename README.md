# gazet-tutorial

Fine-tune a small language model to do **natural-language geocoding**, in two Colab
notebooks and a three-hour workshop. This is a teaching-sized reproduction of
[gazet](https://github.com/developmentseed/gazet): one country (India), one table
(Overture administrative divisions: country, states and union territories, districts),
eight query templates, a 0.8B model with a LoRA adapter trained on a free T4, and an
honest benchmark that includes gpt-oss:20b, the model gazet runs in production.

```
"which districts border Nagpur?"
   -> model extracts ["Nagpur"]
   -> fuzzy search returns 10 candidate rows (Nagpur, Nagaur, Nagapattinam, ...)
   -> model writes DuckDB spatial SQL using the right id
   -> SQL runs, polygons come back as GeoJSON
```

| | Notebook | Runtime | Time | What you do |
|---|---|---|---|---|
| 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/01_data_and_dataset.ipynb) `01_data_and_dataset` | CPU | ~30 min | Pull India's boundaries from Overture and clean them, compute spatial relations, write SQL templates, hold out places, generate ~3,600 verified training examples for two tasks, build the benchmark splits |
| 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/02_train_eval_inference.ipynb) `02_train_eval_inference` | T4 GPU | ~70 min | LoRA fine-tune Qwen3.5-0.8B with Unsloth, score it by executing the SQL it writes on every benchmark split, compare with the base model and with gpt-oss:20b, run the geocoder on a map |

**No accounts needed.** No token, no Drive mount. `pip install` of this repo carries the
`artifacts/` folder inside the package, so the dataset, the benchmark and the baseline
predictions produced by notebook 1 are already on the Colab machine and notebook 2 runs on
its own. The only download is India's boundaries from Overture's public S3 bucket.

## What the model learns

Training data is entirely synthetic and never touches an LLM. Each SQL template comes
with a list of question phrasings; a sampled place fills both, the SQL is executed
against the real table, and only queries that return rows are kept.

| family | template | example question |
|---|---|---|
| direct_lookup | `lookup_01` | *where is Nagpur?* |
| disambiguation | `disambiguate_01` | *Bilaspur, Himachal Pradesh* (there are two Bilaspurs) |
| adjacency | `adj_01`, `adj_02` | *neighbours of Madhya Pradesh* (same level) / *which districts border Nagpur?* |
| containment | `contain_01`, `contain_02` | *districts of Karnataka* / *which state is Bhopal in?* |
| buffer | `buffer_01` | *districts within 50 km of Pune* |
| aggregation | `agg_01` | *3 largest districts in Rajasthan* |

One model is trained on two tasks at once, distinguished by the system prompt:
extract place names as JSON, and write SQL given a candidate table plus the question.
The evaluation metric is **execution accuracy**: predicted and reference SQL are both
run, and the sets of returned feature ids must match.

## The benchmark

A small model memorises place names, so the in-distribution test split flatters it.
Notebook 1 holds out **Kerala, Odisha and Goa** entirely (state and every district) plus
30 more districts, and builds five extra test splits:

| split | what it isolates |
|---|---|
| `iid` | regression: same anchors and templates as training |
| `unseen_region` | whole states the model never anchored a question on |
| `unseen_anchor` | individual districts never used, including one of each same-name pair |
| `noisy` | abbreviations (*UP*), old names (*Bangalore*), typos, lowercase; candidates re-derived by fuzzy search on the noisy string, retrieval ceiling reported |
| `compositional` | three templates combining two operators, never used in training |
| `paraphrase` | LLM rewrites of test questions (built offline, see below) |

Notebook 2 scores the base model, the fine-tuned model and the **zero-shot gpt-oss:20b
baseline** on a fixed template-balanced subset of every split (40 SQL + 20 extraction
questions each). The baseline predictions are committed under `artifacts/baselines/`
and were produced once, offline, with the same prompts and candidate tables:

```bash
ollama signin                                   # gpt-oss:20b-cloud via local Ollama
python scripts/run_baseline.py --limit 3        # smoke test
python scripts/run_baseline.py                  # the full subset, resumable
python scripts/paraphrase.py                    # builds the paraphrase split (spot-check it)
python scripts/run_baseline.py --splits paraphrase
```

Any OpenAI-compatible endpoint works (`--base-url`, `--model`).

Zero-shot gpt-oss:20b on the committed subset (40 SQL / 20 extraction questions per
split, `reasoning_effort=low`, mean 1.7 s and 150 completion tokens per call):

| split | SQL valid | SQL execution match | extraction F1 |
|---|---|---|---|
| iid | 0.97 | 0.53 | 0.92 |
| unseen_region | 0.97 | 0.55 | 0.91 |
| unseen_anchor | 0.95 | 0.55 | 0.82 |
| noisy | 1.00 | 0.25 | 0.55 |
| compositional | 0.95 | 0.50 | 1.00 |
| paraphrase | 0.95 | 0.57 | 0.90 |

Scoring is forgiving about plumbing and strict about meaning: the bare table name
`divisions_area`, markdown fences and a missing `id` column (geometry is compared
instead) are all accepted. What gpt-oss still misses are the house rules a fine-tuned
model learns from the data: buffer questions score 0 on every split (it reaches for
`ST_DistanceSphere`, which DuckDB does not have), adjacency often filters to one subtype,
and old names are returned as written ("Orissa") rather than normalised. The fine-tuned
0.8B model's rows are produced in notebook 2 and in the local playground.

## Workshop agenda (3 hours)

| time | block |
|---|---|
| 0:00–0:10 | What gazet does; why synthetic data; why India; what we simplify |
| 0:10–0:50 | Notebook 1 live: data cleaning, relations, templates, holdouts, generation, benchmark |
| 0:50–1:35 | Notebook 2: install, LoRA, masking check, start training; discuss the benchmark and the gpt-oss baseline while it runs |
| 1:35–2:05 | Results: base vs fine-tuned vs gpt-oss:20b per split, error gallery, retrieval ceiling |
| 2:05–2:30 | Take it home: GGUF + `llama serve`, the local playground app (base vs fine-tuned vs gpt-oss:20b side by side) |
| 2:30–3:00 | What comes next: preference tuning and RL on the same benchmark ([roadmap](docs/post-training-roadmap.md)); take-home assignments |

Take-home: train 2 epochs, add a template, add your state's abbreviations and old names
to `gazet_tutorial/india.py` and re-run the noisy split, swap the country.

## Take it home

After training, the tutorial continues on your laptop: merge and download the model from
Colab, convert it to GGUF and quantize it with llama.cpp's tools, serve it with
`llama serve` next to the base model and gpt-oss:20b (also through llama.cpp, installed
from [llama.app](https://llama.app/docs/installation), no Ollama required), and compare
the three in a Streamlit app on free questions and on benchmark questions, with maps.
Every step is a command you type and understand:
[`docs/take-it-home.md`](docs/take-it-home.md). The app itself is in [`app/`](app/README.md).

## Repository layout

```
notebooks/            the two .ipynb files Colab opens (generated from notebooks/src/*.py)
notebooks/src/        percent-format sources; edit these, then scripts/build_notebooks.sh
src/gazet_tutorial/   helper package: DuckDB/Overture access, fuzzy search, SQL utils, eval, benchmark, baseline client, maps
artifacts/            bundled output of notebook 1 (dataset, benchmark, relations) and committed baseline predictions
app/                  Streamlit playground (compare models locally via llama serve)
experiments/          Modal training runs + sweeps with local trackio logging (where notebook 2's defaults come from)
docs/                 take-it-home.md (model -> GGUF -> llama serve, by hand), post-training roadmap (DPO, GRPO)
tests/                pytest suite (synthetic square polygons, no downloads)
scripts/              build_notebooks.sh, run_nb1.sh (regenerates artifacts/), run_baseline.py, paraphrase.py, serve_models.sh
```

The learner-facing logic (templates, generation loop, prompts, training config,
metrics, the `geocode()` function) lives in notebook cells. The package holds
boilerplate so that `pip install` is the entire setup.

## Local development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
scripts/run_nb1.sh                # executes notebook 1 (downloads ~60 MB from Overture S3)
scripts/run_nb1.sh "/path/to/overture/divisions_area/*.parquet"   # or from a local copy
scripts/build_notebooks.sh        # regenerate both .ipynb files after editing notebooks/src
```

Notebook 2 needs a CUDA GPU (Unsloth does not run on Apple Silicon); run it in Colab.

Hyperparameter work happens outside the notebook: `experiments/modal_train.py` runs the same
training on a Modal GPU from a laptop, logs every run to a local trackio database and scores
the benchmark splits during training (the overfitting curve). It is also the fallback if
Colab is unavailable. See [`experiments/README.md`](experiments/README.md).

## Bundled artifacts

`pyproject.toml` includes `artifacts/` in the wheel as `gazet_tutorial/artifacts`, which is
why a plain `pip install git+https://github.com/developmentseed/gazet-tutorial` puts the
dataset, benchmark, relation tables and baseline predictions (about 28 MB) next to the code;
`gazet_tutorial.artifacts_dir()` finds that copy, or a repo checkout's `artifacts/` when you
run locally. They are produced by `scripts/run_nb1.sh` with fixed seeds; regenerate them
whenever templates, prompts or holdouts change, then re-run the baseline scripts. The
trained adapter is not distributed; notebook 2 trains it in about 20 minutes, and
`ADAPTER_DIR` lets you load one you saved earlier.

## Data and licenses

Code is MIT licensed. Boundary data comes from the
[Overture Maps Foundation](https://overturemaps.org) divisions theme, which is
derived from OpenStreetMap and distributed under the
[ODbL](https://opendatacommons.org/licenses/odbl/); the bundled dataset contains
feature ids and names from that data and inherits its terms. Model weights:
[Qwen3.5](https://huggingface.co/Qwen) (Apache 2.0), via Unsloth; the baseline uses
[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (Apache 2.0).
