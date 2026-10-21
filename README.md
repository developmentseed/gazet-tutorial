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
| 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/02_train_eval_inference.ipynb) `02_train_eval_inference` | T4 GPU | ~45 min | LoRA fine-tune Qwen3.5-0.8B with Unsloth, run the geocoder end to end on a map, export a q8_0 GGUF to take home (the benchmark is scored offline against that GGUF) |
| 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/03_dpo.ipynb) `03_dpo` (optional) | T4 GPU | ~25 min | Demo: DPO on top of the SFT adapter with pairs mined from the model's own answers and verified by execution; before/after on the noisy and compositional subset; what preference tuning can and cannot fix |

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

Every model is scored on the same fixed, template-balanced subset of every split (40 SQL
+ 20 extraction questions each). Scoring happens offline, not in the notebook:
`scripts/benchmark_gguf.py` takes the GGUF notebook 2 exports, serves it with
`llama serve`, asks it every subset question, executes the SQL it writes and prints one
table per task next to every other model it finds, the committed **zero-shot gpt-oss:20b
baseline** and the **untouched Qwen3.5-0.8B** among them:

```bash
python scripts/benchmark_gguf.py gazet-in-q8_0.gguf              # serve, predict, score, compare (~5 min)
python scripts/benchmark_gguf.py gazet-in-q8_0.gguf --limit 3    # smoke test
python scripts/benchmark_gguf.py unsloth/Qwen3.5-0.8B-GGUF:Q8_0  # re-run the base model yourself
python scripts/benchmark_gguf.py --compare-only                  # print the tables again
```

Predictions land in `work/benchmark/<name>/` and the run is resumable. The committed
baselines under `artifacts/baselines/` were produced once with the same prompts and
candidate tables; gpt-oss:20b through Ollama cloud, the base model through the same
script:

```bash
ollama signin                                   # gpt-oss:20b-cloud via local Ollama
python scripts/run_baseline.py --limit 3        # smoke test
python scripts/run_baseline.py                  # the full subset, resumable
python scripts/paraphrase.py                    # builds the paraphrase split (spot-check it)
python scripts/run_baseline.py --splits paraphrase
```

Any OpenAI-compatible endpoint works (`--base-url`, `--model`).

On the committed subset (40 SQL / 20 extraction questions per split), the three models
`scripts/benchmark_gguf.py` compares. The fine-tuned column is the `reference` run from
`experiments/` (notebook 2's defaults, one epoch) exported to q8_0; gpt-oss:20b ran with
`reasoning_effort=low` (mean 1.7 s and 150 completion tokens per call), the base model
with thinking off, like the fine-tuned one.

**SQL task, execution match**: the predicted and the reference SQL return the same set of
feature ids (same order for the top-N families).

| split | gpt-oss:20b zero-shot | Qwen3.5-0.8B base | Qwen3.5-0.8B fine-tuned |
|---|---|---|---|
| iid | 0.53 | 0.00 | 1.00 |
| unseen_region | 0.55 | 0.00 | 1.00 |
| unseen_anchor | 0.55 | 0.00 | 1.00 |
| noisy | 0.25 | 0.00 | 0.85 |
| compositional | 0.50 | 0.00 | 0.18 |
| paraphrase | 0.57 | 0.03 | 0.98 |
| mean | 0.49 | 0.00 | 0.83 |

**Place extraction task, F1**: set precision / recall over the place names in the JSON
the model returns, against the gold places.

| split | gpt-oss:20b zero-shot | Qwen3.5-0.8B base | Qwen3.5-0.8B fine-tuned |
|---|---|---|---|
| iid | 0.92 | 0.77 | 1.00 |
| unseen_region | 0.91 | 0.73 | 1.00 |
| unseen_anchor | 0.82 | 0.86 | 1.00 |
| noisy | 0.55 | 0.41 | 0.45 |
| compositional | 1.00 | 0.90 | 1.00 |
| paraphrase | 0.90 | 0.88 | 1.00 |
| mean | 0.85 | 0.76 | 0.91 |

The base model never writes SQL that returns the right rows (it does not know the
schema or the id convention) but already extracts places reasonably well; fine-tuning
fixes the first and sharpens the second. The two rows to argue about are `noisy`,
the one place the fine-tuned extractor loses to gpt-oss (it copies abbreviations and
old names instead of normalising them), and `compositional`, where gpt-oss's general
SQL ability beats a model that has only seen eight templates.

Scoring is forgiving about plumbing and strict about meaning: the bare table name
`divisions_area`, markdown fences and a missing `id` column (geometry is compared
instead) are all accepted. What gpt-oss still misses are the house rules a fine-tuned
model learns from the data: buffer questions score 0 on every split (it reaches for
`ST_DistanceSphere`, which DuckDB does not have), adjacency often filters to one subtype,
and old names are returned as written ("Orissa") rather than normalised. Your own
model's rows come from `scripts/benchmark_gguf.py`; the playground replays the same
questions interactively.

## Workshop agenda (3 hours)

| time | block |
|---|---|
| 0:00–1:00 | Introduction and theory: what gazet does and why a small language model; how fine-tuning works (LoRA, chat templates, loss masking, execution-based evaluation); why synthetic data; why India and what we simplify |
| 1:00–1:30 | Notebook 1 live: synthetic data for training, from Overture boundaries to verified examples, holdouts and the benchmark splits |
| 1:30–2:00 | Break (start notebook 2's install cell before you leave) |
| 2:00–2:30 | Notebook 2: LoRA, the masking check, training; the geocoder end to end on a map while the GGUF exports |
| 2:30–3:00 | Model export and the app: download the GGUF, `llama serve` it, try it in the local playground next to the base model and gpt-oss:20b |
| 3:00–3:30 | Explore on your own: score your GGUF on the benchmark (`scripts/benchmark_gguf.py`), ask the app questions, try the DPO demo (notebook 3), start the take-home assignments |

Take-home: train 2 epochs, add a template, add your state's abbreviations and old names
to `gazet_tutorial/india.py` and re-run the noisy split, swap the country.

## Take it home

**No time to train?** The workshop's reference run is published on Hugging Face, as a
[LoRA adapter](https://huggingface.co/developmentseed/gazet-tutorial-qwen3.5-0.8b-lora)
and as a [q8_0 GGUF](https://huggingface.co/developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF).
With llama.cpp installed, one command serves the geocoder model, and the benchmark script
takes the same spec:

```bash
llama serve -hf developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF:Q8_0 --port 8081 --jinja -c 4096 -ngl 99
python scripts/benchmark_gguf.py developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF:Q8_0
scripts/serve_models.sh start        # base, the published model and gpt-oss:20b for the playground app
```

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
artifacts/            bundled output of notebook 1 (dataset, benchmark, relations), committed baseline predictions, DPO pairs (dpo/)
app/                  Streamlit playground (compare models locally via llama serve)
experiments/          Modal training runs + sweeps with local trackio logging (where notebook 2's defaults come from)
docs/                 take-it-home.md (model -> GGUF -> llama serve, by hand), post-training roadmap (DPO, GRPO)
tests/                pytest suite (synthetic square polygons, no downloads)
scripts/              build_notebooks.sh, run_nb1.sh (regenerates artifacts/), benchmark_gguf.py (score a GGUF vs gpt-oss and base), mine_pairs.py (DPO pairs for notebook 3), run_baseline.py, paraphrase.py, serve_models.sh
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
scripts/build_notebooks.sh        # regenerate the .ipynb files after editing notebooks/src
```

Notebooks 2 and 3 need a CUDA GPU (Unsloth does not run on Apple Silicon); run them in Colab.

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
