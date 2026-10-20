# %% [markdown]
# # Notebook 2 · Fine-tune, evaluate, geocode
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/02_train_eval_inference.ipynb)
#
# Part 2 of the tutorial. In notebook 1 we built ~3,600 synthetic examples of two
# tasks over India's administrative boundaries: *extract place names from a question*
# and *write spatial SQL over a candidate table*. Here we teach both to one small
# model and measure what it actually learned.
#
# **Runtime → Change runtime type → T4 GPU** before you start. Free-tier T4 is enough.
#
# **In this notebook (~70 minutes):**
#
# 1. Load Qwen3.5-0.8B with Unsloth and attach LoRA adapters.
# 2. Render the chat examples and mask everything except the assistant's answer.
# 3. Train for one epoch (~20 minutes after a few minutes of kernel compilation).
# 4. Evaluate with **execution accuracy** on every benchmark split: in-distribution,
#    unseen states, unseen districts, noisy names, compositional queries. Compare the
#    base model, the fine-tuned model and **gpt-oss:20b**, the 20B model gazet runs
#    in production, on the same questions.
# 5. Run the full geocoder on real questions and look at the results on a map.
# 6. (Optional) Export to GGUF for CPU inference, the way gazet is deployed.
#
# No accounts are needed. The dataset, the benchmark and the gpt-oss predictions ship
# inside the package you `pip install`, so they are already on this machine.

# %% [markdown]
# ## 0. Install
#
# This is Unsloth's recommended Colab install for Qwen3.5 (pinned torch 2.8,
# transformers 5.2). It takes 3–5 minutes. If pip prints dependency warnings at
# the end, ignore them; if it asks you to restart the runtime, do so and re-run.

# %%
import importlib.util
import os
import subprocess
import sys

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or importlib.util.find_spec("google.colab") is not None

def sh(cmd: str):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if importlib.util.find_spec("unsloth") is None:
    sh(f"{sys.executable} -m pip install --upgrade -qqq uv")
    sh('uv pip install -qqq --system "torch==2.8.0" "triton>=3.3.0" torchvision "torchaudio==2.8.0" '
       'bitsandbytes xformers==0.0.32.post2 '
       '"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" '
       '"unsloth[base] @ git+https://github.com/unslothai/unsloth"')
    sh('uv pip install -qqq --system --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2')
    sh('uv pip install -qqq --system transformers==5.2.0')
    sh('unsloth install-kernels || true')

if importlib.util.find_spec("gazet_tutorial") is None:
    sh(f"{sys.executable} -m pip install -q git+https://github.com/developmentseed/gazet-tutorial")

# %% tags=["parameters"]
BASE_MODEL = "unsloth/Qwen3.5-0.8B"
DATASET_DIR = None        # None -> the dataset bundled with the package (output of notebook 1)
BENCH_DIR = None          # None -> the bundled benchmark splits
BASELINE_DIR = None       # None -> bundled gpt-oss:20b zero-shot predictions (artifacts/baselines/gpt-oss-20b-cloud/zero)
GEODATA_SOURCE = None     # None -> download India divisions from Overture S3 (~60 MB, 2-3 min)
MAX_SEQ_LENGTH = 1024     # notebook 1 measured P99 ≈ 820 tokens for SQL examples
EPOCHS = 1                # 2 epochs noticeably better; left as homework to fit the session
MAX_STEPS = None          # set to 20 for a smoke test
N_EVAL = 60               # rows per benchmark split and task for the fine-tuned model (subset rows always first)
ADAPTER_DIR = None        # path to an adapter you saved earlier (e.g. uploaded zip, unpacked) -> skip training and load it
SAVE_MERGED = False       # also save a merged 16-bit copy (~1.6 GB) to take home and convert to GGUF locally
RUN_GGUF = False          # export a q8_0 GGUF here in Colab instead (builds llama.cpp, adds ~10 min)
SEED = 108
TRACK = False             # log loss + eval curves with trackio: stays on this VM, dashboard embeds below the training cell

# %%
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch

from gazet_tutorial import artifacts_dir, data_dir
from gazet_tutorial.train import TrainConfig
from gazet_tutorial.geodata import connect, ensure_geodata
from gazet_tutorial.data import load_jsonl, label_mask_preview
from gazet_tutorial.search import fuzzy_search, fetch_by_id, candidates_csv, merge_candidates, per_place_limit
from gazet_tutorial.sqlutil import execute, strip_fences, format_sql
from gazet_tutorial.eval import (
    generate_batch, prompt_messages, evaluate_sql, evaluate_places, compare, failures, parse_places,
)
from gazet_tutorial.benchmark import SPLITS, load_benchmark, load_subset_ids
from gazet_tutorial.baseline import load_predictions, align
from gazet_tutorial.viz import plot_loss, plot_family_bars, map_features

pd.set_option("display.max_colwidth", 100)
torch.manual_seed(SEED)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (training will not work)")

ART = artifacts_dir()     # the artifacts/ folder bundled inside the installed package
DATASET_DIR = Path(DATASET_DIR) if DATASET_DIR else ART / "dataset"
BENCH_DIR = Path(BENCH_DIR) if BENCH_DIR else ART / "benchmark"
BASELINE_DIR = Path(BASELINE_DIR) if BASELINE_DIR else ART / "baselines" / "gpt-oss-20b-cloud" / "zero"
SKIP_TRAINING = ADAPTER_DIR is not None
OUT = data_dir() / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
prompts = json.loads((DATASET_DIR / "prompts.json").read_text())
SQL_SYSTEM, PLACE_SYSTEM, USER_TEMPLATE = prompts["sql_system"], prompts["place_system"], prompts["user_template"]
print("dataset:", DATASET_DIR, "\nbenchmark:", BENCH_DIR, "\nbaseline:", BASELINE_DIR)

# %% [markdown]
# The geodata is needed again for evaluation (we execute the generated SQL) and for
# the live demo. Each Colab notebook is a fresh machine, so we fetch it again: ~60 MB
# from Overture's S3 bucket, 2–3 minutes.

# %%
con = connect(spatial=True)
geodata_path = ensure_geodata(source=GEODATA_SOURCE, con=connect(spatial=True, httpfs=GEODATA_SOURCE is None))

# %% [markdown]
# ## 1. Load the base model and add LoRA adapters
#
# **LoRA** (Low-Rank Adaptation) freezes the 0.8B pretrained weights and learns small
# low-rank matrices on top of the attention and MLP projections. With rank 16 that
# is about 1–2% of the parameters, which is why this fits on a T4 and trains in minutes.
#
# Unsloth wraps Hugging Face models with faster kernels and memory tricks. For
# Qwen3.5 it returns a *processor* (the model family is multimodal); we only need
# its text tokenizer.

# %%
from unsloth import FastLanguageModel, is_bfloat16_supported

# Every hyperparameter below comes from one dataclass shared with experiments/modal_train.py,
# so what a sweep teaches us lands here by changing a default. Print it to see them all.
CFG = TrainConfig(max_seq_length=MAX_SEQ_LENGTH, epochs=EPOCHS, max_steps=MAX_STEPS, seed=SEED)
print(CFG)

model, processor = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,                 # 4-bit QLoRA is not recommended for Qwen3.5
    use_gradient_checkpointing="unsloth",
)
tokenizer = getattr(processor, "tokenizer", processor)

if SKIP_TRAINING:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=False)
    print("loaded trained adapter from", ADAPTER_DIR)
else:
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.lora_r,                       # 16
        lora_alpha=CFG.alpha,               # 2 × r, Unsloth's recommendation for Qwen
        lora_dropout=CFG.lora_dropout,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        bias="none",
        random_state=CFG.seed,
        use_gradient_checkpointing=False,   # already set above
    )

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters: {total/1e6:,.0f}M total, {trainable/1e6:,.1f}M trainable ({100*trainable/total:.2f}%)")

# %% [markdown]
# ## 2. Render the training text
#
# Each example is a list of chat messages. The tokenizer's **chat template** turns
# it into one string with special tokens marking the turns. Both tasks are mixed
# into one training set: the model learns to read the system prompt and behave
# accordingly.

# %%
from datasets import Dataset

def render(rows):
    return [{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False),
             "task_family": r["metadata"]["task_family"]} for r in rows]

train_rows = load_jsonl(DATASET_DIR / "sql/train.jsonl") + load_jsonl(DATASET_DIR / "places/train.jsonl")
val_rows = load_jsonl(DATASET_DIR / "sql/val.jsonl") + load_jsonl(DATASET_DIR / "places/val.jsonl")
train_ds = Dataset.from_list(render(train_rows)).shuffle(seed=SEED)
val_ds = Dataset.from_list(render(val_rows)).shuffle(seed=SEED).select(range(min(128, len(val_rows))))
print(f"train {len(train_ds)}  val {len(val_ds)}")
print(train_ds[0]["text"][:1500] + " …")

# %% [markdown]
# Notice the end of the string: `<|im_start|>assistant\n<think>\n\n</think>\n\n` followed
# by the answer. That empty `<think>` block is how Qwen3.5 writes a *non-thinking*
# reply. We will use the same marker to tell the trainer where the answer starts.

# %% [markdown]
# ## 3. Trainer: loss only on the answer
#
# With plain language-model training the loss would also reward predicting the
# system prompt and the candidate CSV, which we never want the model to generate.
# `train_on_responses_only` masks every token before the assistant marker so the
# loss covers only the SQL / JSON.

# %%
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

if TRACK and importlib.util.find_spec("trackio") is None:
    sh(f"{sys.executable} -m pip install -q trackio")

if not SKIP_TRAINING:
    config = SFTConfig(
        output_dir=str(OUT / "trainer"),
        dataset_text_field="text",
        max_length=CFG.max_seq_length,
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.grad_accum,   # effective batch 16
        num_train_epochs=CFG.epochs,
        max_steps=CFG.max_steps or -1,
        learning_rate=CFG.lr,
        lr_scheduler_type=CFG.scheduler,
        warmup_steps=CFG.warmup_steps,
        weight_decay=CFG.weight_decay,
        optim="adamw_8bit",
        max_grad_norm=1.0,
        fp16=not is_bfloat16_supported(),       # T4 has no bf16
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        eval_strategy="steps" if TRACK else "no",   # val loss along the way only when tracking; execution-based eval comes after
        eval_steps=CFG.eval_every,
        save_strategy="no",
        report_to="trackio" if TRACK else "none",
        project="gazet-tutorial",               # trackio project name (local SQLite on this VM, no Hub)
        trackio_space_id=None,                  # the default would try to create a Hugging Face Space
        run_name=CFG.run_name(),
        seed=CFG.seed,
        dataset_num_proc=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

# %% [markdown]
# **Sanity check the mask.** Decode only the tokens that carry loss. You should see
# just an SQL query or a JSON object, nothing from the prompt.

# %%
if not SKIP_TRAINING:
    ex = trainer.train_dataset[0]
    print(label_mask_preview(tokenizer, ex["input_ids"], ex["labels"]))

# %% [markdown]
# ## 4. Train
#
# About the first minute or two: Qwen3.5 uses custom Triton kernels that compile on
# first use, so the progress bar sits at step 0 for a while. After that, expect
# roughly 3–5 s/step on a T4 (it trains Qwen3.5 in float32 because fp16 is unstable
# for this architecture); one epoch is ~360 steps.

# %%
if not SKIP_TRAINING:
    t0 = time.time()
    stats = trainer.train()
    print(f"\ntrained in {(time.time() - t0)/60:.1f} min; final loss {stats.training_loss:.3f}; "
          f"peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    plot_loss(trainer.state.log_history)

    adapter_dir = OUT / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("adapter saved to", adapter_dir)

FastLanguageModel.for_inference(model)

# %% [markdown]
# The adapter is ~60 MB. It is saved on this VM only; if you want to keep it, zip and
# download it (last cell). Next time, upload and unpack it and set `ADAPTER_DIR` to
# skip training.

# %% [markdown]
# ## 5. Evaluate by executing the SQL, on every benchmark split
#
# String comparison is the wrong metric for text-to-SQL: `WHERE a AND b` and
# `WHERE b AND a` are the same query. We run both the predicted and the reference
# SQL against the real table and compare **the set of feature ids they return**
# (ordered for the top-N families). We keep exact match too, to show how much it
# under-reports. For place extraction we parse the JSON and compute set P / R / F1.
#
# Three models are scored on the **same fixed subset** of each split (40 SQL + 20
# extraction questions, template-balanced):
#
# - **base**: Qwen3.5-0.8B untouched (PEFT lets us switch the adapter off, no second model);
# - **finetuned**: the adapter we just trained (also scored on more rows, `N_EVAL`);
# - **gpt-oss:20b, zero-shot**: predictions produced offline by `scripts/run_baseline.py`
#   with the same system prompt, user template and candidate table; this is the model
#   gazet uses when no fine-tuned model is available.
#
# The splits (built in notebook 1): `iid`, `unseen_region` (Kerala, Odisha, Goa),
# `unseen_anchor` (30 held-out districts), `noisy` (abbreviations, old names, typos,
# lowercase), `compositional` (three unseen templates), and `paraphrase` if it was built.

# %%
bench = load_benchmark(BENCH_DIR)
subset = load_subset_ids(BENCH_DIR)
splits = [s for s in SPLITS if s in bench]
print("splits:", {s: len(bench[s]["sql"]) for s in splits})

def ordered_rows(split: str, task: str, n: int | None) -> list:
    """Subset rows first (so every model sees them), then the rest, cut to n."""
    rows = bench[split][task]
    ids = subset.get(split, {}).get(task, [])
    by_id = {r["metadata"]["sample_id"]: r for r in rows}
    first = [by_id[i] for i in ids if i in by_id]
    rest = [r for r in rows if r["metadata"]["sample_id"] not in set(ids)]
    out = first + rest
    return out[:n] if n else out

EVAL = {s: {t: ordered_rows(s, t, N_EVAL) for t in ("sql", "places")} for s in splits}
SUBSET = {s: {t: ordered_rows(s, t, len(subset.get(s, {}).get(t, [])) or 40) for t in ("sql", "places")} for s in splits}

def run_eval(label: str, sets: dict) -> dict:
    t0, out = time.time(), {}
    for split, tasks in sets.items():
        sql_out = generate_batch(model, tokenizer, [prompt_messages(s) for s in tasks["sql"]], max_new_tokens=256, batch_size=8, progress=False)
        places_out = generate_batch(model, tokenizer, [prompt_messages(s) for s in tasks["places"]], max_new_tokens=64, batch_size=16, progress=False)
        out[split] = {"sql": evaluate_sql(tasks["sql"], sql_out, con, geodata_path), "places": evaluate_places(tasks["places"], places_out)}
        print(f"  [{label}] {split:14s} sql exec_match {out[split]['sql'].exec_match.mean():.2f}   places f1 {out[split]['places'].f1.mean():.2f}")
    print(f"[{label}] done in {(time.time()-t0)/60:.1f} min")
    return out

results = {}
results["finetuned"] = run_eval("finetuned", EVAL)
with model.disable_adapter():
    results["base"] = run_eval("base", SUBSET)

# %% [markdown]
# ### The gpt-oss:20b rows
#
# Loaded from JSONL, scored with the same functions. If a file is missing (for
# example you rebuilt the benchmark), run `scripts/run_baseline.py` against an
# Ollama or any OpenAI-compatible endpoint and drop the output here.

# %%
gpt = {}
for split in splits:
    per_task = {}
    for task in ("sql", "places"):
        preds = load_predictions(BASELINE_DIR / f"{task}_{split}.jsonl")
        rows, p = align(bench[split][task], preds)
        if rows:
            per_task[task] = evaluate_sql(rows, p, con, geodata_path) if task == "sql" else evaluate_places(rows, p)
    if per_task:
        gpt[split] = per_task
if gpt:
    results["gpt-oss:20b zero-shot"] = gpt
    print("gpt-oss predictions found for:", {s: {t: len(d) for t, d in v.items()} for s, v in gpt.items()})
else:
    print("no gpt-oss predictions under", BASELINE_DIR)

# %% [markdown]
# ### The headline table
#
# Rows are models, columns are splits. All three models on the fixed subset, plus
# the fine-tuned model on its larger sample. Read it column by column: how far does
# each model fall from `iid` as the questions move away from the training data?

# %%
def matrix(results: dict, task: str, metric: str, restrict_to_subset: bool) -> pd.DataFrame:
    table = {}
    for model_name, per_split in results.items():
        label = model_name + (" (subset)" if restrict_to_subset else " (all rows)")
        for split, tasks in per_split.items():
            df = tasks.get(task)
            if df is None:
                continue
            if restrict_to_subset:
                ids = set(subset.get(split, {}).get(task, []))
                df = df[df["sample_id"].isin(ids)] if ids else df
            if len(df):
                table.setdefault(label, {})[split] = df[metric].mean()
    out = pd.DataFrame(table).T
    return out[[s for s in splits if s in out.columns]]

sql_table = pd.concat([matrix(results, "sql", "exec_match", True), matrix({"finetuned": results["finetuned"]}, "sql", "exec_match", False)])
places_table = pd.concat([matrix(results, "places", "f1", True), matrix({"finetuned": results["finetuned"]}, "places", "f1", False)])
print("SQL: execution accuracy")
display(sql_table.round(3))
print("\nPlace extraction: F1")
display(places_table.round(3))

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sql_table.T.plot.bar(ax=axes[0], rot=20, title="SQL execution accuracy by split"); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8)
places_table.T.plot.bar(ax=axes[1], rot=20, title="Place extraction F1 by split"); axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ### Per family, in distribution
# On `iid`, which query shapes are hard? Compare the three models per template family.

# %%
iid = {m: r["iid"]["sql"] for m, r in results.items()}
display(compare(iid, by="task_family")[["n", "exec_match", "exact_match"]].round(3))
fig, ax = plt.subplots(figsize=(10, 3.5))
plot_family_bars(iid, metric="exec_match", ax=ax); ax.set_title("iid: execution accuracy per family")

# %% [markdown]
# ### Noisy names: was it the model or the search?
#
# A noisy question can fail for two different reasons: the fuzzy search never
# retrieved the right row (nothing a SQL model can do), or the row was there and the
# model still picked wrong. Splitting by that tells you which component to fix.

# %%
if "noisy" in results["finetuned"]:
    meta = {r["metadata"]["sample_id"]: r["metadata"] for r in bench["noisy"]["sql"]}
    df = results["finetuned"]["noisy"]["sql"].copy()
    df["noise_type"] = df["sample_id"].map(lambda i: meta[i]["noise_type"])
    df["retrieved"] = df["sample_id"].map(lambda i: meta[i]["anchor_in_candidates"])
    out = df.groupby("noise_type").agg(n=("exec_match", "size"), retrieval_ceiling=("retrieved", "mean"), exec_match=("exec_match", "mean"))
    out["exec_match_when_retrieved"] = df[df["retrieved"]].groupby("noise_type")["exec_match"].mean()
    display(out.round(3))
    print("\nexamples the model got wrong although the right row was in the table:")
    for _, row in df[df["retrieved"] & ~df["exec_match"]].head(4).iterrows():
        print(f"  [{row.noise_type}] {row.question!r} -> {row.error}")

# %% [markdown]
# ### What the base model does with the same prompt
# Typical zero-shot failures: invented columns, a `read_parquet` path that does not
# exist, an explanation instead of SQL. The fine-tuned model has learned the schema
# and the house style. gpt-oss:20b usually writes valid SQL but in its own style;
# look at how often it also returns the right rows.

# %%
q0 = results["base"]["iid"]["sql"].loc[0]
print(f"Q: {q0.question}\n--- base ---\n{q0.pred[:600]}")
print("\n--- finetuned ---\n" + results["finetuned"]["iid"]["sql"].loc[0, "pred"])
if gpt:
    g = results["gpt-oss:20b zero-shot"]["iid"]["sql"]
    g0 = g[g["sample_id"] == q0.sample_id]
    if len(g0):
        print("\n--- gpt-oss:20b ---\n" + g0.iloc[0]["pred"][:600])

# %% [markdown]
# ### Error gallery
# Where does the fine-tuned model still fail? On `iid` it is usually a disambiguation
# miss; on `compositional` it is a template it has never seen, answered with the
# nearest one it knows.

# %%
for split in ("iid", "compositional"):
    if split not in results["finetuned"]:
        continue
    print(f"\n===== {split} =====")
    for _, row in failures(results["finetuned"][split]["sql"], "exec_match", n=4).iterrows():
        print(f"\n[{row.task_family}] {row.question}   -> {row.error}")
        print("  gold:", re.sub(r"\s+", " ", row.gold)[:180])
        print("  pred:", re.sub(r"\s+", " ", row.pred)[:180])

for model_name, per_split in results.items():
    for split, tasks in per_split.items():
        for task, df in tasks.items():
            df.to_json(OUT / f"eval_{model_name.replace(' ', '_').replace(':', '-')}_{split}_{task}.json", orient="records", force_ascii=False, indent=1)
sql_table.to_csv(OUT / "headline_sql.csv"); places_table.to_csv(OUT / "headline_places.csv")

# %% [markdown]
# ## 6. The geocoder, end to end
#
# This is the same pipeline gazet serves behind its API:
#
# 1. **extract** place names from the question (model, places prompt)
# 2. **search** each name with Jaro-Winkler similarity → candidate table
# 3. **write SQL** given the candidates and the question (model, SQL prompt)
# 4. **execute** in DuckDB and return GeoJSON
#
# The prompts are read from `prompts.json`, so inference matches training byte for byte.

# %%
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

def chat(messages, max_new_tokens):
    return generate_batch(model, tokenizer, [messages], max_new_tokens=max_new_tokens, progress=False)[0]

def extract_places(question: str) -> list:
    out = chat([{"role": "system", "content": PLACE_SYSTEM}, {"role": "user", "content": question}], 64)
    return parse_places(out) or [question]          # fall back to the whole query, like gazet

def geocode(question: str, verbose: bool = True) -> dict:
    places = extract_places(question)
    limit = per_place_limit(len(places))
    candidates = merge_candidates([fuzzy_search(con, geodata_path, p, limit=limit) for p in places])
    user = USER_TEMPLATE.format(candidates=candidates_csv(candidates), question=question)
    sql = strip_fences(chat([{"role": "system", "content": SQL_SYSTEM}, {"role": "user", "content": user}], 256))
    try:
        result = execute(con, sql, geodata_path)
        error = None
    except Exception as exc:
        result, error = pd.DataFrame(), str(exc).splitlines()[0]
    anchors = pd.concat([fetch_by_id(con, geodata_path, i, geometry=True) for i in dict.fromkeys(UUID.findall(sql))], ignore_index=True) \
        if UUID.findall(sql) else pd.DataFrame()
    if verbose:
        print(f"Q: {question}\n  places: {places}\n  candidates: {candidates['name'].tolist()}\n  sql:\n{format_sql(sql)}")
        print(f"  -> {len(result)} feature(s)" + (f"  ERROR: {error}" if error else "") +
              (": " + ", ".join(result['name'].astype(str).head(12)) if 'name' in result else ""))
    return {"question": question, "places": places, "candidates": candidates, "sql": sql, "result": result, "anchors": anchors, "error": error}

r = geocode("which states border Madhya Pradesh?")
map_features(r["result"], anchors=r["anchors"])

# %%
EXAMPLES = [
    "show me Nagpur",
    "Bilaspur, Himachal Pradesh",
    "districts of Karnataka",
    "districts within 50 km of Pune",
    "3 largest districts in Rajasthan",
    "which state is Bhopal in?",
]
results_demo = {q: geocode(q) for q in EXAMPLES}

# %%
# Pick any of the examples to map.
r = results_demo["districts within 50 km of Pune"]
map_features(r["result"], anchors=r["anchors"])

# %% [markdown]
# Try your own. The model only knows the eight query shapes it was trained on, and
# only India. Three questions worth watching:
#
# - a held-out state: *"districts of Kerala"* (the model never anchored a question on it);
# - an abbreviation: *"districts of UP"* (the extractor has to normalise it, or the
#   search will not find Uttar Pradesh);
# - a composition: *"districts of Karnataka that border Tamil Nadu"* (no template for it).
#
# Each of those is a row in the benchmark table above, and each is the kind of gap
# you close with more data, a new template, or the preference / RL stages described
# in `docs/post-training-roadmap.md`.

# %%
for q in ["districts of Kerala", "districts of UP", "districts of Karnataka that border Tamil Nadu"]:
    geocode(q)

# %%
r = geocode("neighbours of Lakshadweep")   # a trick question: islands touch nothing
# map_features(r["result"], anchors=r["anchors"])

# %% [markdown]
# ## 7. Take the model home
#
# gazet runs this model on CPU with llama.cpp's server from a quantized GGUF file (~0.9 GB
# at 8 bits). The repo ships a **local playground** (`app/README.md`): serve the base
# model, your fine-tuned model and gpt-oss:20b with `llama serve`, then compare them
# on free questions and on benchmark questions in a Streamlit app, with maps.
#
# The step-by-step version (merge → convert → quantize → serve → curl → app), with every
# command explained, is `docs/take-it-home.md` in the repo. It starts from a merged
# 16-bit copy of this model: set `SAVE_MERGED = True` and download the zip the next cell
# writes. (`RUN_GGUF = True` instead builds llama.cpp here and exports a q8_0 GGUF
# directly, ~10 min, if you would rather skip the local conversion.)

# %%
if SAVE_MERGED:
    import shutil
    merged_dir = OUT / "merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    zip_path = shutil.make_archive(str(merged_dir), "zip", merged_dir)
    print("merged model:", merged_dir, "->", zip_path)
    # from google.colab import files; files.download(zip_path)

if RUN_GGUF:
    gguf_dir = OUT / "gguf"
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q8_0")
    print(sorted(p.name for p in gguf_dir.glob("*.gguf")))
    # CPU inference with llama-cpp-python, using the chat template embedded in the GGUF:
    # !pip install -q llama-cpp-python
    # from llama_cpp import Llama
    # llm = Llama(model_path=str(next(gguf_dir.glob("*q8_0*.gguf"))), n_ctx=2048, verbose=False)
    # out = llm.create_chat_completion(messages=[{"role": "system", "content": PLACE_SYSTEM},
    #                                            {"role": "user", "content": "districts of Karnataka"}], max_tokens=64, temperature=0)
    # print(out["choices"][0]["message"]["content"])

# %% [markdown]
# ## Wrap-up
#
# You built a geocoder from three ingredients: a public gazetteer, templated synthetic
# data verified by execution, and a 0.8B model with a LoRA adapter trained in minutes.
# You also measured it honestly: not just on questions like the training ones, but on
# unseen states, unseen districts, noisy names and compositions, next to a 20B model.
#
# **Homework**
# - Train for 2 epochs and compare the headline table.
# - Add your state's abbreviations and old names to `gazet_tutorial/india.py`, rebuild the noisy split.
# - Add a template (e.g. "X without Y" with `ST_Difference`) and retrain.
# - Change the country code and bbox in notebook 1 to your own country.
#
# **What comes next** (a later session): preference tuning and reinforcement learning
# on exactly this benchmark, using execution as the reward. The plan is in
# [`docs/post-training-roadmap.md`](https://github.com/developmentseed/gazet-tutorial/blob/main/docs/post-training-roadmap.md).
#
# The full-scale version (248 countries, Natural Earth physical features, 81 templates,
# 43k samples) is [gazet](https://github.com/developmentseed/gazet), with a live demo
# at [gazet.ds.io](https://gazet.ds.io).

# %%
# Optional: keep your adapter (Colab only).
# import shutil; from google.colab import files
# files.download(shutil.make_archive(str(OUT / "adapter"), "zip", OUT / "adapter"))
