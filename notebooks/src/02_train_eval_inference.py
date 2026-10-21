# %% [markdown]
# # Notebook 2 · Fine-tune, geocode, export
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/developmentseed/gazet-tutorial/blob/main/notebooks/02_train_eval_inference.ipynb)
#
# Part 2 of the tutorial. In notebook 1 we built ~3,600 synthetic examples of two
# tasks over India's administrative boundaries: *extract place names from a question*
# and *write spatial SQL over a candidate table*. Here we teach both to one small
# model, run it as a geocoder, and export it in the format gazet deploys.
#
# **Runtime → Change runtime type → T4 GPU** before you start. Free-tier T4 is enough.
#
# **In this notebook (~45 minutes):**
#
# 1. Load Qwen3.5-0.8B with Unsloth and attach LoRA adapters.
# 2. Render the chat examples and mask everything except the assistant's answer.
# 3. Train for one epoch (~20 minutes after a few minutes of kernel compilation).
# 4. Run the full geocoder on real questions and look at the results on a map.
# 5. Export to GGUF for CPU inference, the way gazet is deployed, and take it home.
#
# Scoring the model on the benchmark splits from notebook 1 (base vs fine-tuned vs
# gpt-oss:20b, by executing the SQL) is done offline against the exported GGUF, so
# this notebook stays short enough for the session.
#
# No accounts are needed. The dataset ships inside the package you `pip install`,
# so it is already on this machine.

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
GEODATA_SOURCE = None     # None -> download India divisions from Overture S3 (~60 MB, 2-3 min)
MAX_SEQ_LENGTH = 1024     # notebook 1 measured P99 ≈ 820 tokens for SQL examples
EPOCHS = 1                # 2 epochs noticeably better; left as homework to fit the session
MAX_STEPS = None          # set to 20 for a smoke test
RUN_GGUF = True           # export a q8_0 GGUF at the end (builds llama.cpp, adds ~10 min)
SEED = 108
TRACK = False             # log loss curves with trackio: stays on this VM, dashboard embeds below the training cell

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
from gazet_tutorial.eval import generate_batch, parse_places
from gazet_tutorial.viz import plot_loss, map_features

pd.set_option("display.max_colwidth", 100)
torch.manual_seed(SEED)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (training will not work)")

ART = artifacts_dir()     # the artifacts/ folder bundled inside the installed package
DATASET_DIR = Path(DATASET_DIR) if DATASET_DIR else ART / "dataset"
OUT = data_dir() / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
prompts = json.loads((DATASET_DIR / "prompts.json").read_text())
SQL_SYSTEM, PLACE_SYSTEM, USER_TEMPLATE = prompts["sql_system"], prompts["place_system"], prompts["user_template"]
print("dataset:", DATASET_DIR)

# %% [markdown]
# The geodata is needed again for the live demo (we execute the SQL the model
# writes). Each Colab notebook is a fresh machine, so we fetch it again: ~60 MB
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
    eval_strategy="steps" if TRACK else "no",   # val loss along the way only when tracking
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
# The adapter is ~60 MB and lives on this VM only. The GGUF we export at the end is
# the file you take home; if you also want the adapter, zip and download it (last cell).

# %% [markdown]
# ## 5. The geocoder, end to end
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
# Each of those is one of the benchmark splits notebook 1 built (`unseen_region`,
# `noisy`, `compositional`). We score them offline against the exported GGUF with
# `scripts/benchmark_gguf.py`, next to the base model and gpt-oss:20b; each gap is
# the kind you close with more data, a new template, or a preference-tuning stage on
# top of this model.

# %%
for q in ["districts of Kerala", "districts of UP", "districts of Karnataka that border Tamil Nadu"]:
    geocode(q)

# %%
r = geocode("neighbours of Lakshadweep")   # a trick question: islands touch nothing
# map_features(r["result"], anchors=r["anchors"])

# %% [markdown]
# ## 6. Export to GGUF and take the model home
#
# gazet runs this model on CPU with llama.cpp's server from a quantized GGUF file (~0.9 GB
# at 8 bits). The cell below merges the adapter into the base weights, converts the
# result with llama.cpp's converter and quantizes it to q8_0, all on this VM (~10 min,
# most of it building llama.cpp). Download the `.gguf` it prints, then back on your
# laptop follow `docs/take-it-home.md`: `llama serve` it next to the base model and
# gpt-oss:20b, compare the three in the Streamlit playground (`app/README.md`) with maps,
# and score it on the benchmark with `scripts/benchmark_gguf.py`, which prints your model's
# execution accuracy per split next to gpt-oss:20b and the base model.

# %%
if RUN_GGUF:
    # Unsloth writes the merged 16-bit checkpoint to OUT/merged (1.7 GB, a by-product)
    # and the quantized file next to it, in OUT/merged_gguf/.
    model.save_pretrained_gguf(str(OUT / "merged"), tokenizer, quantization_method="q8_0")
    ggufs = sorted(OUT.rglob("*.gguf"), key=lambda p: p.stat().st_mtime)
    if not ggufs:
        raise FileNotFoundError(f"no .gguf written under {OUT}; scroll up for the converter's error")
    for p in ggufs:
        print(f"{p.relative_to(OUT)}  {p.stat().st_size/1e9:.2f} GB")
    gguf_path = next((p for p in ggufs if "q8_0" in p.name.lower()), ggufs[-1])
    print("\nGGUF:", gguf_path)
    # Download it to your laptop (Colab only, ~0.9 GB; or use the file browser on the left):
    # from google.colab import files; files.download(str(gguf_path))

    # CPU inference with llama-cpp-python, using the chat template embedded in the GGUF:
    # !pip install -q llama-cpp-python
    # from llama_cpp import Llama
    # llm = Llama(model_path=str(gguf_path), n_ctx=2048, verbose=False)
    # out = llm.create_chat_completion(messages=[{"role": "system", "content": PLACE_SYSTEM},
    #                                            {"role": "user", "content": "districts of Karnataka"}], max_tokens=64, temperature=0)
    # print(out["choices"][0]["message"]["content"])

# %% [markdown]
# ## Wrap-up
#
# You built a geocoder from three ingredients: a public gazetteer, templated synthetic
# data verified by execution, and a 0.8B model with a LoRA adapter trained in minutes.
# The GGUF you just exported is the same artifact gazet deploys.
#
# **Homework**
# - Score the GGUF on the benchmark (`scripts/benchmark_gguf.py`): not just on questions
#   like the training ones, but on unseen states, unseen districts, noisy names and
#   compositions, next to the base model and gpt-oss:20b.
# - Train for 2 epochs, export again, and compare the two on the benchmark.
# - Add your state's abbreviations and old names to `gazet_tutorial/india.py`, rebuild the noisy split.
# - Add a template (e.g. "X without Y" with `ST_Difference`) and retrain.
# - Change the country code and bbox in notebook 1 to your own country.
#
# **What comes next** (a later session): preference tuning and reinforcement learning
# on exactly this benchmark, using execution as the reward.
#
# The full-scale version (248 countries, Natural Earth physical features, 81 templates,
# 43k samples) is [gazet](https://github.com/developmentseed/gazet), with a live demo
# at [gazet.ds.io](https://gazet.ds.io).

# %%
# Optional: also keep the adapter (Colab only).
# import shutil; from google.colab import files
# files.download(shutil.make_archive(str(OUT / "adapter"), "zip", OUT / "adapter"))
