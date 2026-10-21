# Take the model home: from Colab to a local geocoder, step by step

Notebook 2 leaves you with a q8_0 GGUF of the fine-tuned model on a Colab machine. This
page walks through serving it next to the base model and gpt-oss:20b, and comparing the
three in a small app. Every step is a command you type; read what each one does before
running it, because this is the part of the workshop that is most useful back at work.

You need: macOS or Linux, [`uv`](https://docs.astral.sh/uv/), `git`, about 3 GB of disk
(15 GB if you also serve gpt-oss), and a clone of this repository.

## 1. Get the model out of Colab

A LoRA adapter is a small set of delta matrices that only make sense on top of the exact
base model. For deployment you **merge** them into the base weights once, convert the
result to llama.cpp's file format and quantize it. Notebook 2's last cell does all three
on the Colab VM (`RUN_GGUF = True`, the default) and prints the path of the resulting
`.gguf` (about 0.9 GB). Download it from the Colab file browser, or uncomment the
`files.download(...)` line in that cell, and put it somewhere handy:

```bash
cd /path/to/gazet-tutorial
mv ~/Downloads/*q8_0*.gguf gazet-in-q8_0.gguf     # the name the serve commands below use
```

No GGUF of your own? The workshop's reference run is published as
[developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF](https://huggingface.co/developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF);
in step 6 use `llama serve -hf developmentseed/gazet-tutorial-qwen3.5-0.8b-GGUF:Q8_0` in place
of `-m gazet-in-q8_0.gguf` and llama.cpp downloads it (0.8 GB, no account). The adapter it was
made from is at
[developmentseed/gazet-tutorial-qwen3.5-0.8b-lora](https://huggingface.co/developmentseed/gazet-tutorial-qwen3.5-0.8b-lora).

With the GGUF in hand you can skip straight to [step 2](#2-install-llamacpp) for the
runtime and then to serving. Steps 3 and 4 show how the conversion works if you would
rather do it yourself from the saved adapter (`outputs/adapter/` on the VM, merged with
`peft`), or if you need a different quantization.

## 2. Install llama.cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) is the inference engine gazet runs in
production. Its binaries now ship as one `llama` command with subcommands (`llama serve`,
`llama cli`, `llama quantize`, `llama download`, …). Install it as described at
<https://llama.app/docs/installation>:

```bash
curl -LsSf https://llama.app/install.sh | sh
llama version        # e.g. "version: 0.7.0-dev (build 11706, commit f74e2fa6a)"
```

Note the **build number**; you need it in the next step. The CLI reference is at
<https://llama.app/docs/cli> and the server at <https://llama.app/docs/serve>.

## 3. Get the converter that matches your build

Converting a Hugging Face checkpoint to llama.cpp's file format is done by a Python
script, `convert_hf_to_gguf.py`, which is not part of the binary install; it lives in
the llama.cpp source tree. Converter and runtime must come from the **same release**:
the file format is stable, but how a given architecture's tensors are named and laid out
inside it moves with the code, and a file written by a mismatched converter can fail to
load with an error like `missing tensor 'blk.24.ssm_conv1d.weight'`. llama.cpp tags every
build as `b<number>`, so clone the tag that matches `llama version`:

```bash
TAG="b$(llama version 2>&1 | grep -o 'build [0-9]*' | grep -o '[0-9]*')"
echo $TAG                                                   # b11706 for the build above
git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp ~/llama.cpp
```

## 4. Convert the checkpoint to GGUF

GGUF is a single-file format that stores the weights together with the tokenizer and the
chat template, so the server needs nothing else. The converter needs PyTorch and
transformers to read the checkpoint. Run it in a throwaway environment with `uv run` so
none of that lands in your project:

```bash
uv run --with "transformers>=5" --with torch --with gguf --with sentencepiece --with protobuf --with "numpy<3" \
  python ~/llama.cpp/convert_hf_to_gguf.py ~/Downloads/merged \
  --outtype bf16 --outfile gazet-in-bf16.gguf
```

Why not llama.cpp's own `requirements/requirements-convert_hf_to_gguf.txt`? It pins
transformers 4.x, and a checkpoint saved by transformers 5 (which Unsloth uses in the
notebook) records a tokenizer class that 4.x does not know (`TokenizersBackend`), so the
pinned version stops with a `ValueError`. Version skew between training and deployment
tooling is a normal part of this work; reading the traceback is the skill.

The first run downloads the requirements (PyTorch is the big one); later runs are
instant. `--outtype bf16` keeps the weights as they were trained. Look at the size:

```bash
ls -lh gazet-in-bf16.gguf        # ~1.5 GB for a 0.8B-parameter model
```

## 5. Quantize

Quantization stores each weight in fewer bits. Q8_0 (8-bit) is close to lossless and
halves the file; smaller types (Q4_K_M and friends) cut it further at some cost in
accuracy, which for a 0.8B model doing exact id lookups you do not want. gazet ships
Q8_0.

```bash
llama quantize gazet-in-bf16.gguf gazet-in-q8_0.gguf Q8_0
ls -lh gazet-in-*.gguf           # bf16 1.5 GB -> q8_0 0.8 GB
```

## 6. Serve it and talk to it

`llama serve` exposes an OpenAI-compatible API (default port 9931; we give each model its
own port). `--jinja` makes it use the chat template embedded in the GGUF (the one the
notebook trained with), `-c` is the context window in tokens (a SQL prompt is about 800),
`-ngl 99` puts every layer on the GPU.

```bash
llama serve -m gazet-in-q8_0.gguf --port 8081 --jinja -c 4096 -ngl 99
```

In a second terminal, ask it to extract places. The `chat_template_kwargs` turns Qwen3.5's
thinking mode off, exactly as during training:

```bash
curl -s localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "You are a geographic entity extractor. Extract the place names the user is asking about and return valid JSON only.\n\nOUTPUT FORMAT:\n{\"places\": [{\"place\": \"<name>\"}]}"},
    {"role": "user", "content": "districts within 50 km of Nagpur"}
  ],
  "temperature": 0, "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

Expected: `{"places": [{"place": "Nagpur"}]}`. (The full system prompts the notebook used
are in `artifacts/dataset/prompts.json`; the app sends those.)

## 7. Score it on the benchmark

The notebook stopped at a map. The benchmark from notebook 1 is how you find out what the
model actually learned, and it runs against the GGUF you just served, with execution
accuracy as the metric (predicted and reference SQL must return the same feature ids):

```bash
.venv/bin/python scripts/benchmark_gguf.py gazet-in-q8_0.gguf
```

It starts its own `llama serve` on port 8089 (or pass `--base-url http://localhost:8081/v1
--name gazet-in` to use the one from step 6), asks the 40 SQL + 20 extraction questions
of every split, writes the answers to `work/benchmark/gazet-in-q8_0/`, scores them and
prints two tables: your model next to the committed zero-shot gpt-oss:20b and the
untouched Qwen3.5-0.8B, one column per split. About five minutes on a laptop GPU. The
run is resumable, `--limit 3` is a smoke test, and `--compare-only` reprints the tables.
Read the columns right to left: `iid` flatters a small model; `unseen_region`, `noisy`
and `compositional` are where the interesting gaps are.

## 8. The other two models

The **base model** is what you started from, before fine-tuning. Unsloth publishes it as
GGUF, and `llama serve` downloads straight from the Hub into its cache (public, no
account); the `:Q8_0` suffix picks the quantization:

```bash
llama serve -hf unsloth/Qwen3.5-0.8B-GGUF:Q8_0 --port 8080 --jinja -c 4096 -ngl 99
```

**gpt-oss:20b** is the model gazet calls when there is no fine-tuned one. It is 12 GB in
its native 4-bit format and needs about 16 GB of RAM:

```bash
llama serve -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 --port 8082 --jinja -c 8192 -ngl 99
```

If that is too big for your machine, skip it: the app compares whichever servers are
up. If you run Ollama, point the gpt-oss row at `http://127.0.0.1:11434/v1` with model
`gpt-oss:20b` in the app's sidebar instead. `llama serve -cl` lists what is in the
download cache.

Each server is one process on one port; stop them with Ctrl-C. Once the commands are
familiar, `scripts/serve_models.sh start gazet-in-q8_0.gguf` launches all three in the
background.

## 9. Compare them in the app

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[app]"     # the tutorial package + Streamlit
.venv/bin/streamlit run app/playground.py
```

The sidebar lists the three endpoints with a live health dot; edit the table to add any
other OpenAI-compatible server. Then:

- **Ask**: type a question. Each live model gets a column with the places it extracted,
  the candidate table the fuzzy search returned, the SQL it wrote, the rows that came
  back and a map. Try the held-out state (*districts of Kerala*), an abbreviation
  (*districts of UP*) and a composition (*districts of Karnataka that border Tamil Nadu*).
- **Benchmark**: pick questions from a split, narrowed by template family, noise type
  or a word in the question, and send them through every live model. Each row gets a
  tick or cross per model; filter to the rows one model failed, click one, and you see
  the gold features next to what each model returned, on maps.

What you have at the end is gazet's production shape: a small quantized model behind an
HTTP API, a candidate search in DuckDB, and a UI that only ever talks HTTP. Swapping the
model is a one-line change to the server command.
