"""Merge a LoRA adapter (a run's final adapter or any checkpoint) into the base model, on CPU.

    uv run --with "transformers>=5" --with torch --with peft --with sentencepiece \
        python experiments/merge_adapter.py work/runs/reference/trainer/checkpoint-50 work/merged/reference-step50

Then convert and quantize the output folder exactly as docs/take-it-home.md does for the
Colab checkpoint. Runs with `save_merged: true` already have `merged/` for the final step.
"""

import json
import sys
from pathlib import Path

adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((adapter / "adapter_config.json").read_text())
base = cfg["base_model_name_or_path"]
print(f"base {base} + adapter {adapter} -> {out}")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
model.save_pretrained(str(out), safe_serialization=True)
AutoTokenizer.from_pretrained(str(adapter) if (adapter / "tokenizer_config.json").exists() else base).save_pretrained(str(out))
print("done:", sorted(p.name for p in out.iterdir()))
