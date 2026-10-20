"""Training pieces shared by notebook 2 and `experiments/modal_train.py`.

Hyperparameters live in `TrainConfig`, so a learning from a Modal sweep is a one-line
default change that reaches the notebook and the scripts alike. Everything that needs
a GPU (`unsloth`, `trl`, `torch`) is imported inside the functions, so the package
installs and tests on a Mac.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"   # Qwen3.5 non-thinking reply starts here


@dataclass
class TrainConfig:
    base_model: str = "unsloth/Qwen3.5-0.8B"
    max_seq_length: int = 1024
    epochs: float = 1.0
    max_steps: int | None = None          # overrides epochs when set (smoke tests)
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int | None = None         # default 2 × r, Unsloth's recommendation for Qwen
    lora_dropout: float = 0.0
    weight_decay: float = 0.01
    batch_size: int = 8
    grad_accum: int = 2                   # effective batch 16
    warmup_steps: int = 20
    scheduler: str = "linear"
    gradient_checkpointing: bool = True   # recompute activations in backward: needed on a 16 GB T4, costs ~30% time on a 48 GB GPU
    group_by_length: bool = False         # batch similar lengths together: less padding, faster (changes sample order)
    eval_every: int = 25                  # steps between val-loss + benchmark checks
    val_cap: int = 128                    # val rows for the loss curve
    bench_n: int = 20                     # benchmark rows per split per task during training
    gen_batch: int = 8                    # generation batch for benchmark eval (places uses 2x); 32+ on a big GPU
    final_n: int = 40                     # benchmark rows per split per task for the end-of-run score
    save_every: int | None = None         # keep an adapter checkpoint every N steps (serve step 50 next to the end in the app)
    save_merged: bool = False             # also save merged 16-bit weights (~1.6 GB) for the GGUF path
    bench_splits: tuple[str, ...] = ("iid", "unseen_region", "unseen_anchor", "noisy")
    data_fraction: float = 1.0           # train on a random fraction of the training rows (data-scale study)
    seed: int = 108
    name: str | None = None

    @property
    def alpha(self) -> int:
        return self.lora_alpha or 2 * self.lora_r

    def run_name(self) -> str:
        return self.name or f"r{self.lora_r}-lr{self.lr:g}-ep{self.epochs:g}-{datetime.now():%m%d-%H%M}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bench_splits"] = list(self.bench_splits)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig field(s): {sorted(unknown)}; known: {sorted(known)}")
        d = dict(d)
        if "bench_splits" in d:
            d["bench_splits"] = tuple(d["bench_splits"])
        return cls(**d)

    def with_overrides(self, **kw) -> TrainConfig:
        """A copy with every non-None keyword applied (CLI flags that were not given stay default)."""
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


# ── model, data, trainer ────────────────────────────────────────────────────

def build_model(cfg: TrainConfig):
    """Base model + fresh LoRA adapter, exactly as notebook 2 does it. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model, processor = FastLanguageModel.from_pretrained(
        cfg.base_model, max_seq_length=cfg.max_seq_length, load_in_4bit=False,
        use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.alpha, lora_dropout=cfg.lora_dropout,
        finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, bias="none", random_state=cfg.seed, use_gradient_checkpointing=False,
    )
    return model, tokenizer


def render(rows: list[dict], tokenizer) -> list[dict]:
    return [{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False),
             "task_family": r["metadata"]["task_family"]} for r in rows]


def load_train_val(dataset_dir: str | Path, tokenizer, cfg: TrainConfig):
    """Both tasks mixed, rendered with the chat template, shuffled; val capped for a cheap loss curve."""
    from datasets import Dataset

    from .data import load_jsonl

    dataset_dir = Path(dataset_dir)
    train_rows = load_jsonl(dataset_dir / "sql/train.jsonl") + load_jsonl(dataset_dir / "places/train.jsonl")
    val_rows = load_jsonl(dataset_dir / "sql/val.jsonl") + load_jsonl(dataset_dir / "places/val.jsonl")
    train_ds = Dataset.from_list(render(train_rows, tokenizer)).shuffle(seed=cfg.seed)
    if cfg.data_fraction < 1.0:
        train_ds = train_ds.select(range(max(1, int(len(train_ds) * cfg.data_fraction))))
    val_ds = Dataset.from_list(render(val_rows, tokenizer)).shuffle(seed=cfg.seed).select(range(min(cfg.val_cap, len(val_rows))))
    return train_ds, val_ds


def build_trainer(model, tokenizer, train_ds, val_ds, cfg: TrainConfig, output_dir: str | Path,
                  report_to: str = "none", project: str = "gazet-tutorial", callbacks: list | None = None):
    """SFTTrainer with loss on the answer only; val loss every `cfg.eval_every` steps."""
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # trackio_space_id defaults to a Hugging Face Space in transformers; None keeps the logs local (TRACKIO_DIR)
    extra = {"project": project, "trackio_space_id": None} if report_to == "trackio" else {}
    args = SFTConfig(
        output_dir=str(output_dir), dataset_text_field="text", max_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum, num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps else -1, learning_rate=cfg.lr, lr_scheduler_type=cfg.scheduler,
        warmup_steps=cfg.warmup_steps, weight_decay=cfg.weight_decay, optim="adamw_8bit", max_grad_norm=1.0,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
        eval_strategy="steps", eval_steps=cfg.eval_every, group_by_length=cfg.group_by_length,
        save_strategy="steps" if cfg.save_every else "no", save_steps=cfg.save_every or 500, save_total_limit=None,
        save_only_model=True,
        report_to=report_to, run_name=cfg.run_name(), seed=cfg.seed, dataset_num_proc=2, **extra,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds,
                         args=args, callbacks=callbacks)
    return train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)


# ── benchmark evaluation (execution accuracy) ───────────────────────────────

def ordered_rows(bench: dict, subset: dict, split: str, task: str, n: int | None) -> list[dict]:
    """Subset rows first (so every model sees them), then the rest, cut to n."""
    rows = bench[split][task]
    ids = subset.get(split, {}).get(task, []) if subset else []
    by_id = {r["metadata"]["sample_id"]: r for r in rows}
    first = [by_id[i] for i in ids if i in by_id]
    rest = [r for r in rows if r["metadata"]["sample_id"] not in set(ids)]
    out = first + rest
    return out[:n] if n else out


def benchmark_eval(model, tokenizer, bench: dict, con, geodata_path, n: int | None, splits, subset: dict | None = None,
                   frames: dict | None = None, gen_batch: int = 8) -> dict[str, float]:
    """Flat metrics {f"{split}/sql_exec", f"{split}/sql_valid", f"{split}/places_f1"}; per-row frames go into `frames` if given."""
    from .eval import evaluate_places, evaluate_sql, generate_batch, prompt_messages

    was_training = getattr(model, "training", False)
    # generation turns the KV cache on; a training forward pass with use_cache=True then keeps caches
    # for every layer of every batch and blows past 40 GB, so remember the flag and restore it below
    cfg_obj = getattr(model, "config", None)
    was_cache = getattr(cfg_obj, "use_cache", None)
    model.eval()
    out: dict[str, float] = {}
    for split in splits:
        if split not in bench:
            continue
        sql_rows = ordered_rows(bench, subset, split, "sql", n)
        pl_rows = ordered_rows(bench, subset, split, "places", n)
        sql_pred = generate_batch(model, tokenizer, [prompt_messages(s) for s in sql_rows], max_new_tokens=256, batch_size=gen_batch, progress=False)
        pl_pred = generate_batch(model, tokenizer, [prompt_messages(s) for s in pl_rows], max_new_tokens=64, batch_size=2 * gen_batch, progress=False)
        sql_df = evaluate_sql(sql_rows, sql_pred, con, geodata_path)
        pl_df = evaluate_places(pl_rows, pl_pred)
        out[f"{split}/sql_exec"] = float(sql_df.exec_match.mean()) if len(sql_df) else 0.0
        if "sql_valid" in sql_df.columns:
            out[f"{split}/sql_valid"] = float(sql_df.sql_valid.mean()) if len(sql_df) else 0.0
        out[f"{split}/places_f1"] = float(pl_df.f1.mean()) if len(pl_df) else 0.0
        if frames is not None:
            frames[split] = {"sql": sql_df, "places": pl_df}
    if cfg_obj is not None and was_cache is not None:
        cfg_obj.use_cache = was_cache
    if was_training:
        model.train()
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def make_benchmark_callback(evaluate, every: int, prefix: str = "bench/"):
    """A TrainerCallback that runs `evaluate(model, tokenizer) -> dict` every `every` steps and logs it
    through the trainer (so it reaches trackio next to train/eval loss). Set `cb.trainer` after building."""
    from transformers import TrainerCallback

    class BenchmarkCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.history: list[dict] = []

        def on_step_end(self, args, state, control, **kw):
            if state.global_step and state.global_step % every == 0:
                self._run(state.global_step, kw.get("model"), kw.get("processing_class") or kw.get("tokenizer"))
            return control

        def _run(self, step, model, tokenizer):
            if model is None or self.trainer is None:
                return
            metrics = {prefix + k: v for k, v in evaluate(model, tokenizer).items()}
            self.history.append({"step": step, **metrics})
            self.trainer.log(metrics)

    return BenchmarkCallback()


def save_run(run_dir: str | Path, model, tokenizer, cfg: TrainConfig, final: dict, log_history: list, extra: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    (run_dir / "eval.json").write_text(json.dumps({"final": final, **(extra or {})}, indent=1))
    (run_dir / "log_history.json").write_text(json.dumps(log_history, indent=1))
    return run_dir
