import pytest

from gazet_tutorial.train import (
    INSTRUCTION_PART,
    RESPONSE_PART,
    TrainConfig,
    make_benchmark_callback,
    ordered_rows,
)


def test_train_config_roundtrip_and_defaults():
    cfg = TrainConfig()
    assert cfg.alpha == 32 and cfg.run_name().startswith("r16-lr0.0002-ep1-")
    assert TrainConfig(lora_r=8).alpha == 16 and TrainConfig(lora_alpha=64, lora_r=8).alpha == 64
    d = cfg.to_dict()
    assert d["bench_splits"] == ["iid", "unseen_region", "unseen_anchor", "noisy"]
    assert TrainConfig.from_dict(d) == cfg
    assert TrainConfig.from_dict({"lr": 1e-4, "name": "x"}).run_name() == "x"
    with pytest.raises(ValueError, match="unknown TrainConfig field"):
        TrainConfig.from_dict({"learning_rate": 1e-4})
    assert TrainConfig(data_fraction=0.5).data_fraction == 0.5
    over = cfg.with_overrides(lr=None, epochs=2, lora_r=32)
    assert over.lr == cfg.lr and over.epochs == 2 and over.alpha == 64
    assert RESPONSE_PART.startswith("<|im_start|>assistant") and INSTRUCTION_PART == "<|im_start|>user\n"


def test_ordered_rows_puts_subset_first():
    rows = [{"metadata": {"sample_id": f"s{i}"}} for i in range(5)]
    bench = {"iid": {"sql": rows}}
    out = ordered_rows(bench, {"iid": {"sql": ["s3", "s1"]}}, "iid", "sql", 3)
    assert [r["metadata"]["sample_id"] for r in out] == ["s3", "s1", "s0"]
    assert len(ordered_rows(bench, {}, "iid", "sql", None)) == 5


def test_benchmark_callback_logs_every_n_steps():
    class State:
        global_step = 0

    class Trainer:
        def __init__(self):
            self.logged = []

        def log(self, m):
            self.logged.append(m)

    calls = []
    cb = make_benchmark_callback(lambda m, t: calls.append(1) or {"iid/sql_exec": 0.5}, every=2)
    cb.trainer = Trainer()
    state = State()
    for step in range(1, 5):
        state.global_step = step
        cb.on_step_end(None, state, None, model=object(), processing_class=object())
    assert len(calls) == 2 and cb.trainer.logged == [{"bench/iid/sql_exec": 0.5}] * 2
    assert [h["step"] for h in cb.history] == [2, 4]
