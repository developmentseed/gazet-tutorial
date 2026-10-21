# Local playground

A Streamlit app that runs the gazet pipeline (extract places → fuzzy search → write SQL
→ execute in DuckDB) against the base model, your fine-tuned model and gpt-oss:20b at
once, and shows what each one returns on a map.

```bash
uv pip install --python .venv/bin/python -e ".[app]"
.venv/bin/streamlit run app/playground.py
```

Two tabs. **Ask** takes a free question. **Benchmark** lets you pick questions from the
benchmark splits (by split, template family, noise type or text), send them through the
same models, and then filter the rows by which model got them right.

The models are `llama serve` processes on ports 8080, 8081 and 8082.
`scripts/serve_models.sh start` launches all three with the published reference model on
8081 (`scripts/serve_models.sh start your.gguf` serves your own instead); how to get there
from a Colab checkpoint is a tutorial of its own: [`docs/take-it-home.md`](../docs/take-it-home.md).
