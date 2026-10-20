#!/usr/bin/env bash
# Convert the percent-format sources in notebooks/src/ into the .ipynb files Colab opens.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in notebooks/src/*.py; do
  .venv/bin/jupytext --to ipynb --set-kernel python3 "$f" -o "notebooks/$(basename "${f%.py}").ipynb"
done
