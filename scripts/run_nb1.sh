#!/usr/bin/env bash
# Regenerate the bundled artifacts by executing notebook 1 locally.
# Usage: scripts/run_nb1.sh [local_overture_glob]
#   With a local Overture divisions parquet glob the run avoids the S3 download.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:-}"
mkdir -p work
.venv/bin/jupytext --to ipynb --set-kernel python3 notebooks/src/01_data_and_dataset.py -o notebooks/01_data_and_dataset.ipynb
rm -rf artifacts/dataset artifacts/relations artifacts/benchmark
ARGS=(-p OUT_DIR "$PWD/artifacts")
if [[ -n "$SRC" ]]; then ARGS+=(-p GEODATA_SOURCE "$SRC"); fi
.venv/bin/papermill notebooks/01_data_and_dataset.ipynb work/01_executed.ipynb -k gazet-tutorial --cwd notebooks "${ARGS[@]}"
echo "artifacts regenerated under artifacts/ (baselines/ untouched)"
