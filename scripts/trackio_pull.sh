#!/usr/bin/env bash
# Pull the trackio database and run outputs from the Modal volume and open the dashboard.
# Usage: scripts/trackio_pull.sh            # dashboard
#        scripts/trackio_pull.sh <run-name> # also fetch runs/<run-name>/adapter into work/runs/<run-name>
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/trackio work/runs
.venv/bin/modal volume get --force gazet-tutorial trackio work/
if [[ -n "${1:-}" ]]; then
  .venv/bin/modal volume get --force gazet-tutorial "runs/$1" work/runs/
  echo "adapter at work/runs/$1/adapter (use as ADAPTER_DIR in notebook 2, or merge + convert per docs/take-it-home.md)"
fi
TRACKIO_DIR="$PWD/work/trackio" exec .venv/bin/trackio show --project gazet-tutorial
