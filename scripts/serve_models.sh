#!/usr/bin/env bash
# Convenience: serve the playground's models with `llama serve` (llama.app CLI), one port each.
# Learn the individual commands first: docs/take-it-home.md.
#   scripts/serve_models.sh start [model.gguf ...]   # base :8080, gpt-oss :8082, each GGUF on :8081, :8083, :8084 ...
#   scripts/serve_models.sh stop
#   SKIP_GPT_OSS=1 scripts/serve_models.sh start reference-q8_0.gguf short-q8_0.gguf reference-step50-q8_0.gguf
# Each GGUF becomes a sidebar row in the app, named after the file (work/served.json).
# Models given with -hf are downloaded into llama's cache on first start (no account needed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/logs
BASE_HF="${BASE_HF:-unsloth/Qwen3.5-0.8B-GGUF:Q8_0}"
GPT_OSS_HF="${GPT_OSS_HF:-ggml-org/gpt-oss-20b-GGUF:MXFP4}"
COMMON=(--host 127.0.0.1 --jinja -ngl 99)

start() {
  command -v llama >/dev/null || { echo "llama not found: install it from https://llama.app/docs/installation"; exit 1; }
  echo "base      :8080  $BASE_HF"
  nohup llama serve -hf "$BASE_HF" --port 8080 -c 4096 -np 2 "${COMMON[@]}" > work/logs/base.log 2>&1 &
  if [[ -z "${SKIP_GPT_OSS:-}" ]]; then
    echo "gpt-oss   :8082  $GPT_OSS_HF (12 GB download on first start)"
    nohup llama serve -hf "$GPT_OSS_HF" --port 8082 -c 8192 -np 1 "${COMMON[@]}" > work/logs/gpt-oss.log 2>&1 &
  fi
  local port=8081 rows=()
  for gguf in "$@"; do
    [[ -f "$gguf" ]] || { echo "not a file: $gguf"; exit 1; }
    local name; name="$(basename "$gguf" .gguf)"
    echo "$name  :$port  $gguf"
    nohup llama serve -m "$gguf" --port "$port" -c 4096 -np 2 -a "$name" "${COMMON[@]}" > "work/logs/$name.log" 2>&1 &
    rows+=("{\"name\": \"$name\", \"port\": $port}")
    port=$((port == 8081 ? 8083 : port + 1))   # 8082 is gpt-oss
  done
  printf '[%s]\n' "$(IFS=,; echo "${rows[*]}")" > work/served.json
  echo "logs in work/logs/; rows for the app in work/served.json; health: curl -s localhost:8080/v1/models"
}

stop() {
  pkill -f "llama serve .*--port 80[89][0-9]" && echo "stopped" || echo "nothing running"
  rm -f work/served.json
}

case "${1:-}" in
  start) shift; start "$@" ;;
  stop) stop ;;
  *) echo "usage: $0 start [model.gguf ...] | stop"; exit 1 ;;
esac
