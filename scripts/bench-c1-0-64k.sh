#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${LLM_INFERENCE_BENCH:-$ROOT/../llm-inference-bench}"
BENCH="$BENCH_DIR/llm_decode_bench.py"
RUN_ID="${RUN_ID:-run-mtp4-c1-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$RUN_ID}"
PORT="${PORT:-9300}"
DURATION="${DURATION:-30}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
DISPLAY_MODE="${DISPLAY_MODE:-screen}"

if [[ ! -f "$BENCH" ]]; then
  printf 'llm-inference-bench not found at %s\n' "$BENCH_DIR" >&2
  printf 'Clone it beside this repository or set LLM_INFERENCE_BENCH.\n' >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
bench_cmd=(
  python3 "$BENCH"
  --host localhost
  --port "$PORT"
  --model GLM-5.2
  --concurrency 1
  --contexts 0,16k,32k,64k
  --standalone-prefill
  --prefill-contexts 8k,16k,32k,64k
  --prefill-duration "$PREFILL_DURATION"
  --prefill-metric auto
  --token-targeting exact
  --duration "$DURATION"
  --decode-warmup-seconds 5
  --max-tokens 8192
  --temperature 0
  --dcp-size 4
  --display-mode "$DISPLAY_MODE"
  --hw-monitor-interval 0.5
  --output "$RUN_DIR/bench.json"
)

if [[ "$DISPLAY_MODE" == "plain" ]]; then
  "${bench_cmd[@]}" 2>&1 | tee "$RUN_DIR/bench.log"
else
  "${bench_cmd[@]}"
fi
