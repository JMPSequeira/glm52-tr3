#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL_ID="${MODEL_ID:-glm52-tr3-mtp4}"
PORT="${PORT:-9300}"
MTP="${MTP:-4}"
MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}"
MTP_BATCH_SCHEDULE="${MTP_BATCH_SCHEDULE:-}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-}"
IMAGE="${IMAGE:-glm52-tr3:runtime-v2}"
ENTRYPOINT="${ENTRYPOINT:-/opt/glm52-tr3/serve-final.sh}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache/huggingface}"
CACHE="${CACHE:-$HOME/.cache/vllm-glm52-tr3-v19}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-podman}"
MODEL="${MODEL:-/models/hub/models--brandonmusic--GLM-5.2-NVFP4-TR3-Hybrid/snapshots/002eb6732dd8def0359915572eb5e22129244321}"
TR3_ROUTE1="${TR3_ROUTE1:-0}"
TR3_OVERLAP="${TR3_OVERLAP:-0}"
TRELLIS256="${TRELLIS256:-0}"
TRELLIS256_MIN_M="${TRELLIS256_MIN_M:-1}"
TRELLIS256_MAX_M="${TRELLIS256_MAX_M:-16}"
TRELLIS256_TILE_CONFIG="${TRELLIS256_TILE_CONFIG:-}"
PLANNED_TAIL="${PLANNED_TAIL:-1}"
PLANNED_TAIL_PARITY="${PLANNED_TAIL_PARITY:-0}"
PLANNED_TAIL_PARITY_REL_MAX="${PLANNED_TAIL_PARITY_REL_MAX:-0.005}"
PLANNED_TAIL_PARITY_COS_MIN="${PLANNED_TAIL_PARITY_COS_MIN:-0.99999}"
PLANNED_TAIL_OVERLAP="${PLANNED_TAIL_OVERLAP:-1}"
PLANNED_TAIL_MIN_M="${PLANNED_TAIL_MIN_M:-1}"
PLANNED_TAIL_MAX_M="${PLANNED_TAIL_MAX_M:-32}"
PLANNED_TAIL_BLOCK_M="${PLANNED_TAIL_BLOCK_M:-8}"
PLANNED_PREFILL="${PLANNED_PREFILL:-1}"
PLANNED_PREFILL_MAX_M="${PLANNED_PREFILL_MAX_M:-5120}"
PLANNED_PREFILL_BLOCK_M="${PLANNED_PREFILL_BLOCK_M:-64}"
PLANNED_PREFILL_PARITY_MAX_M="${PLANNED_PREFILL_PARITY_MAX_M:-128}"
PLANNED_PREFILL_PARITY_REL_MAX="${PLANNED_PREFILL_PARITY_REL_MAX:-0.01}"
PLANNED_PREFILL_PARITY_COS_MIN="${PLANNED_PREFILL_PARITY_COS_MIN:-0.99995}"
MIXED_TRELLIS="${MIXED_TRELLIS:-1}"
MIXED_TRELLIS_MAX_M="${MIXED_TRELLIS_MAX_M:-4}"
MIXED_TRELLIS_PARITY="${MIXED_TRELLIS_PARITY:-0}"
MIXED_TRELLIS_PARITY_REL_MAX="${MIXED_TRELLIS_PARITY_REL_MAX:-0.01}"
MIXED_TRELLIS_PARITY_COS_MIN="${MIXED_TRELLIS_PARITY_COS_MIN:-0.99995}"
MIXED_TRELLIS_TILE_CONFIG="${MIXED_TRELLIS_TILE_CONFIG:-128,128,128,128}"
FUSED_TR3="${FUSED_TR3:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-5120}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GRAPH="${GRAPH:-16}"
DISABLE_SHARED_EXPERTS_STREAM="${DISABLE_SHARED_EXPERTS_STREAM:-0}"
TR3_CAP="${TR3_CAP:-128}"
CKV_GATHER="${CKV_GATHER:-1}"
CKV_GATHER_MIN_TOKENS="${CKV_GATHER_MIN_TOKENS:-512}"
CKV_GATHER_MAX_TOKENS="${CKV_GATHER_MAX_TOKENS:-65536}"
DCP_A2A_MAX_TOKENS="${DCP_A2A_MAX_TOKENS:-64}"
DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-0}"
CKV_PREFETCH_DEPTH="${CKV_PREFETCH_DEPTH:-0}"
SPARSE_CKV_GATHER="${SPARSE_CKV_GATHER:-0}"
SPARSE_CKV_TRANSPORT="${SPARSE_CKV_TRANSPORT:-direct}"
SPARSE_CKV_BULK_PREFETCH="${SPARSE_CKV_BULK_PREFETCH:-0}"
SPARSE_CKV_MAX_SEQS="${SPARSE_CKV_MAX_SEQS:-$MAX_NUM_SEQS}"
SPARSE_CKV_POOL_RECORDS="${SPARSE_CKV_POOL_RECORDS:-0}"
SPEC_EXTEND_AS_DECODE="${SPEC_EXTEND_AS_DECODE:-auto}"
SPEC_DECODE_MAX_Q="${SPEC_DECODE_MAX_Q:-8}"
ABSORB_BMM="${ABSORB_BMM:-0}"
MTP_LOCAL_ARGMAX="${MTP_LOCAL_ARGMAX:-0}"
PROFILE_HOST_DIR="${PROFILE_HOST_DIR:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
SERVER_ARGS=(
  --seed 0
  --num-gpu-blocks-override 4096
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  SERVER_ARGS+=(--enforce-eager)
fi
EXTRA_ARGS=()
if [[ -n "${GPUS:-}" ]]; then
  EXTRA_ARGS+=(-e "GPUS=$GPUS")
fi
if [[ -n "$PROFILE_HOST_DIR" ]]; then
  mkdir -p "$PROFILE_HOST_DIR"
  EXTRA_ARGS+=(-v "$PROFILE_HOST_DIR:/tmp/vllm-profile")
  SERVER_ARGS+=(
    --profiler-config.profiler=torch
    --profiler-config.torch_profiler_dir=/tmp/vllm-profile
    --profiler-config.torch_profiler_with_stack=false
    --profiler-config.torch_profiler_record_shapes=false
    --profiler-config.torch_profiler_with_memory=false
    --profiler-config.torch_profiler_with_flops=false
    --profiler-config.torch_profiler_use_gzip=true
    --profiler-config.torch_profiler_dump_cuda_time_total=false
    --profiler-config.ignore_frontend=true
    --profiler-config.delay_iterations=0
    --profiler-config.max_iterations=0
    --profiler-config.warmup_iterations=0
    --profiler-config.active_iterations=5
    --profiler-config.wait_iterations=0
  )
fi
SERVER_ARGS+=("$@")

port_accepting() {
  python3 - "$PORT" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}
# Removing the old name only handles this recipe's container. A different
# detached experiment can still own the host-network port and make its
# /health response look like readiness for the new container.
"$ENGINE" rm --force "$MODEL_ID" >/dev/null 2>&1 || true
if port_accepting; then
  printf 'Refusing to launch %s: port %s already accepts connections\n' \
    "$MODEL_ID" "$PORT" >&2
  exit 98
fi

mkdir -p "$CACHE"

container_id="$("$ENGINE" run \
  --detach \
  --name "$MODEL_ID" \
  --gpus all \
  --network host \
  --ipc host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${EXTRA_ARGS[@]}" \
  --entrypoint "$ENTRYPOINT" \
  -e "MODEL=$MODEL" \
  -e MODEL_REVISION= \
  -e SERVED_MODEL_NAME=GLM-5.2 \
  -e "PORT=$PORT" \
  -e TP=4 \
  -e DCP=4 \
  -e DCP_BACKEND=a2a \
  -e "DCP_A2A_MAX_TOKENS=$DCP_A2A_MAX_TOKENS" \
  -e DCP_A2A_LARGE_BACKEND=ag_rs \
  -e DCP_PREFILL_WORKSPACE=1 \
  -e "MTP=$MTP" \
  -e "MTP_DRAFT_SAMPLE_METHOD=$MTP_DRAFT_SAMPLE_METHOD" \
  -e "MTP_BATCH_SCHEDULE=$MTP_BATCH_SCHEDULE" \
  -e "CUDAGRAPH_CAPTURE_SIZES=$CUDAGRAPH_CAPTURE_SIZES" \
  -e "VLLM_MTP_LOCAL_ARGMAX=$MTP_LOCAL_ARGMAX" \
  -e "MAX_NUM_SEQS=$MAX_NUM_SEQS" \
  -e "GRAPH=$GRAPH" \
  -e MAX_MODEL_LEN=1048576 \
  -e "MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS" \
  -e GPU_MEMORY_UTILIZATION=0.976 \
  -e KV_CACHE_DTYPE=nvfp4_ds_mla \
  -e MOE_MODE=a16 \
  -e MOE_BACKEND=b12x \
  -e LINEAR_BACKEND=auto \
  -e ONLINE_QUANT=nf3-mxfp8 \
  -e QUANTIZATION=nvfp4_tr3_hybrid \
  -e LOAD_FORMAT=safetensors \
  -e TR3_PREFIX_CACHE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e "VLLM_DISABLE_SHARED_EXPERTS_STREAM=$DISABLE_SHARED_EXPERTS_STREAM" \
  -e "DRY_RUN=${DRY_RUN:-0}" \
  -e "B12X_TR3_ROUTE1=$TR3_ROUTE1" \
  -e "B12X_TR3_OVERLAP=$TR3_OVERLAP" \
  -e "B12X_TR3_MAX_TOKENS_PER_EXPERT=$TR3_CAP" \
  -e "B12X_TRELLIS256_MOE=$TRELLIS256" \
  -e "VLLM_DCP_QUERY_SPLIT=$DCP_QUERY_SPLIT" \
  -e "VLLM_B12X_MLA_CKV_GATHER=$CKV_GATHER" \
  -e "VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS=$CKV_GATHER_MIN_TOKENS" \
  -e "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=$CKV_GATHER_MAX_TOKENS" \
  -e "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=$CKV_PREFETCH_DEPTH" \
  -e "VLLM_B12X_MLA_SPARSE_DECODE_CKV_GATHER=$SPARSE_CKV_GATHER" \
  -e "VLLM_B12X_MLA_SPARSE_DECODE_TRANSPORT=$SPARSE_CKV_TRANSPORT" \
  -e "VLLM_B12X_MLA_SPARSE_DECODE_BULK_PREFETCH=$SPARSE_CKV_BULK_PREFETCH" \
  -e "VLLM_B12X_MLA_SPARSE_DECODE_MAX_SEQS=$SPARSE_CKV_MAX_SEQS" \
  -e "VLLM_B12X_MLA_SPARSE_DECODE_POOL_RECORDS=$SPARSE_CKV_POOL_RECORDS" \
  -e "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=$SPEC_EXTEND_AS_DECODE" \
  -e "VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=$SPEC_DECODE_MAX_Q" \
  -e "VLLM_B12X_ABSORB_BMM=$ABSORB_BMM" \
  -e "B12X_TRELLIS256_MIN_M=$TRELLIS256_MIN_M" \
  -e "B12X_TRELLIS256_MAX_M=$TRELLIS256_MAX_M" \
  -e "B12X_TRELLIS256_TILE_CONFIG=$TRELLIS256_TILE_CONFIG" \
  -e "VLLM_TR3_PLANNED_TAIL=$PLANNED_TAIL" \
  -e "VLLM_TR3_PLANNED_TAIL_PARITY=$PLANNED_TAIL_PARITY" \
  -e "VLLM_TR3_PLANNED_TAIL_PARITY_REL_MAX=$PLANNED_TAIL_PARITY_REL_MAX" \
  -e "VLLM_TR3_PLANNED_TAIL_PARITY_COS_MIN=$PLANNED_TAIL_PARITY_COS_MIN" \
  -e "VLLM_TR3_PLANNED_TAIL_OVERLAP=$PLANNED_TAIL_OVERLAP" \
  -e "VLLM_TR3_PLANNED_TAIL_MIN_M=$PLANNED_TAIL_MIN_M" \
  -e "VLLM_TR3_PLANNED_TAIL_MAX_M=$PLANNED_TAIL_MAX_M" \
  -e "VLLM_TR3_PLANNED_TAIL_BLOCK_M=$PLANNED_TAIL_BLOCK_M" \
  -e "VLLM_TR3_PLANNED_PREFILL=$PLANNED_PREFILL" \
  -e "VLLM_TR3_PLANNED_PREFILL_MAX_M=$PLANNED_PREFILL_MAX_M" \
  -e "VLLM_TR3_PLANNED_PREFILL_BLOCK_M=$PLANNED_PREFILL_BLOCK_M" \
  -e "VLLM_TR3_PLANNED_PREFILL_PARITY_MAX_M=$PLANNED_PREFILL_PARITY_MAX_M" \
  -e "VLLM_TR3_PLANNED_PREFILL_PARITY_REL_MAX=$PLANNED_PREFILL_PARITY_REL_MAX" \
  -e "VLLM_TR3_PLANNED_PREFILL_PARITY_COS_MIN=$PLANNED_PREFILL_PARITY_COS_MIN" \
  -e "VLLM_TR3_MIXED_PERSISTENT=$MIXED_TRELLIS" \
  -e "VLLM_TR3_MIXED_PERSISTENT_MAX_M=$MIXED_TRELLIS_MAX_M" \
  -e "VLLM_TR3_MIXED_PERSISTENT_PARITY=$MIXED_TRELLIS_PARITY" \
  -e "VLLM_TR3_MIXED_PERSISTENT_PARITY_REL_MAX=$MIXED_TRELLIS_PARITY_REL_MAX" \
  -e "VLLM_TR3_MIXED_PERSISTENT_PARITY_COS_MIN=$MIXED_TRELLIS_PARITY_COS_MIN" \
  -e "VLLM_TR3_MIXED_PERSISTENT_TILE_CONFIG=$MIXED_TRELLIS_TILE_CONFIG" \
  -e VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm52-tr3/glm52_nvfp4_mla_outer_scales.json \
  -v "$MODEL_CACHE:/models:ro" \
  -v "$CACHE:/cache" \
  "$IMAGE" \
  "${SERVER_ARGS[@]}")"

fatal_marker="$(mktemp)"
ready_marker="$(mktemp)"
log_pid=""

stop_log_monitor() {
  if [[ -n "$log_pid" ]] && kill -0 "$log_pid" 2>/dev/null; then
    kill -TERM "$log_pid" 2>/dev/null || true
  fi
  if [[ -n "$log_pid" ]]; then
    wait "$log_pid" 2>/dev/null || true
  fi
  log_pid=""
}

terminate_server() {
  "$ENGINE" stop --time 5 "$MODEL_ID" >/dev/null 2>&1 \
    || "$ENGINE" kill "$MODEL_ID" >/dev/null 2>&1 \
    || true
  stop_log_monitor
  rm -f "$fatal_marker" "$fatal_marker.tmp" \
    "$ready_marker" "$ready_marker.tmp"
}
trap terminate_server EXIT
trap 'exit 143' INT TERM

"$ENGINE" logs --follow "$MODEL_ID" 2>&1 \
  | "$ROOT/scripts/watch-server-logs.py" "$fatal_marker" "$ready_marker" &
log_pid=$!

health_ok() {
  python3 - "$PORT" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

ready=0
health_failures=0
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while true; do
  if [[ -s "$fatal_marker" ]]; then
    printf 'Fatal server error detected; terminating container %s:\n' \
      "$MODEL_ID" >&2
    cat "$fatal_marker" >&2
    terminate_server
    trap - EXIT INT TERM
    exit 1
  fi

  if ! kill -0 "$log_pid" 2>/dev/null; then
    printf 'Server log monitor exited while container %s was running\n' \
      "$MODEL_ID" >&2
    terminate_server
    trap - EXIT INT TERM
    exit 1
  fi

  state="$("$ENGINE" inspect \
    --format '{{.State.Running}} {{.State.ExitCode}}' \
    "$container_id" 2>/dev/null || true)"
  if [[ -z "$state" ]]; then
    printf 'Lost container state for %s\n' "$MODEL_ID" >&2
    terminate_server
    trap - EXIT INT TERM
    exit 1
  fi
  read -r running status <<<"$state"
  if [[ "$running" != "true" ]]; then
    stop_log_monitor
    rm -f "$fatal_marker" "$fatal_marker.tmp" \
      "$ready_marker" "$ready_marker.tmp"
    trap - EXIT INT TERM
    if ((ready == 0)); then
      printf 'Server exited before readiness (status=%d)\n' "$status" >&2
      ((status == 0)) && status=1
    elif ((status != 0)); then
      printf 'Server exited after readiness (status=%d)\n' "$status" >&2
    fi
    exit "$status"
  fi

  if [[ -s "$ready_marker" ]] && health_ok; then
    health_failures=0
    if ((ready == 0)); then
      ready=1
      printf 'Server ready: http://127.0.0.1:%s/health\n' "$PORT"
    fi
  elif ((ready == 1)); then
    health_failures=$((health_failures + 1))
    if ((health_failures >= 3)); then
      printf 'Server health failed %d consecutive checks; terminating %s\n' \
        "$health_failures" "$MODEL_ID" >&2
      terminate_server
      trap - EXIT INT TERM
      exit 1
    fi
  fi

  if ((ready == 0 && SECONDS >= deadline)); then
    printf 'Server readiness timed out after %ss: http://127.0.0.1:%s/health\n' \
      "$STARTUP_TIMEOUT_SECONDS" "$PORT" >&2
    terminate_server
    trap - EXIT INT TERM
    exit 124
  fi
  sleep 2
done
