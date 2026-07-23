#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_ID="${MODEL_ID:-glm52-tr3-no-mtp}"
export MTP=0
export MTP_DRAFT_SAMPLE_METHOD=probabilistic
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-5120}"
export GRAPH="${GRAPH:-16}"
export PLANNED_TAIL=1
export PLANNED_TAIL_OVERLAP=1
export PLANNED_TAIL_MIN_M=1
export PLANNED_TAIL_MAX_M=32
export PLANNED_TAIL_BLOCK_M=8
export PLANNED_PREFILL=1
export PLANNED_PREFILL_MAX_M=5120
export PLANNED_PREFILL_BLOCK_M=64
export TRELLIS256=1
export TRELLIS256_MIN_M=1
export TRELLIS256_MAX_M=16
export MIXED_TRELLIS=1
export MIXED_TRELLIS_MAX_M=4
export MIXED_TRELLIS_TILE_CONFIG=128,128,128,128
export CKV_GATHER=1
export CKV_GATHER_MIN_TOKENS=512
export CKV_GATHER_MAX_TOKENS=65536
export ABSORB_BMM=0
export SPEC_EXTEND_AS_DECODE=auto
export DISABLE_SHARED_EXPERTS_STREAM=0

exec "$ROOT/scripts/launch-common.sh" "$@"
