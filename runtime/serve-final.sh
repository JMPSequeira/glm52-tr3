#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

/opt/venv/bin/python /opt/glm52-tr3/patch-v19-fused-tr3.py
launcher_source=/usr/local/bin/serve-glm52-v16.sh
launcher="$launcher_source"
draft_sample_method="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}"
mtp_batch_schedule="${MTP_BATCH_SCHEDULE:-}"
case "$draft_sample_method" in
  greedy|probabilistic) ;;
  *)
    printf 'MTP_DRAFT_SAMPLE_METHOD must be greedy or probabilistic, got %q\n' \
      "$draft_sample_method" >&2
    exit 2
    ;;
esac

if [[ "$draft_sample_method" != "probabilistic" || -n "$mtp_batch_schedule" ]]; then
  launcher=/tmp/serve-glm52-v16.sh
  /opt/venv/bin/python - "$launcher" "$draft_sample_method" \
    "$mtp_batch_schedule" <<'PY'
import json
from pathlib import Path
import sys

destination, draft_sample_method, schedule_text = sys.argv[1:]
source = Path("/usr/local/bin/serve-glm52-v16.sh").read_text()
sampler_anchor = '"draft_sample_method":"probabilistic"'
if source.count(sampler_anchor) != 1:
    raise RuntimeError("unexpected MTP draft sampler launcher anchor")
replacement = f'"draft_sample_method":"{draft_sample_method}"'
source = source.replace(sampler_anchor, replacement)

if schedule_text:
    schedule = json.loads(schedule_text)
    if (
        not isinstance(schedule, list)
        or not schedule
        or any(
            not isinstance(entry, list)
            or len(entry) != 3
            or any(not isinstance(value, int) or value < 1 for value in entry)
            for entry in schedule
        )
    ):
        raise ValueError(
            "MTP_BATCH_SCHEDULE must be a non-empty JSON list of "
            "[start_batch, end_batch, depth] integer triples"
        )
    schedule_json = json.dumps(schedule, separators=(",", ":"))
    schedule_anchor = f'{replacement}}}'
    if source.count(schedule_anchor) != 1:
        raise RuntimeError("unexpected MTP batch schedule launcher anchor")
    source = source.replace(
        schedule_anchor,
        f'{replacement},"num_speculative_tokens_per_batch_size":'
        f'{schedule_json}}}',
    )

Path(destination).write_text(source)
Path(destination).chmod(0o755)
PY
fi

capture_args=()
if [[ -n "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
  IFS=',' read -r -a capture_sizes <<< "$CUDAGRAPH_CAPTURE_SIZES"
  for size in "${capture_sizes[@]}"; do
    if [[ ! "$size" =~ ^[1-9][0-9]*$ ]]; then
      printf 'CUDAGRAPH_CAPTURE_SIZES must contain positive comma-separated integers, got %q\n' \
        "$CUDAGRAPH_CAPTURE_SIZES" >&2
      exit 2
    fi
  done
  capture_args=(--cudagraph-capture-sizes "${capture_sizes[@]}")
fi
for argument in "$@"; do
  if [[ "$argument" == "--cudagraph-capture-sizes" ]]; then
    capture_args=()
    break
  fi
done

exec "$launcher" "${capture_args[@]}" "$@"
