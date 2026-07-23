#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

/opt/venv/bin/python /opt/glm52-tr3/patch-v19-fused-tr3.py
launcher=/usr/local/bin/serve-glm52-v16.sh
draft_sample_method="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}"
case "$draft_sample_method" in
  greedy|probabilistic) ;;
  *)
    printf 'MTP_DRAFT_SAMPLE_METHOD must be greedy or probabilistic, got %q\n' \
      "$draft_sample_method" >&2
    exit 2
    ;;
esac

if [[ "$draft_sample_method" != "probabilistic" ]]; then
  launcher=/tmp/serve-glm52-v16.sh
  /opt/venv/bin/python - "$launcher" "$draft_sample_method" <<'PY'
from pathlib import Path
import sys

source = Path("/usr/local/bin/serve-glm52-v16.sh").read_text()
old = '"draft_sample_method":"probabilistic"'
new = f'"draft_sample_method":"{sys.argv[2]}"'
if source.count(old) != 1:
    raise RuntimeError("unexpected MTP draft sampler launcher anchor")
Path(sys.argv[1]).write_text(source.replace(old, new))
Path(sys.argv[1]).chmod(0o755)
PY
fi

exec "$launcher" "$@"
