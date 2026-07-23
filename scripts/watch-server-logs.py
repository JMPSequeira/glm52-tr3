#!/usr/bin/env python3
"""Stream server logs and atomically flag unrecoverable engine failures."""

from pathlib import Path
import re
import sys

_FATAL = re.compile(
    r"Traceback \(most recent call last\)|"
    r"RuntimeError:|AssertionError:|OutOfMemoryError|CUDA error:|"
    r"EngineCore failed to start|Engine core initialization failed|"
    r"EngineDeadError|Worker failed with error|WorkerProc initialization failed|"
    r"Server exited before readiness"
)
_READY = re.compile(r"Application startup complete\.")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            f"usage: {sys.argv[0]} FATAL_MARKER [READY_MARKER]"
        )
    marker = Path(sys.argv[1])
    ready_marker = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    flagged = False
    ready_flagged = False
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        if not flagged and _FATAL.search(line):
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(line)
            temporary.replace(marker)
            flagged = True
        if (
            ready_marker is not None
            and not ready_flagged
            and _READY.search(line)
        ):
            temporary = ready_marker.with_suffix(".tmp")
            temporary.write_text(line)
            temporary.replace(ready_marker)
            ready_flagged = True


if __name__ == "__main__":
    main()
