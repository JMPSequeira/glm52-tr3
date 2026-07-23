#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${CONTAINER_ENGINE:-podman}"
IMAGE="${IMAGE:-glm52-tr3:runtime-v1}"

exec "$ENGINE" build \
  --file "$ROOT/docker/Containerfile" \
  --tag "$IMAGE" \
  "$ROOT"
