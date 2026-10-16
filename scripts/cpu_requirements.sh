#!/usr/bin/env bash
# Print dependency pins from uv.lock for a CPU-only install.
#
#   scripts/cpu_requirements.sh            every locked package (incl. dev tools) except torch + CUDA
#   scripts/cpu_requirements.sh runtime    the same without the dev group (for the Docker image)
#   scripts/cpu_requirements.sh torch      the locked torch pin, to install from the PyTorch CPU index
#
# uv.lock records PyPI's torch, which pulls about 3 GB of NVIDIA libraries. ScholarTrack's
# networks are tiny and run on CPU, so CI and the Docker image install the CPU build of the
# *same* torch version instead; every other package keeps its exact locked version.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-all}"
groups="--all-groups"
[[ "$mode" == "runtime" ]] && groups="--no-dev"

pins=$(uv export --frozen --no-hashes --no-emit-project --no-annotate --no-header $groups)

if [[ "$mode" == "torch" ]]; then
  grep -E '^torch==' <<<"$pins" | cut -d' ' -f1
else
  grep -vE '^(torch|triton|nvidia-|cuda-)' <<<"$pins"
fi
