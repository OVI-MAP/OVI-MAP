#!/bin/bash
# =============================================================================
# Run the OVI-MAP Docker container.
# =============================================================================
# Usage:
#   bash docker/run.sh                              # interactive shell
#   bash docker/run.sh bash scripts/run_pipeline_async.sh office0
#   bash docker/run.sh python -c "import torch; print(torch.cuda.is_available())"
#
# Data directories:
#   The repo root is mounted at /workspace.
#   Override DATA_DIR to mount your dataset location (default: ~/Data).
# =============================================================================
set -e

IMAGE_NAME="${IMAGE_NAME:-ovi-map}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ovi-map-dev}"
DATA_DIR="${DATA_DIR:-$HOME/Data}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Create data directory on host if it doesn't exist
mkdir -p "${DATA_DIR}"

docker run -it --rm \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    --net=host \
    -v "${REPO_ROOT}:/workspace" \
    -v "${DATA_DIR}:/data" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY="${DISPLAY:-}" \
    -e "DATA=/data" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$@"
