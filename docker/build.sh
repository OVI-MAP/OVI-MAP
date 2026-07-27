#!/bin/bash
# =============================================================================
# Build the OVI-MAP Docker image.
# =============================================================================
set -e

IMAGE_NAME="${IMAGE_NAME:-ovi-map}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

cd "$(dirname "$0")/.."

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f docker/Dockerfile .
