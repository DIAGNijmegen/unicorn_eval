#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKERFILE_DIR="${SCRIPT_DIR}/.."  # Go up one level to find Dockerfile
DOCKER_IMAGE_TAG="unicorn_eval:latest"

# Override tag if provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi
echo "Building Docker image: $DOCKER_IMAGE_TAG"
docker build "$DOCKERFILE_DIR" \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" 
