set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $# -lt 3 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <INPUT_FOLDER> <GROUND_TRUTH_FOLDER> <OUTPUT_FOLDER> [DOCKER_IMAGE_TAG]"
    exit 1
fi

INPUT_DIR="$1"
GROUND_TRUTH_DIR="$2"
OUTPUT_DIR="$3"
DOCKER_IMAGE_TAG="${4:-unicorn_eval}"


echo "Using DOCKER_IMAGE_TAG: $DOCKER_IMAGE_TAG"
echo "Input:        $INPUT_DIR"
echo "Ground truth: $GROUND_TRUTH_DIR"
echo "Output:       $OUTPUT_DIR"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

echo "=+= (Re)build the container"

source "${SCRIPT_DIR}/do_build.sh" "$DOCKER_IMAGE_TAG"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"
}

if [ -d "$OUTPUT_DIR" ]; then
  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwx "$OUTPUT_DIR"
fi

trap cleanup EXIT

echo "=+= Running evaluation container"
docker run --rm \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$GROUND_TRUTH_DIR":/opt/ml/input/data/ground_truth:ro \
  --volume "$OUTPUT_DIR":/output \
  "$DOCKER_IMAGE_TAG"

echo "=+= Wrote results to ${OUTPUT_DIR}"