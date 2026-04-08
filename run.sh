#!/usr/bin/env bash
# ============================================================================
# run.sh — Build and run the ncast MPI+CUDA benchmark in Docker
# ============================================================================
#
# Prerequisites:
#   - Docker with BuildKit
#   - NVIDIA Container Toolkit  (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
#   - At least 2 NVIDIA GPUs (or 1 GPU — the benchmark will still run, ranks share it)
#
# Usage:
#   ./run.sh build          # build the Docker image
#   ./run.sh run   [ARGS]   # run the benchmark (pass extra mpirun args)
#   ./run.sh shell          # interactive shell inside the container
#   ./run.sh cbuild         # build via Docker Compose
#   ./run.sh crun  [ARGS]   # run via Docker Compose
#   ./run.sh cshell         # interactive shell via Docker Compose
#   ./run.sh all            # build + run
# ============================================================================
set -euo pipefail

IMAGE_NAME="${NCAST_IMAGE:-ncast-bench}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yaml"

# ---------------------------------------------------------------------------
cmd_build() {
    echo "==> Building Docker image: ${IMAGE_NAME}"
    DOCKER_BUILDKIT=1 docker build \
        -t "${IMAGE_NAME}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"
}

# ---------------------------------------------------------------------------
cmd_run() {
    echo "==> Running benchmark (results → test-output/)"
    docker run --rm --gpus all --ipc=host \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "${SCRIPT_DIR}/test-input:/workspace/test-input:ro" \
        -v "${SCRIPT_DIR}/test-output:/workspace/test-output" \
        "${IMAGE_NAME}" "$@" \
        2>&1 | tee "${SCRIPT_DIR}/test-output/benchmark-$(date +%Y%m%d-%H%M%S).log"
}

# ---------------------------------------------------------------------------
cmd_shell() {
    echo "==> Opening interactive shell"
    docker run --rm -it --gpus all --ipc=host \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "${SCRIPT_DIR}/test-input:/workspace/test-input:ro" \
        -v "${SCRIPT_DIR}/test-output:/workspace/test-output" \
        --entrypoint /bin/bash \
        "${IMAGE_NAME}"
}

# ---------------------------------------------------------------------------
cmd_cbuild() {
    echo "==> Building Docker Compose service"
    docker compose -f "${COMPOSE_FILE}" build
}

# ---------------------------------------------------------------------------
cmd_crun() {
    echo "==> Running benchmark via Docker Compose (results → test-output/)"
    docker compose -f "${COMPOSE_FILE}" run --rm ncast "$@" \
        2>&1 | tee "${SCRIPT_DIR}/test-output/benchmark-$(date +%Y%m%d-%H%M%S).log"
}

# ---------------------------------------------------------------------------
cmd_cshell() {
    echo "==> Opening interactive shell via Docker Compose"
    docker compose -f "${COMPOSE_FILE}" run --rm ncast /bin/bash
}

# ---------------------------------------------------------------------------
case "${1:-all}" in
    build)  cmd_build ;;
    run)    shift; cmd_run "$@" ;;
    shell)  cmd_shell ;;
    cbuild) cmd_cbuild ;;
    crun)   shift; cmd_crun "$@" ;;
    cshell) cmd_cshell ;;
    all)    cmd_build; cmd_run ;;
    *)
        echo "Usage: $0 {build|run|shell|cbuild|crun|cshell|all}"
        exit 1
        ;;
esac
