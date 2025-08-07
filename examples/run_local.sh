#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ------------------ Usage Help ------------------

print_usage() {
cat <<EOF
Usage: bash run_local.sh [SCRIPT_PATH] [SCRIPT_ARGS...]

This script launches a Primus task (e.g., pretraining, fine-tuning, preflight) inside a Docker/Podman container.

Positional Arguments:
    SCRIPT_PATH      Path to the script to run inside the container (relative to \$PRIMUS_PATH)
    SCRIPT_ARGS      Optional arguments passed to the script inside the container

Environment Variables:
    DOCKER_IMAGE     Docker image to use [Default: docker.io/rocm/megatron-lm:v25.5_py310]
    MASTER_ADDR      Master node IP or hostname [Default: localhost]
    MASTER_PORT      Master node port [Default: 1234]
    NNODES           Total number of nodes [Default: 1]
    NODE_RANK        Rank of this node [Default: 0]
    GPUS_PER_NODE    GPUs per node [Default: 8]
    PRIMUS_*         Any environment variable prefixed with PRIMUS_ will be passed into the container.
    CLEAN_DOCKER_CONTAINER  Whether to remove all containers before start [0/1]

Example:
    EXP=examples/megatron/exp_pretrain.yaml DATA_PATH=/mnt/data \
    bash run_local.sh examples/run_pretrain.sh --test 1
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# Default docker image
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}

# Project root
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

# Dataset directory
DATA_PATH=${DATA_PATH:-"$(pwd)/data"}

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
HOSTNAME=$(hostname)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "${LOG_FILE}" ]]; then
    export LOG_FILE="${PRIMUS_PATH}/output/log_local_${TIMESTAMP}"
else
    # Append `.container.txt` only if LOG_FILE is explicitly set
    export LOG_FILE="${LOG_FILE}.container.txt"
fi
mkdir -p "$(dirname "$LOG_FILE")"

# ------------------ Collect environment variables to pass into the container ------------------
ENV_ARGS=()

# Explicitly listed variables.
# Only include those with non-empty values to avoid overriding default settings
# or polluting the container environment with empty variables.
ENV_VAR_LIST=(
    MASTER_ADDR
    MASTER_PORT
    NNODES
    NODE_RANK
    GPUS_PER_NODE
    EXP
    HF_TOKEN
    LOG_FILE
    DATA_PATH
    TRAIN_LOG
    HSA_NO_SCRATCH_RECLAIM
    NVTE_CK_USES_BWD_V3
    NCCL_IB_HCA
    GLOO_SOCKET_IFNAME
    NCCL_SOCKET_IFNAME
    REBUILD_BNXT
    PATH_TO_BNXT_TAR_PACKAGE
)
for var in "${ENV_VAR_LIST[@]}"; do
    if [[ -n "${!var}" ]]; then
        ENV_ARGS+=("--env" "$var")
    fi
done

# Automatically include all environment variables that start with PRIMUS_
# Again, only include those that have non-empty values
while IFS='=' read -r name _; do
    ENV_ARGS+=("--env" "$name")
done < <(env | grep "^PRIMUS_")

# ------------------ Mount volumes into the container ------------------
# Mount the project root and dataset directory into the container
VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH" -v "$DATA_PATH":"$DATA_PATH")

# Optionally mount the BNXT tar package if the path is set and the file exists
# This avoids mounting non-existent paths that may cause container startup failures
if [[ -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    abs_bnxt_path=$(realpath "$PATH_TO_BNXT_TAR_PACKAGE")
    VOLUME_ARGS+=(-v "$abs_bnxt_path":"$abs_bnxt_path")
fi

export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-0}
# ------------------ Docker/Podman Proxy Function ------------------
docker_podman_proxy() {
    if command -v podman &>/dev/null; then
        podman "$@"
    elif command -v docker &>/dev/null; then
        docker "$@"
    else
        echo "Neither Docker nor Podman found!" >&2
        return 1
    fi
}

if [[ "$CLEAN_DOCKER_CONTAINER" == "1" ]]; then
    echo "Node-${NODE_RANK}: Cleaning up existing containers..."
    CONTAINERS=$(docker_podman_proxy ps -aq)
    if [[ -n "$CONTAINERS" ]]; then
        echo "$CONTAINERS" | xargs -r -I{} sh -c 'docker_podman_proxy rm -f "$@"' _ {}
        echo "Node-${NODE_RANK}: Removed containers: $CONTAINERS"
    else
        echo "Node-${NODE_RANK}: No containers to remove."
    fi
fi

ARGS=("$@")

if [ "$NODE_RANK" = "0" ]; then
    echo "[NODE-$NODE_RANK($HOSTNAME)] ========== Cluster Info =========="
    echo "[NODE-$NODE_RANK($HOSTNAME)] MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK($HOSTNAME)] MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK($HOSTNAME)] NNODES: $NNODES"
    echo "[NODE-$NODE_RANK($HOSTNAME)] GPUS_PER_NODE: $GPUS_PER_NODE"
    echo "[NODE-$NODE_RANK($HOSTNAME)] HOSTNAME: $HOSTNAME"
    echo "[NODE-$NODE_RANK($HOSTNAME)] LOG_FILE: ${LOG_FILE}"
    echo "[NODE-$NODE_RANK($HOSTNAME)] VOLUME_ARGS:"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "[NODE-${NODE_RANK}(${HOSTNAME})]     ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
    done
    echo "[NODE-${NODE_RANK}(${HOSTNAME})] ENV_ARGS:"
    for ((i = 0; i < ${#ENV_ARGS[@]}; i+=2)); do
        env_key="${ENV_ARGS[i+1]}"
        env_value="${!env_key}"
        echo "[NODE-${NODE_RANK}(${HOSTNAME})]     ${ENV_ARGS[i]} ${env_key} ${env_value}"
    done
    echo "[NODE-${NODE_RANK}(${HOSTNAME})] ARGS: ${ARGS[*]}"
    echo
fi


# ------------------ Launch Training Container ------------------
docker_podman_proxy run --rm \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged --device=/dev/infiniband \
    "${ENV_ARGS[@]}" \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && \
        bash examples/run_pretrain.sh \"\$@\" 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
