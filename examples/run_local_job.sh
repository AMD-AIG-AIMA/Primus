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
    bash examples/run_local_job.sh pretrain --config examples/megatron/exp_pretrain.yaml
    bash examples/run_local_job.sh benchmark --mbs-list 1 2 --model llama2_7B
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# ------------------ Default Values ------------------
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ------------------ Log Setup ------------------
LOG_FILE="${LOG_FILE:-${PRIMUS_PATH}/output/log_local_${TIMESTAMP}}"
[[ "$LOG_FILE" != *.container.txt ]] && LOG_FILE="${LOG_FILE}.container.txt"
mkdir -p "$(dirname "$LOG_FILE")"

# ------------------ Environment Variables to Pass ------------------
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

ENV_ARGS=()
for var in "${ENV_VAR_LIST[@]}"; do
    if [[ -n "${!var}" ]]; then
        ENV_ARGS+=("--env" "$var")
    fi
done

# Pass all PRIMUS_* variables
# Again, only include those that have non-empty values
while IFS='=' read -r name _; do
    ENV_ARGS+=("--env" "$name")
done < <(env | grep "^PRIMUS_")

# ------------------ Mount volumes into the container ------------------
# Mount the project root and dataset directory into the container
echo "data path $DATA_PATH"
VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH")
if [[ -n "${DATA_PATH:-}" && -d "${DATA_PATH}" ]]; then
    DATA_PATH=$(realpath "$DATA_PATH")
    VOLUME_ARGS+=(-v "$DATA_PATH":"$DATA_PATH")
fi

export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-0}

# ------------------ Optional Container Cleanup ------------------
if command -v podman >/dev/null 2>&1; then
    DOCKER_CLI="podman"
elif command -v docker >/dev/null 2>&1; then
    DOCKER_CLI="docker"
else
    echo "Neither Docker nor Podman found!" >&2
    exit 1
fi

if [[ "$CLEAN_DOCKER_CONTAINER" == "1" ]]; then
    echo "Node-${NODE_RANK}: Cleaning up existing containers..."
    CONTAINERS="$($DOCKER_CLI ps -aq)"
    if [[ -n "$CONTAINERS" ]]; then
        printf '%s\n' "$CONTAINERS" | xargs -r -n1 "$DOCKER_CLI" rm -f
        echo "Node-${NODE_RANK}: Removed containers: $CONTAINERS"
    else
        echo "Node-${NODE_RANK}: No containers to remove."
    fi
fi

ARGS=("$@")

# ------------------ Print Info ------------------
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
"${DOCKER_CLI}" run --rm \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --device=/dev/infiniband \
    "${ENV_ARGS[@]}" \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && \
        bash examples/run_job.sh \"\$@\" 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
