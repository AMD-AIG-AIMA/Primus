#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat <<EOF
Usage: bash primus-run-container.sh [OPTIONS] -- [SCRIPT_ARGS...]

Launch a Primus task (train / benchmark / preflight / etc.) in a Docker/Podman container.

Options:
    --image <DOCKER_IMAGE>    Docker image to use [default: \$DOCKER_IMAGE or rocm/megatron-lm:v25.5_py310]
    --mount <HOST[:CONTAINER]>
        Mount a host directory into the container.
        - If only HOST is given, mounts to same path inside container.
        - If HOST:CONTAINER is given, mounts host directory to container path.
        (repeatable; for data, output, cache, etc.)

    --log-file <PATH>         Path to write container logs (default: ./output/log_container_TIMESTAMP.container.txt)
    --clean                   Remove all containers before launch
    --env <KEY=VALUE>         Set environment variable in container (repeatable)
    --help                    Show this message and exit

Examples:
    bash primus-run-container.sh --mount /mnt/data -- train --config /mnt/data/exp.yaml --data-path /mnt/data
    bash primus-run-container.sh --mount /mnt/profile_out -- benchmark gemm --output /mnt/profile_out/result.txt
    bash primus-run-container.sh \\
        --mount /mnt/data \\
        --mount /mnt/output \\
        -- train --config /mnt/data/exp.yaml --out-dir /mnt/output
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

echo "[primus-cli-container] Received args: $*"

# Default Values
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)

# Cluster-related defaults
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Parse CLI options
DOCKER_IMAGE=""
LOG_FILE=""
CLEAN_DOCKER_CONTAINER=0
ENV_OVERRIDES=()
MOUNTS=()
POSITIONAL_ARGS=()

handled=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            DOCKER_IMAGE="$2"; shift 2;;
        --mount)
            MOUNTS+=("$2"); shift 2;;
        --log-file)
            LOG_FILE="$2"; shift 2;;
        --clean)
            CLEAN_DOCKER_CONTAINER=1; shift;;
            --master-addr)
            MASTER_ADDR="$2"; shift 2;;
        --master-port)
            MASTER_PORT="$2"; shift 2;;
        --nnodes)
            NNODES="$2"; shift 2;;
        --node-rank)
            NODE_RANK="$2"; shift 2;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"; shift 2;;
        --env)
            if [[ "$2" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
                ENV_OVERRIDES+=("$2")
                shift 2
            else
                echo "[ERROR] --env requires KEY=VALUE format." >&2
                exit 1
            fi
            ;;
        --help|-h)
            print_usage; exit 0;;
        --)
            shift; POSITIONAL_ARGS+=("$@"); break;;
        *)
            # echo "[ERROR] Unknown option for container: $1" >&2
            # exit 1
            if [[ $handled -eq 0 ]]; then
                POSITIONAL_ARGS+=("$@")
                break
            else
                echo "[primus-cli-container] ERROR: Unknown option: $1" >&2
                exit 1
            fi
            ;;
    esac
    handled=1
done

# Defaults (fallback)
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.5_py310"}
LOG_FILE="${LOG_FILE:-${PRIMUS_PATH}/output/log_container_${TIMESTAMP}.container.txt}"
mkdir -p "$(dirname "$LOG_FILE")"

# ------------- Pass key cluster envs & all PRIMUS_* vars -------------
declare -A ENV_SEEN
ENV_ARGS=()
for env_kv in "${ENV_OVERRIDES[@]}"; do
    ENV_ARGS+=("--env" "$env_kv")
    key="${env_kv%%=*}"
    ENV_SEEN["$key"]=1
done
for var in MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE; do
    if [[ -z "${ENV_SEEN[$var]:-}" ]]; then
        ENV_ARGS+=("--env" "$var")
        ENV_SEEN["$var"]=1
    fi
done

# ----------------- Volume Mounts -----------------
# Mount the project root and dataset directory into the container
VOLUME_ARGS=(-v "$PRIMUS_PATH":"$PRIMUS_PATH")
for mnt in "${MOUNTS[@]}"; do
    # Parse --mount argument (HOST[:CONTAINER])
    if [[ "$mnt" == *:* ]]; then
        host_path="${mnt%%:*}"
        container_path="${mnt#*:}"
        # Check that the host path exists and is a directory
        if [[ ! -d "$host_path" ]]; then
            echo "[ERROR] --mount $host_path does not exist or is not a directory. Please check your path." >&2
            exit 1
        fi
        VOLUME_ARGS+=(-v "$(realpath "$host_path")":"$container_path")
    else
        # Mount to same path inside container
        if [[ ! -d "$mnt" ]]; then
            echo "[ERROR] --mount $mnt does not exist or is not a directory. Please check your path." >&2
            exit 1
        fi
        abs_path="$(realpath "$mnt")"
        VOLUME_ARGS+=(-v "$abs_path":"$abs_path")
    fi
done


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
    echo "[NODE-$NODE_RANK($HOSTNAME)]  MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  NNODES: $NNODES"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  GPUS_PER_NODE: $GPUS_PER_NODE"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  HOSTNAME: $HOSTNAME"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  LOG_FILE: ${LOG_FILE}"
    echo "[NODE-$NODE_RANK($HOSTNAME)]  VOLUME_ARGS:"
    for ((i = 0; i < ${#VOLUME_ARGS[@]}; i+=2)); do
        echo "[NODE-${NODE_RANK}(${HOSTNAME})]      ${VOLUME_ARGS[i]} ${VOLUME_ARGS[i+1]}"
    done
    echo "[NODE-${NODE_RANK}(${HOSTNAME})]  ENV_ARGS:"
    for ((i = 0; i < ${#ENV_ARGS[@]}; i+=2)); do
        # env_key="${ENV_ARGS[i+1]}"
        # env_value="${!env_key}"
        # echo "[NODE-${NODE_RANK}(${HOSTNAME})]      ${ENV_ARGS[i]} ${env_key} ${env_value}"
        echo "[NODE-${NODE_RANK}(${HOSTNAME})]      ${ENV_ARGS[i]} ${ENV_ARGS[i+1]}"
    done
    echo "[NODE-${NODE_RANK}(${HOSTNAME})]  ARGS: ${ARGS[*]}"
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
    --env MASTER_ADDR \
    --env MASTER_PORT \
    --env NNODES \
    --env NODE_RANK \
    --env GPUS_PER_NODE \
    "${ENV_ARGS[@]}" \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]:  container started at $(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && \
        bash bin/primus-cli-entrypoint.sh \"\$@\" 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]:  container finished at $(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"
