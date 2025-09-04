#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat << EOF
Primus Direct Launcher

Usage:
    primus-cli direct [--env KEY=VALUE ...] -- <primus-args>

Description:
    Launch Primus training, benchmarking, or preflight directly on the host (or inside a container),
    with distributed settings controlled via environment variables.

Distributed options: (set before running or rely on primus-env.sh defaults):
    --nnodes <N>              Number of nodes [default: 1]
    --node-rank <RANK>        Current node rank [default: 0]
    --gpus-per-node <N>       Number of GPUs per node [default: 8]
    --master-addr <ADDR>      Master node address [default: localhost]
    --master-port <PORT>      Master node port [default: 1234]

Arguments:
    --env KEY=VALUE               Set environment variable for this run (repeatable)
    --help                        Show this usage message and exit

Examples:
    # Pretrain with a config file (single node)
    primus-cli direct -- train pretrain --config examples/megatron/exp_pretrain.yaml

    # Distributed GEMM benchmark, 2 nodes, each with 8 GPUs
    primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=0 --env MASTER_ADDR=host1 -- \
        benchmark gemm -M 4096 -N 4096 -K 4096

    # Launch on another node as rank 1 (for multi-node setup)
    primus-cli direct --env NNODES=2 --env GPUS_PER_NODE=8 --env NODE_RANK=1 --env MASTER_ADDR=host1 -- \
        benchmark gemm -M 4096 -N 4096 -K 4096
EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# Step 0: Setup log directory and generate log file path
LOG_DIR="${LOG_DIR:-output/logs}"
mkdir -p "${LOG_DIR}"

JOB_ID="${SLURM_JOB_ID:-nojob}"
LOG_FILE="${LOG_DIR}/log_${JOB_ID}_$(date +%Y%m%d_%H%M%S).txt"


# Step 1: Source the environment setup script (centralizes all exports and helper functions).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/primus-env.sh"

# Step 1.5: Parse and export --env KEY=VALUE overrides from cMASTER_PORTMASTER_PORT
NEW_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
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
            if [[ $# -lt 2 ]]; then
                LOG_INFO_RANK0 "ERROR: --env requires KEY=VALUE (got nothing)" >&2
                exit 2
            fi
            if [[ "$2" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*) ]]; then
                export "${2?Missing KEY=VALUE for --env}"
                key="${BASH_REMATCH[1]}"
                LOG_INFO_RANK0 "[ENV OVERRIDE] $key=${!key}"
                shift 2
            else
                LOG_INFO_RANK0 "ERROR: --env requires KEY=VALUE (got '$2')" >&2
                exit 2
            fi
            ;;
        *)
            NEW_ARGS+=("$1")
            shift
            ;;
    esac
done
set "${NEW_ARGS[@]}"

pip install -r requirements.txt

# Step 2: Build torchrun distributed arguments.
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)


# Step 3: Build local rank filter argument.
# Only local rank 0 on first node and last local rank on last node are filtered for special logging.
LAST_NODE=$((NNODES - 1))
FILTERS=()
# Add local rank 0 on the first node
if [ "$NODE_RANK" -eq 0 ]; then
    FILTERS+=(0)
fi

# Add the last local rank on the last node
if [ "$NODE_RANK" -eq "$LAST_NODE" ]; then
    FILTERS+=($((GPUS_PER_NODE - 1)))
fi

# Build filter argument (only if FILTERS is non-empty)
if [ "${#FILTERS[@]}" -gt 0 ]; then
    LOCAL_FILTER=$(IFS=,; echo "${FILTERS[*]}")
    FILTER_ARG=(--local-ranks-filter "$LOCAL_FILTER")
else
    FILTER_ARG=()
fi


# Step 4: Build the final command.
# Note: ${LOCAL_RANKS} removed; only FILTER_ARG is used.
CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${FILTER_ARG[@]}" -- primus/cli/main.py "${@}")
LOG_INFO "Launching distributed training with command: ${CMD[*]} 2>&1 | tee $LOG_FILE"
"${CMD[@]}" | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
# CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} -- primus/cli/main.py $* "

# LOG_INFO "Launching distributed training with command: $CMD 2>&1 | tee $LOG_FILE"

# eval "$CMD" 2>&1 | tee "$LOG_FILE"
# exit_code=${PIPESTATUS[0]}

# Print log based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "torchrun exited with code $exit_code"
else
    LOG_INFO "torchrun finished successfully (code 0)"
fi

exit "$exit_code"
