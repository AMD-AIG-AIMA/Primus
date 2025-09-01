#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat << EOF
Usage: bash $(basename "$0") [--help]

Environment variables (set before running or rely on primus-env.sh defaults):

    NNODES=1                      # Number of nodes (default: 1)
    NODE_RANK=0                   # Current node rank (default: 0)
    GPUS_PER_NODE=8               # Number of GPUs per node (default: 8)
    MASTER_ADDR=localhost         # Master node address (default: localhost)
    MASTER_PORT=1234              # Master node port (default: 1234)

Examples:
    # Pretrain with config
    bash examples/scripts/primus-run-direct.sh -- train pretrain --config examples/megatron/exp_pretrain.yaml

    # Benchmark GEMM
    bash examples/scripts/primus-run-direct.sh -- benchmark gemm -M 4096 -N 4096 -K 4096
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
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/primus-env.sh"


# Step 1.5: Parse and export --env KEY=VALUE overrides from command line
NEW_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            if [[ $# -lt 2 ]]; then
                LOG_INFO_RANK0 "ERROR: --env requires KEY=VALUE (got nothing)" >&2
                exit 2
            fi
            if [[ "$2" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*) ]]; then
                export "${2?Missing KEY=VALUE for --env}"
                key="${BASH_REMATCH[1]}"
                # LOG_INFO_RANK0 "[ENV OVERRIDE] $key=${!key}"
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
CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} -- primus/cli/main.py $* "

LOG_INFO "Launching distributed training with command: $CMD 2>&1 | tee $LOG_FILE"

exit 0
eval "$CMD" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

# Print log based on exit code
if [[ $exit_code -ge 128 ]]; then
    LOG_ERROR "torchrun crashed due to signal $((exit_code - 128))"
elif [[ $exit_code -ne 0 ]]; then
    LOG_ERROR "torchrun exited with code $exit_code"
else
    LOG_INFO "torchrun finished successfully (code 0)"
fi

exit "$exit_code"
