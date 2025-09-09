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

Distributed options (export before running or set with --env KEY=VALUE):
    --env NNODES=<N>             Number of nodes [default: 1]
    --env NODE_RANK=<RANK>       Current node rank [default: 0]
    --env GPUS_PER_NODE=<N>      Number of GPUs per node [default: 8]
    --env MASTER_ADDR=<ADDR>     Master node address [default: localhost]
    --env MASTER_PORT=<PORT>     Master node port [default: 1234]

You can set these variables by exporting them in your shell, e.g.:
    export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=host1

Examples:
    # Pretrain with a config file (single node)
    primus-cli direct -- train pretrain --config examples/megatron/exp_pretrain.yaml

    # Benchmark GEMM (single node)
    primus-cli direct -- benchmark gemm

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

pip install -qq -r requirements.txt

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


if [[ "$1" == "train" && "$2" == "pretrain" ]]; then
    shift 2

    EXTRA_ARGS=""
    LOG_INFO "[Primus Entrypoint] Detected 'train' command, running data preparation..."

    PRIMUS_PATCH_ARGS_FILE=$(mktemp /tmp/primus_patch_args.XXXXXX.yaml)
    trap 'rm -f "$PRIMUS_PATCH_ARGS_FILE"' EXIT

    SCRIPT="examples/scripts/prepare_experiment.py"

      # Run the prepare_experiment.py script with required and optional arguments
    if ! python3 "$SCRIPT" --patch_args "$PRIMUS_PATCH_ARGS_FILE" "$*" ; then
        LOG_ERROR "$SCRIPT failed, aborting."
        exit 1
    fi

    if [[ -f "$PRIMUS_PATCH_ARGS_FILE" ]]; then
        LOG_INFO_RANK0 "Loading patch args from $PRIMUS_PATCH_ARGS_FILE"
        source_yaml_args() {
            local file=$1
            local key=$2
            grep -E "^${key}:" "$file" | cut -d':' -f2- | xargs
        }

        EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" train_args)

        if [[ -n "$EXTRA_ARGS" ]]; then
            LOG_INFO_RANK0 "Patched TRAIN args: $EXTRA_ARGS"
        fi

    else
        LOG_INFO_RANK0 "No patch args file found at $PRIMUS_PATCH_ARGS_FILE, skipping patch args."
    fi

    LOG_INFO "[Primus Entrypoint] Data preparation complete, proceeding to training..."
    set -- train pretrain "$@" "$EXTRA_ARGS"
fi


# Step 4: Build the final command.
CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} -- primus/cli/main.py $* "
LOG_INFO "Launching distributed training with command: $CMD 2>&1 | tee $LOG_FILE"
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
