#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# ----------- Cluster/Node environment setup ------------

# Get current script dir for resolving downstream scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------- Distributed environment variables ---------
# Pick master node address from SLURM_NODELIST, or fallback
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
else
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
MASTER_PORT="${MASTER_PORT:-1234}"

# Get all node hostnames (sorted, as needed)
readarray -t NODE_ARRAY < <(scontrol show hostnames "$SLURM_NODELIST")
# (Optional: sort by IP if needed, e.g., for deterministic rank mapping)
# Uncomment if you need IP sort
# readarray -t NODE_ARRAY < <(
#     for node in $(scontrol show hostnames "$SLURM_NODELIST"); do
#         getent hosts "$node" | awk '{print $1, $2}'
#     done | sort -k1,1n | awk '{print $2}'
# )

NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-${NNODES:-1}}}"
NODE_RANK="${SLURM_NODEID:-${SLURM_PROCID:-${NODE_RANK:-0}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

echo "[primus-run-slurm] MASTER_ADDR=$MASTER_ADDR"
echo "[primus-run-slurm] MASTER_PORT=$MASTER_PORT"
echo "[primus-run-slurm] NNODES=$NNODES"
echo "[primus-run-slurm] NODE_RANK=$NODE_RANK"
echo "[primus-run-slurm] GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[primus-run-slurm] NODE_LIST: ${NODE_ARRAY[*]}"

# ------------- Dispatch based on mode ---------------

# Default: 'container' mode, unless user overrides
MODE="container"
if [[ $# -gt 0 && "$1" =~ ^(container|native|host)$ ]]; then
    MODE="$1"
    shift
fi

case "$MODE" in
    # container)
    #     # Call container launcher script with all remaining args
    #     exec bash "$SCRIPT_DIR/primus-cli-container.sh" "$@"
    #     ;;
    # native|host)
    #     # Directly run on host with all remaining args
    #     exec bash "$SCRIPT_DIR/primus-cli-direct.sh" "$@"
    #     ;;
    # *)
    #     echo "Unknown mode: $MODE. Use 'container' or 'native'."
    #     exit 2
    #     ;;

    container)
        script_path="$SCRIPT_DIR/primus-cli-container.sh"
        echo "[primus-cli-slurm-entry] Executing: bash $script_path $*"
        exec bash "$script_path" "$@"
        ;;
    native|host)
        script_path="$SCRIPT_DIR/primus-cli-direct.sh"
        echo "[primus-cli-slurm-entry] Executing: bash $script_path $*"
        exec bash "$script_path" "$@"
        ;;
    *)
        echo "Unknown mode: $MODE. Use 'container' or 'native'."
        exit 2
        ;;
esac
