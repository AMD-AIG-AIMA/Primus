#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ENTRY="$SCRIPT_DIR/primus-run-launch.sh"

# # Default mode: auto (try to infer), or allow explicit --sbatch/--srun/--local
# MODE="auto"
# SLURM_FLAGS=()

# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --sbatch)
#             MODE="sbatch";
#             shift;;
#         --srun)
#             MODE="srun";
#             shift;;
#         --output|--error|-p|-A|-q|-t|-J|-N)
#             SLURM_FLAGS+=("$1" "$2");
#             shift 2;;
#         --nodes|--nodelist|--partition|--reservation|--qos|--time|--job-name)
#             SLURM_FLAGS+=("$1" "$2");
#             shift 2;;
#         --output=*|--error=*|-p=*|-A=*|-q=*|-t=*|-J=*|-N=*)
#             SLURM_FLAGS+=("$1");
#             shift;;
#         --nodes=*|--nodelist=*|--partition=*|--reservation=*|--qos=*|--time=*|--job-name=*)
#             SLURM_FLAGS+=("$1");
#             shift;;
#         --)
#             shift;
#             break;;
#         *)
#             break;;
#     esac
# done

# [[ $# -gt 0 ]] || { echo "Usage: primus-run-slurm.sh [--sbatch|--srun|--local] [slurm-flags] -- <primus args>"; exit 2; }

# # --------- Mode auto-detect (if not set explicitly) ----------
# if [[ "$MODE" == "auto" ]]; then
#     if [[ -n "${SLURM_JOB_ID:-}" ]]; then
#         # If inside a Slurm allocation (eg, srun), treat as local launch
#         MODE="local"
#     else
#         MODE="local"
#     fi
# fi

# # --------- Dispatch to proper launcher ----------
# case "$MODE" in
#     sbatch)
#         exec sbatch "${SLURM_FLAGS[@]}" "$ENTRY" -- "$@"
#         ;;
#     srun)
#         exec srun "${SLURM_FLAGS[@]}" "$ENTRY" -- "$@"
#         ;;
#     local)
#         exec "$ENTRY" -- "$@"
#         ;;
#     *)
#         echo "Unknown mode $MODE"; exit 1
#         ;;
# esac


# --------- Cluster & distributed variables setup ---------

# Determine the master node address for distributed training.
# If SLURM_NODELIST is set (running in a Slurm job), pick the first node as MASTER_ADDR.
# Otherwise, use the value of MASTER_ADDR if provided, or fallback to localhost.
if [[ -z "${MASTER_ADDR:-}" && -n "${SLURM_NODELIST:-}" ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
else
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi

# Port for distributed communication; can be overridden, default is 1234.
MASTER_PORT="${MASTER_PORT:-1234}"

# Number of nodes participating in the job.
# Prefer SLURM_NNODES, fallback to SLURM_JOB_NUM_NODES, or default to 1.
if [[ -n "${SLURM_NNODES:-}" ]]; then
    NNODES="${SLURM_NNODES}"
elif [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
    NNODES="${SLURM_JOB_NUM_NODES}"
else
    NNODES="${NNODES:-1}"
fi

# The rank (index) of the current node in the distributed setup.
# Prefer SLURM_NODEID, fallback to SLURM_PROCID, or default to 0.
if [[ -n "${SLURM_NODEID:-}" ]]; then
    NODE_RANK="${SLURM_NODEID}"
elif [[ -n "${SLURM_PROCID:-}" ]]; then
    NODE_RANK="${SLURM_PROCID}"
else
    NODE_RANK="${NODE_RANK:-0}"
fi

# Number of GPUs per node; default is 8 if not specified.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Optional: Export variables for downstream scripts/processes
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE

# Print current distributed setup for logging/debugging
echo "[primus-run-slurm] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES NODE_RANK=$NODE_RANK GPUS_PER_NODE=$GPUS_PER_NODE"
