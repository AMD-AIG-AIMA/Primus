#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=primus-job
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
##SBATCH --nodes=2
##SBATCH --partition=****
##SBATCH --reservation=****

###############################################################################
## Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
##
## See LICENSE for license information.
###############################################################################


###############################################################################
# Usage Guide:
#
# This script supports running distributed training in 3 Slurm modes:
#
#   1. [Bash + srun] Run directly via:
#        NNODES=2 CPUS_PER_TASK=256 PARTITION=xxx RESERVATION=xxx \
#        bash run_test.sh benchmark gemm
#
#   2. [salloc] Allocate nodes interactively and run:
#        salloc -N2 --cpus-per-task=256 --reservation=xxx
#        bash run_test.sh benchmark gemm
#
#   3. [sbatch] Submit as a batch job:
#        IMPORTANT: You MUST explicitly set SCRIPT_DIR in sbatch mode!
#        Example:
#           export SCRIPT_DIR=/path/to/scrips
#           sbatch --nodes=2 --reservation=xxx run_test.sh benchmark gemm
#
# Supported ENV variables (can be passed inline):
#   NNODES         - Number of nodes to use (default: 1)
#   CPUS_PER_TASK  - CPUs per task for srun (default: 256)
#   PARTITION      - Slurm partition name
#   RESERVATION    - Reservation name
#   NODELIST       - Optional comma-separated list of nodes
#   MASTER_PORT    - Port used for torch.distributed (default: 12345)
#   LOG_DIR        - Directory to save logs (default: ./logs)
#   SCRIPT_DIR     - [REQUIRED for sbatch] Path to the directory containing run_local_job.sh
#
# Output:
#   Logs will be saved to: ${LOG_DIR}/log_JOB-${JOB_ID}_${TIMESTAMP}.slurm.txt
###############################################################################

set -euo pipefail

# ----------------- Default environment variables -----------------
export MASTER_PORT="${MASTER_PORT:-12345}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-256}"
export PARTITION="${PARTITION:-}"
export RESERVATION="${RESERVATION:-}"

export LOG_DIR="${LOG_DIR:-"./logs"}"
mkdir -p "$LOG_DIR"

export SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"

# ----------------- Log file configuration -----------------
JOB_ID="${SLURM_JOB_ID:-manual}"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/log_JOB-${JOB_ID}_${TIMESTAMP}}"


launch_srun_with_args() {
    local srun_args=("${!1}")  # Expand srun argument array from variable name
    shift  # Remove array name from positional args

    echo "[INFO] Launching srun: ${srun_args[*]}"

    srun "${srun_args[@]}" bash -c "
        readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
        if [[ \"\$SLURM_NODEID\" = \"0\" ]]; then
            echo \"========== Slurm cluster info ==========\"
            echo \"SLURM_NODELIST: \${node_array[*]}\"
            echo \"SLURM_NNODES: \${SLURM_NNODES}\"
            echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
            echo \"\"
        fi
        export MASTER_ADDR=\${node_array[0]}
        export MASTER_PORT=\${MASTER_PORT}
        export NNODES=\${SLURM_NNODES}
        export NODE_RANK=\${SLURM_PROCID}
        export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
        export LOG_FILE=\${LOG_FILE}
        bash \${SCRIPT_DIR}/run_local_pretrain.sh \"\$@\" 2>&1 | tee ${LOG_FILE}.slurm.txt
    " bash "$@"
}

# ----------------- Scene 1: Outside Slurm (e.g., bash + srun) -----------------
run_via_bash_srun() {
    local srun_args=(
        --exclusive
        --ntasks-per-node=1
        --cpus-per-task="$CPUS_PER_TASK"
    )
    export NNODES="${NNODES:-1}"
    export NODELIST=${NODELIST:-}
    [[ -n "$PARTITION" ]] && srun_args+=("--partition=$PARTITION")
    [[ -n "$RESERVATION" ]] && srun_args+=("--reservation=$RESERVATION")
    [[ -n "$NODELIST" ]] && srun_args+=("--nodelist=$NODELIST") || srun_args+=("-N" "$NNODES")

    echo "[INFO] Launching outer srun: ${srun_args[*]}"
    launch_srun_with_args srun_args[@] "$@"
}

# ----------------- Scene 2: Inside salloc shell -----------------
run_within_salloc() {
    export NNODES="${NNODES:-1}"
    readarray -t node_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    total_nodes="${#node_array[@]}"
    local use_nodes="${NNODES:-$total_nodes}"

    if (( use_nodes > total_nodes )); then
        echo "[ERROR] Requested NNODES=$use_nodes exceeds allocated nodes $total_nodes"
        exit 1
    fi

    selected_nodes=("${node_array[@]:0:$use_nodes}")
    selected_nodelist=$(IFS=,; echo "${selected_nodes[*]}")

    echo "[INFO] Using $use_nodes node(s): $selected_nodelist"
    local srun_args=(
        --exclusive
        --ntasks-per-node=1
        --cpus-per-task="$CPUS_PER_TASK"
        -N "$use_nodes"
        --nodelist="$selected_nodelist"
    )

    launch_srun_with_args srun_args[@] "$@"
}

# ----------------- Scene 3: Inside sbatch script -----------------
run_within_sbatch() {
    echo "[INFO] Detected sbatch session (JOB_ID=$SLURM_JOB_ID)."

    readarray -t node_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    total_nodes="${#node_array[@]}"
    selected_nodes=("${node_array[@]}")
    selected_nodelist=$(IFS=,; echo "${selected_nodes[*]}")

    echo "[INFO] Using all allocated nodes ($total_nodes): $selected_nodelist"

    local srun_args=(
        --exclusive
        --ntasks-per-node="$SLURM_NTASKS_PER_NODE"
        --cpus-per-task="$SLURM_CPUS_PER_TASK"
    )

    launch_srun_with_args srun_args[@] "$@"
}

# ----------------- Entry: bash / salloc / sbatch -----------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ "${SLURM_STEP_ID:-}" == "4294967294" || "${SLURM_STEP_ID:-}" == "4294967290" ]]; then
        echo "[INFO] Detected salloc session (interactive shell)"
        run_within_salloc "$@"
    else
        echo "[INFO] Detected sbatch job (batch submission)"
        run_within_sbatch "$@"
    fi
else
    echo "[INFO] Not running under Slurm"
    run_via_bash_srun "$@"
fi
