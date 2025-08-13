#!/bin/bash
#SBATCH --job-name=primus-job          # [Optional] Slurm job name
#SBATCH --output=logs/slurm-%j.out     # [Optional] Stdout log
#SBATCH --error=logs/slurm-%j.err      # [Optional] Stderr log
##SBATCH --partition=amd-aig-fc        # (Optional) Partition if set via #SBATCH
##SBATCH --nodes=2                     # (Optional) Set nodes here if desired
##SBATCH --nodelist=pdfc-aig-[000001,000002]  # (Optional) Or override in script
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=256
##SBATCH --exclusive

set -euo pipefail

# ------------------ Help ------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
cat <<EOF
Usage: bash run_slurm.sh [OPTIONS]

Launch a distributed Primus task via Slurm and Docker.

Requirements:
  - Slurm job scheduler with 'srun'
  - Docker or Podman runtime

Environment Variables:
  NNODES          Number of nodes to use [default: 1]
  MASTER_PORT     Port for distributed communication [default: 12345]
  CPUS_PER_TASK   CPUs per task [default: 256]
  LOG_DIR         Directory for logs [default: ./output]

Example:
  export EXP=examples/megatron/exp_pretrain.yaml
  export DATA_PATH=/mnt/data
  NNODES=2 bash run_slurm.sh

EOF
exit 0
fi

# ----------------- Default environment variables -----------------
export MASTER_PORT="${MASTER_PORT:-12345}"   # Master port for torchrun / training backend
export NNODES="${NNODES:-1}"                 # Number of nodes (used only if NODELIST is not set)
export LOG_DIR="${LOG_DIR:-"./logs"}"        # Directory to store logs
export CPUS_PER_TASK="${CPUS_PER_TASK:-256}" # Number of CPU cores per task
export PARTITION="${PARTITION:-}"            # Slurm partition name (optional)

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
mkdir -p "$LOG_DIR"

# ----------------- Log file configuration -----------------
JOB_ID="${SLURM_JOB_ID:-manual}"                   # Use Slurm job ID if available, otherwise fallback
TIMESTAMP=$(date +%Y%m%d%H%M%S)                    # Timestamp for unique log filename
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/log_JOB-${JOB_ID}_${TIMESTAMP}}"

# ----------------- Construct srun arguments -----------------
SRUN_ARGS=(
  --exclusive                                      # Allocate nodes exclusively
  --ntasks-per-node=1                              # One task per node
  --cpus-per-task="$CPUS_PER_TASK"                 # Number of CPUs per task
)

# Add partition if specified
if [[ -n "$PARTITION" ]]; then
  SRUN_ARGS+=("--partition=$PARTITION")
fi

# Either use specific NODELIST or default to NNODES
if [[ -n "${NODELIST:-}" ]]; then
  SRUN_ARGS+=("--nodelist=$NODELIST")
else
  SRUN_ARGS+=("-N" "$NNODES")
fi

# ----------------- Launch with srun -----------------

srun -N "${NNODES}" \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_TASK:-256}" \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"========== Slurm cluster info ==========\"
              echo \"SLURM_NODELIST: \${node_array[*]} \${SLURM_JOB_NODELIST}\"
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
          bash ${SCRIPT_DIR}/run_local_job.sh \"\$@\" 2>&1 | tee ${LOG_FILE}.slurm.txt
     " bash "$@"
