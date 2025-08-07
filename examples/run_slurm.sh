#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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
  EXP             Path to experiment YAML file
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

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-1}
export LOG_DIR=${LOG_DIR:-"./logs"}
export CPUS_PER_TASK=${CPUS_PER_TASK:-256}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
mkdir -p "$LOG_DIR"

JOB_ID=${SLURM_JOB_ID:-manual}
TIMESTAMP=$(date +%Y%m%d%H%M%S)
export LOG_FILE=${LOG_FILE:-"${LOG_DIR}/log_JOB-${JOB_ID}_${TIMESTAMP}"}

srun -N "${NNODES}" \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_TASK:-256}" \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
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
          bash ${SCRIPT_DIR}/run_local.sh \"\$@\" 2>&1 | tee ${LOG_FILE}.slurm.txt
     " bash "$@"
