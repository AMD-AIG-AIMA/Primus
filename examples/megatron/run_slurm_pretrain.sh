#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# Usage Guide: run_slurm_pretrain.sh
#
# This script is used to automatically launch a Primus distributed pretraining
# task in a Slurm cluster environment using Docker containers.
#
# Requirements:
#   - Slurm job scheduler with `srun`
#   - Docker or Podman runtime (for container execution)
#
# Environment Variables:
#   - EXP (optional)          : Path to the experiment config YAML file
#       Default: examples/megatron/exp_pretrain.yaml
#
#   - DOCKER_IMAGE (optional) : Docker image to use
#       Default: docker.io/rocm/megatron-lm:latest
#
#   - MASTER_PORT (optional)  : Master node port for distributed training
#       Default: 12345
#
#   - NUM_NODES (optional)    : Number of nodes to use
#       Default: 1
#
#   - LOG_DIR (optional)      : Output directory for logs
#       Default: output/
#
# Example Usage:
#   NUM_NODES=2 EXP=examples/megatron/exp_pretrain.yaml bash examples/megatron/run_slurm_pretrain.sh

export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:latest"}
export MASTER_PORT=${MASTER_PORT:-12345}

NUM_NODES=${NUM_NODES:-1}
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"========== Slurm cluster info ==========\"
              echo \"SLURM_NODELIST: \${node_array[*]}\"
              echo \"SLURM_NNODES: \$SLURM_NNODES\"
              echo \"SLURM_GPUS_ON_NODE: \$SLURM_GPUS_ON_NODE\"
              echo \"SLURM_WORLD_SIZE: \$SLURM_WORLD_SIZE\"
              echo \"SLURM_CPUS_PER_TASK: \$SLURM_CPUS_PER_TASK\"
              echo \"\"
          fi
          export MASTER_ADDR=\${node_array[0]}
          export MASTER_PORT=\${MASTER_PORT}
          export NNODES=\${SLURM_NNODES}
          export NODE_RANK=\${SLURM_PROCID}
          export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
          bash ${SCRIPT_DIR}/run_local_pretrain.sh 2>&1 | tee ${LOG_FILE}
     "
