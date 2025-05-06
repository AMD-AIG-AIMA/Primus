#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

export MODEL_CONFIG=${MODEL_CONFIG:-deepseek_v2_lite}
export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:v25.4"}

export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
export NUM_NODES=${NUM_NODES:-1}

export MASTER_PORT=${MASTER_PORT:-12345}

export DATA_PATH=${DATA_PATH:-""}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     --gpus-per-node=8 \
     --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"==========Slurm cluster info==========\"
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
          bash ${SCRIPT_DIR}/run_pretrain.sh 2>&1 | tee output/log_slurm_pretrain.txt
     "
