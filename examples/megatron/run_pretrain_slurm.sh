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

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     bash ${SCRIPT_DIR}/launch_pretrain_slurm.sh 2>&1 | tee output/log_slurm_pretrain.txt
