#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

export RUN_ENV=slurm
export MODEL_CONFIG=deepseek_v2_lite

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

srun -N 1 \
     --nodelist=smc300x-ccs-aus-a16-10 \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=64 \
     bash ${SCRIPT_DIR}/run_pretrain.sh 2>&1 | tee output/log_slurm_pretrain.txt
