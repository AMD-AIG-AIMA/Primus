#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

export RUN_ENV=slurm
export MODEL_CONFIG=deepseek_v2_lite

export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
export NUM_NODES=${NUM_NODES:-8}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

     # --nodelist=smc300x-ccs-aus-a16-10 \
srun -N ${NUM_NODES} \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=256 \
     bash ${SCRIPT_DIR}/run_pretrain.sh 2>&1 | tee output/log_slurm_pretrain.txt
