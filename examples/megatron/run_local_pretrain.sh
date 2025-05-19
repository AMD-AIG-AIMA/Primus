#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# Usage Guide: run_local_pretrain.sh
#
# This script runs a Primus distributed pretraining task *inside* a Docker container
# using local environment variables and cluster settings (e.g., from Slurm).
#
# Requirements:
#   - Docker or Podman must be installed and accessible
#   - A valid model config YAML file referenced by MODEL_CONFIG
#
# Environment Variables:
#   - EXP (optional): Experiment config file (YAML format)
#       Default: examples/megatron/exp_pretrain.yaml
#
#   - DATA_PATH (required): Path to dataset directory
#
#   - MASTER_ADDR, MASTER_PORT (optional): For multi-node distributed setup
#       Defaults: MASTER_ADDR=localhost, MASTER_PORT=1234
#
#   - NNODES, NODE_RANK, GPUS_PER_NODE (optional): Distributed training params
#       Defaults: NNODES=1, NODE_RANK=0, GPUS_PER_NODE=8
#
#   - PRIMUS_* variables (optional): Any additional environment variables prefixed
#       with PRIMUS_ will be forwarded into the container.
#
# Example Usage:
#   export EXP=examples/megatron/exp_pretrain.yaml
#   export DATA_PATH=/mnt/data
#   bash examples/megatron/run_local_pretrain.sh


set -e

# Path to experiment configuration YAML
EXP=${EXP:-"examples/megatron/exp_pretrain.yaml"}

# Default docker image
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:latest"}

# Project root
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")

# Dataset directory
DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}


# ------------------ Validation Checks ------------------

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
  echo "[ERROR] The specified EXP file does not exist: ${EXP}"
  echo "        Primus will use the configuration in EXP to train the model."
  exit 1
fi

# Ensure DATA_PATH is not empty
if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH is empty. Please set DATA_PATH environment variable."
  exit 1
fi

# (Optional) Check if directory exists
if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH directory '$DATA_PATH' does not exist."
  exit 1
fi

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "$NODE_RANK" = "0" ]; then
    echo "========== Cluster info =========="
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "NNODES: $NNODES"
    echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo ""
fi

# Pass all PRIMUS_ environment variables into the container
ENV_ARGS=$(env | grep "^PRIMUS_" | awk -F= '{print "--env", $1}' | xargs)

HOSTNAME=$(hostname)

# ------------------ Launch Training Container ------------------
bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run --rm \
    --env MASTER_ADDR=${MASTER_ADDR} \
    --env MASTER_PORT=${MASTER_PORT} \
    --env NNODES=${NNODES} \
    --env NODE_RANK=${NODE_RANK} \
    --env GPUS_PER_NODE=${GPUS_PER_NODE} \
    --env EXP=${EXP} \
    --env DATA_PATH=${DATA_PATH} \
    --env HF_TOKEN \
    ${ENV_ARGS} \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged --device=/dev/infiniband \
    -v $PRIMUS_PATH:$PRIMUS_PATH \
    -v $DATA_PATH:$DATA_PATH \
    $DOCKER_IMAGE /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        cd $PRIMUS_PATH && \
        pip install -r requirements.txt && \
        bash examples/megatron/run_pretrain.sh 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'
    "
