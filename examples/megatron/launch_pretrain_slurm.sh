#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# available models: primus/configs/models/megatron
MODEL_CONFIG=${MODEL_CONFIG:-deepseek_v2_lite}

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")

# data path
DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/pretrain_data"}
mkdir -p $DATA_PATH

# cluster envs
readarray -t node_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${node_array[0]}
MASTER_PORT=$(shuf -i 1024-65535 -n 1)
SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-8}
SLURM_WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

if [ "$SLURM_NODEID" = "0" ]; then
    echo "==========Slurm cluster info=========="
    echo "[SLURM-NODE-$SLURM_NODEID] NODELIST=${node_array[*]}"
    echo "[SLURM-NODE-$SLURM_NODEID] SLURM_NNODES=$SLURM_NNODES"
    echo "[SLURM-NODE-$SLURM_NODEID] SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
    echo "[SLURM-NODE-$SLURM_NODEID] SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE"
    echo "[SLURM-NODE-$SLURM_NODEID] SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
    echo "[SLURM-NODE-$SLURM_NODEID] SLURM_PROCID: $SLURM_PROCID"
    echo ""
fi

ENV_ARGS=$(env | grep "^${PRIMUS_}" | awk -F= '{print "--env", $1}' | xargs)

DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:latest"}
bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run --rm \
    --env MASTER_ADDR=${MASTER_ADDR} \
    --env MASTER_PORT=${MASTER_PORT} \
    --env NNODES=${SLURM_NNODES} \
    --env NODE_RANK=${SLURM_NODEID} \
    --env GPUS_PER_NODE=${SLURM_GPUS_ON_NODE} \
    --env MODEL_CONFIG=${MODEL_CONFIG} \
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
    $DOCKER_IMAGE /bin/bash -c \
        "echo '[NODE-${SLURM_NODEID}]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        pip install -q loguru wandb nltk && \
        cd $PRIMUS_PATH && \
        bash examples/megatron/run_pretrain.sh 2>&1 && \
        echo '[NODE-${SLURM_NODEID}]: end time=$(date +"%Y.%m.%d %H:%M:%S")'"
