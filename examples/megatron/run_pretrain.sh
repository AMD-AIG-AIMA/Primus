#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

MODEL_CONFIG=${MODEL_CONFIG:-deepseek_v2_lite}
DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/rocm/megatron-lm:latest"}

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")

DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/pretrain_data"}
: "${DATA_PATH:=${PRIMUS_PATH}/pretrain_data}"

mkdir -p $DATA_PATH

# cluster envs
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "$NODE_RANK" = "0" ]; then
    echo "========== Cluster info =========="
    echo "[NODE-$NODE_RANK] MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK] NNODES: $NNODES"
    echo "[NODE-$NODE_RANK] GPUS_PER_NODE: $GPUS_PER_NODE"
echo ""
fi

ENV_ARGS=$(env | grep "^PRIMUS_" | awk -F= '{print "--env", $1}' | xargs)

HOSTNAME=$(hostname)
bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run --rm \
    --env MASTER_ADDR=${MASTER_ADDR} \
    --env MASTER_PORT=${MASTER_PORT} \
    --env NNODES=${NNODES} \
    --env NODE_RANK=${NODE_RANK} \
    --env GPUS_PER_NODE=${GPUS_PER_NODE} \
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
        "echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
         source /opt/conda/etc/profile.d/conda.sh && \
         conda activate py_3.10 && \
         pip install -q loguru wandb nltk && \
         cd $PRIMUS_PATH && \
         bash examples/megatron/launch_pretrain.sh 2>&1 && \
         echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'"
