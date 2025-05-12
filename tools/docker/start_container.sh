#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
DOCKER_IMAGE="docker.io/rocm/megatron-lm:latest"
DATA_PATH=${DATA_PATH:-"/apps/tas/0_public/data"}

bash "${PRIMUS_PATH}"/tools/docker/docker_podman_proxy.sh run -d \
    --name dev_primus \
    --ipc=host \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add video \
    --privileged \
    --env DATA_PATH="${DATA_PATH}" \
    -v "${PRIMUS_PATH}:${PRIMUS_PATH}" \
    -v "${DATA_PATH}:${DATA_PATH}" \
    -w "${PRIMUS_PATH}" \
    $DOCKER_IMAGE sleep infinity
