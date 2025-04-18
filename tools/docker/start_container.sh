#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export DOCKER_IMAGE="docker.io/rocm/megatron-lm:latest"

podman run -d \
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
    -v "${PRIMUS_PATH}:${PRIMUS_PATH}" \
    $DOCKER_IMAGE sleep infinity
