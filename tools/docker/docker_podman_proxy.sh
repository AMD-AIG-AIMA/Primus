#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

if command -v podman &>/dev/null; then
    podman "$@"
elif command -v docker &>/dev/null; then
    docker "$@"
else
    echo "Neither Docker nor Podman found!" >&2
    return 1
fi
