#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/third_party/Megatron-LM}
[[ ! -d "${MEGATRON_PATH}" || -z "$(ls -A "${MEGATRON_PATH}")" ]] && {
    echo "Error: MEGATRON_PATH (${MEGATRON_PATH}) does not exist or is empty"
    exit 1
}

# cluster envs
RUN_ENV="${RUN_ENV:-torchrun}"
if [ "$RUN_ENV" = "torchrun" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-1234}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight experiment info=========="
    echo "[NODE-$NODE_RANK] PRIMUS_PATH: $PRIMUS_PATH"
    echo "[NODE-$NODE_RANK] MEGATRON_PATH: $MEGATRON_PATH"
    echo "[NODE-$NODE_RANK] RUN_ENV: $RUN_ENV"
    echo ""
fi

# Enable high-speed DMA transfers on AMD GPUs
export HSA_ENABLE_SDMA=1  # Enable system DMA (SDMA) engine for better GPU IO throughput

# Prevent scratch memory space from being reclaimed
export HSA_NO_SCRATCH_RECLAIM=1  # Helps stabilize large memory usage patterns (e.g. KV cache, MoE experts)

export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
NCCL_IB_HCA=$(bash "${PRIMUS_PATH}"/examples/scripts/get_nccl_ib_hca.sh)
export NCCL_IB_HCA
export NCCL_IB_GDR_LEVEL=2
export NCCL_NET_GDR_LEVEL=2
IP_INTERFACE=$(bash "${PRIMUS_PATH}"/examples/scripts/get_ip_interface.sh)
export IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IP_INTERFACE}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${IP_INTERFACE}}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export RCCL_MSCCL_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
# VERSION, WARN, INFO, DEBUG
export NCCL_DEBUG=""

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight cluster info=========="
    echo "[NODE-$NODE_RANK] MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NNODES: $NNODES"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] GPUS_PER_NODE: $GPUS_PER_NODE"
    echo "[NODE-$NODE_RANK] HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo ""
else
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
fi

PREFLIGHT_LOG=output/log_torchrun_preflight.txt
if [ "$NODE_RANK" = "0" ]; then
    echo "==========Preflight logging info=========="
    echo "[NODE-$NODE_RANK] PREFLIGHT_LOG: $PREFLIGHT_LOG"
    echo ""
fi

if [ "$RUN_ENV" = "torchrun" ]; then
    export PYTHONPATH=${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}
    export PULSEKIT_PYTORCH_LAUNCHER_SCRIPT=tools/preflight/preflight_perf_test.py
    export PULSEKIT_PYTORCH_LAUNCHER_RESULT_FILES=output/preflight/preflight_report.json
    python -m primus_safe_pulsekit_pytorch_launcher


else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
