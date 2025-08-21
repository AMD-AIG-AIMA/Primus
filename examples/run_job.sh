#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

print_usage() {
cat << EOF
Usage: bash $(basename "$0") [--help]

Environment variables (must set before running):

    NNODES=1                      # Number of nodes (default: 1)
    NODE_RANK=0                   # Current node rank (default: 0)
    GPUS_PER_NODE=8               # Number of GPUs per node (default: 8)
    MASTER_ADDR=localhost         # Master node address (default: localhost)
    MASTER_PORT=1234              # Master node port (default: 1234)

Example:
    bash examples/run_pretrain.sh --config examples/megatron/exp_pretrain.yaml

EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

HOSTNAME=$(hostname)

LOG_INFO() {
    if [ "$*" = "" ]; then
        echo ""
    else
        echo "[NODE-$NODE_RANK($HOSTNAME)] $*"
    fi
}

LOG_INFO_RANK0() {
    if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] $*"
        fi
    fi
}

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*";
}

# Global list to track already printed variables
__PRINTED_EXPORT_VARS_SET=""

log_exported_vars() {
    local title="${1:-Exported Environment Variables (Incremental)}"
    LOG_INFO_RANK0 "========== $title =========="

    local script_source="${BASH_SOURCE[1]}"  # Caller script
    local exported_vars
    exported_vars=$(declare -xp | awk '{print $3}' | cut -d= -f1)

    for var in $exported_vars; do
        [[ -z "$var" ]] && continue
        if [[ "$__PRINTED_EXPORT_VARS_SET" =~ (^| )$var($| ) ]]; then
            continue  # Skip already printed
        fi

        if grep -qE "^[[:space:]]*export[[:space:]]+$var\b" "$script_source"; then
            LOG_INFO_RANK0 "    $var=${!var}"
            __PRINTED_EXPORT_VARS_SET+=" $var"
        fi
    done
}

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

log_exported_vars "Training cluster info"

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
export PRIMUS_PATH
export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}

pip install -r "$PRIMUS_PATH/requirements.txt" --quiet --no-warn-script-location 2>/dev/null


export TRAIN_LOG=${TRAIN_LOG:-"output/log_torchrun_pretrain_$(basename "$EXP" .yaml).txt"}

log_exported_vars "Training info"

# -------------------- NCCL and Communication Setup --------------------
# Set visible GPUs for the current node (0 to GPUS_PER_NODE-1)
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

# ----------------- NCCL and Network Settings -----------------
# VERSION, WARN, INFO, DEBUG, TRACE
export NCCL_DEBUG=

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Set InfiniBand GID index for NCCL communication
export NCCL_IB_GID_INDEX=3

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=0

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA}" ]; then
    NCCL_IB_HCA=$(bash "${PRIMUS_PATH}/examples/scripts/get_nccl_ib_hca.sh")
fi
export NCCL_IB_HCA

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE}" ]; then
    IP_INTERFACE=$(bash "${PRIMUS_PATH}/examples/scripts/get_ip_interface.sh")
fi
export IP_INTERFACE
# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

log_exported_vars "NCCL and Network Settings"

# ----------------- AMD-specific GPU optimizations -----------------
# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=1

# Prevent scratch memory from being reclaimed to stabilize large memory usage patterns (e.g., KV cache, MoE experts)
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
# Setting this to 0 prevents core dumps when using Mixture-of-Experts (MoE) models
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-0}

# Disable MSCCL (RCCL multi-connection feature) for better stability
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1*1024*1024*1024)) # default 1 MB
# https://github.com/microsoft/mscclpp/blob/main/include/mscclpp/env.hpp#L82-L87
export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
# pytorch need set this env to enable register comm
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

log_exported_vars "AMD-specific GPU optimizations"


# ----------------- Performance tuning -----------------
# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=2

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=1

# optimize nvte fp8 cast transpose
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0

# Note: Disable v3 due to accuracy issues. Will fix after TE version 2.1.
export NVTE_CK_USES_BWD_V3=${NVTE_CK_USES_BWD_V3:-0}

# nvte debug envs
export NVTE_DEBUG=0 # 0, 1
export NVTE_DEBUG_LEVEL=0 # 0, 1, 2
export NVTE_FUSED_ATTN_LOG_CONFIG=0 # 0, 1
export PATCH_TE_FLASH_ATTN=${PATCH_TE_FLASH_ATTN:-0}

log_exported_vars "Performance tuning"

# -------------------- setup_pythonpath -------------------
site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH="${PRIMUS_PATH}:${site_packages}:${PYTHONPATH:-}"
log_exported_vars "pythonpath"

# -------------------- Launch Training --------------------
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# TOTAL_RANKS=$((GPUS_PER_NODE * NNODES))
# LOCAL_RANKS="0,$((TOTAL_RANKS - 1))"
# CMD="torchrun ${DISTRIBUTED_ARGS[*]} --local-ranks-filter ${LOCAL_RANKS} primus/cli/main.py $*"

LAST_NODE=$((NNODES - 1))
FILTERS=()

# Add local rank 0 on the first node
if [ "$NODE_RANK" -eq 0 ]; then
    FILTERS+=(0)
fi

# Add the last local rank on the last node
if [ "$NODE_RANK" -eq "$LAST_NODE" ]; then
    FILTERS+=($((GPUS_PER_NODE - 1)))
fi

# Build filter argument (only if FILTERS is non-empty)
if [ "${#FILTERS[@]}" -gt 0 ]; then
    LOCAL_FILTER=$(IFS=,; echo "${FILTERS[*]}")
    FILTER_ARG=(--local-ranks-filter "$LOCAL_FILTER")
else
    FILTER_ARG=()
fi
CMD="torchrun ${DISTRIBUTED_ARGS[*]} ${FILTER_ARG[*]} ${LOCAL_RANKS} primus/cli/main.py $*"


LOG_INFO "Launching distributed training with command: $CMD"

eval "$CMD" 2>&1 | tee "$TRAIN_LOG"
exit_code=${PIPESTATUS[0]}

LOG_INFO "torchrun exited with code $exit_code"

if [[ $exit_code -ne 0 ]]; then
    if [[ $exit_code -ge 128 ]]; then
        signal=$((exit_code - 128))
        LOG_ERROR "torchrun crashed due to signal $signal"
    else
        LOG_ERROR "torchrun exited with code $exit_code"
    fi
fi

exit "$exit_code"
