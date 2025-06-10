#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

print_usage() {
cat << EOF
Usage: bash $(basename "$0") [--help]

Environment variables (must set before running):

    EXP                           # Path to experiment config file (required)
    BACKEND=megatron              # Training backend (default: megatron)
    NNODES=1                      # Number of nodes (default: 1)
    NODE_RANK=0                   # Current node rank (default: 0)
    GPUS_PER_NODE=8               # Number of GPUs per node (default: 8)
    MASTER_ADDR=localhost         # Master node address (default: localhost)
    MASTER_PORT=1234              # Master node port (default: 1234)
    PRIMUS_HIPBLASLT_TUNING_STAGE=0  # HipBLASLt tuning stage: 0/1/2/3 (default: 0)

HipBLASLt tuning stages:
    1: Dump GEMM shapes
    2: Offline tuning
    3: Use tuned config

Example:

    EXP=examples/megatron/exp_pretrain.yaml BACKEND=megatron bash examples/run_pretrain.sh

EOF
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# ----------- Cluster Configuration -----------
# Define distributed training parameters such as number of nodes,
# rank of each node, and master address/port for communication.

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

HOSTNAME=$(hostname)
if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training cluster info=========="
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "NNODES: $NNODES"
    echo "NODE_RANK: $NODE_RANK"
    echo "GPUS_PER_NODE: $GPUS_PER_NODE"
    echo ""
fi

LOG() { echo "$*"; }
LOG_INFO() { echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"; }
LOG_ERROR() { echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*"; }

# ----------- Framework Paths -----------
# Setup essential Python paths for Megatron-LM and Primus,
# ensuring all dependencies are correctly discoverable during execution.

# Set PRIMUS_PATH to the root directory of the framework
PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
LOG_INFO "PRIMUS_PATH is set to: ${PRIMUS_PATH}"

pip install -r $PRIMUS_PATH/requirements.txt  --quiet

# Set the backend framework to "megatron" by default unless overridden by the user.
# This environment variable controls which training backend Primus will use.
export BACKEND=${BACKEND:-"megatron"}
BACKEND_DIR="${PRIMUS_PATH}/examples/${BACKEND}"
if [[ ! -d "$BACKEND_DIR" ]]; then
    LOG_ERROR "Invalid BACKEND: '${BACKEND}'. Directory '${BACKEND_DIR}' not found."
    LOG ""
    exit 1
fi
LOG_INFO "BACKEND     is set to: ${BACKEND}"

export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
LOG_INFO "DATA_PATH   is set to: ${DATA_PATH}"

export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}
LOG_INFO "HF_HOME     is set to: ${HF_HOME}"


# ----------- Basic Framework Configuration -----------
# Load experiment configuration, model definition, and
# tokenizer settings from YAML files. These settings
# govern model architecture and tokenizer behavior.

# Ensure EXP is set, otherwise exit with error
if [ -z "${EXP:-}" ]; then
    LOG_ERROR "EXP must be specified (e.g., examples/megatron/exp_pretrain.yaml)." \
              "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
    LOG_ERROR "The specified EXP file does not exist: ${EXP}" \
              "Primus will use the configuration in EXP to train the model."
    exit 1
fi
LOG_INFO "EXP is set to: ${EXP}"


EXP_NAME=$(basename "$EXP" .yaml)
TRAIN_LOG="output/log_torchrun_pretrain_${EXP_NAME}.txt"

LOG "==========Training info=========="
LOG_INFO "[NODE-$NODE_RANK] EXP: $EXP"
LOG_INFO "[NODE-$NODE_RANK] TRAIN_LOG: $TRAIN_LOG"
LOG ""

# ----------- GPU and Communication Settings -----------
# Configure GPU-related environment variables and communication backend
# for efficient distributed training across multiple devices.

# Set visible GPUs for the current node (0 to GPUS_PER_NODE-1)
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

# ----------------- NCCL and Network Settings -----------------

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
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
    NCCL_SOCKET_IFNAME=$(bash "${PRIMUS_PATH}/examples/scripts/get_ip_interface.sh")
fi
export IP_INTERFACE

# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

# ----------------- AMD-specific GPU optimizations -----------------

# Enable system DMA engine (SDMA) on AMD GPUs for better IO throughput
export HSA_ENABLE_SDMA=1

# Prevent scratch memory from being reclaimed to stabilize large memory usage patterns (e.g., KV cache, MoE experts)
# NOTE: Must disable scratch reclaim to avoid MoE training crash on AMD GPUs
# Setting this to 0 prevents core dumps when using Mixture-of-Experts (MoE) models
export HSA_NO_SCRATCH_RECLAIM=0

# Disable MSCCL (RCCL multi-connection feature) for better stability
export RCCL_MSCCL_ENABLE=0

# ----------------- Performance tuning -----------------

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Limit GPU hardware queues to 2 for performance stability
export GPU_MAX_HW_QUEUES=2

# Limit max CUDA device connections to reduce PCIe traffic
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# Prioritize NCCL communication for PyTorch for higher throughput
export TORCH_NCCL_HIGH_PRIORITY=1

# ----------------- NVTE FP8 -----------------
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
# export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1


# ----------- HipBLASLt Tuning -----------
# Configure HipBLASLt tuning stage to either dump GEMM shapes for profiling
# or apply tuned configurations for optimized GEMM execution on AMD GPUs.

handle_hipblaslt_tuning() {
    local STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
    local TUNE_LOG_PATH=${PRIMUS_PATH}/output/tune_hipblaslt/${MODEL}
    local RESULT_FILE=tune_hipblas_gemm_results.txt

    mkdir -p "$TUNE_LOG_PATH"

    case $STAGE in
        1)
            [[ "$TE_HIPBLASLT_TUNING" == "1" ]] && error_exit "Disable TE_HIPBLASLT_TUNING for shape dump"
            mkdir -p "$TUNE_LOG_PATH/gemm_shape"
            export HIPBLASLT_LOG_MASK=32
            export HIPBLASLT_LOG_FILE=${HIPBLASLT_LOG_FILE:-"$TUNE_LOG_PATH/gemm_shape/dump_hipblaslt_gemm_shape_${NODE_RANK}.txt"}
            unset HIPBLASLT_TUNING_OVERRIDE_FILE
            ;;
        2)
            mkdir -p "$TUNE_LOG_PATH/gemm_tune"
            python "${PRIMUS_PATH}/examples/offline_tune/offline_tune_gemm.py" \
                --dump-shape-path-or-file "$TUNE_LOG_PATH/gemm_shape" \
                --tune-result-path "$TUNE_LOG_PATH/gemm_tune/$RESULT_FILE" \
                --num-devices 8
            echo "GEMM tuning finished. Set PRIMUS_HIPBLASLT_TUNING_STAGE=3 and re-run training."
            exit 0
            ;;
        3)
            TUNE_FILE="$TUNE_LOG_PATH/gemm_tune/$RESULT_FILE"
            [[ ! -f "$TUNE_FILE" ]] && error_exit "Missing tuning result: $TUNE_FILE"
            export HIPBLASLT_TUNING_OVERRIDE_FILE=$TUNE_FILE
            ;;
    esac

    if [ "$NODE_RANK" = "0" ]; then
        echo "========== Training tuning info =========="
        echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING: $TE_HIPBLASLT_TUNING"
        echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_RUN_COUNT: $TE_HIPBLASLT_TUNING_RUN_COUNT"
        echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_ALGO_COUNT: $TE_HIPBLASLT_TUNING_ALGO_COUNT"
        echo "[NODE-$NODE_RANK] NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
        echo "[NODE-$NODE_RANK] PRIMUS_HIPBLASLT_TUNING_STAGE: ${PRIMUS_HIPBLASLT_TUNING_STAGE}"
        echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_MASK: ${HIPBLASLT_LOG_MASK}"
        echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_FILE: ${HIPBLASLT_LOG_FILE}"
        echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_LEVEL: ${HIPBLASLT_LOG_LEVEL}"
        echo "[NODE-$NODE_RANK] HIPBLASLT_TUNING_OVERRIDE_FILE: ${HIPBLASLT_TUNING_OVERRIDE_FILE}"
        if [ $STAGE -eq 1 ]; then
            echo "[NODE-$NODE_RANK] Dump HipBLASLt shapes, make sure train_iters is set to a very small value."
        fi
        echo ""
    fi
}

handle_hipblaslt_tuning

# ----------- Backend Preparation -----------
# Source the backend-specific prepare.sh script to perform necessary setup.
# This may include dataset preprocessing, tokenizer setup, or other initialization
# required before training. If sourcing fails, the script will exit immediately.

LOG_INFO "Preparing using backend: ${BACKEND}"
SCRIPT="${PRIMUS_PATH}/examples/${BACKEND}/prepare.sh"
# shellcheck disable=SC1090
if ! source "$SCRIPT"; then
    LOG_ERROR "Backend preparation failed: $SCRIPT"
    exit 1
fi

# ----------- Python Path Setup -----------
# Configure PYTHONPATH to include:
#   - site-packages (for installed packages),
#   - Primus project root,
#   - all subdirectories in third_party.
# This ensures both internal modules and third-party dependencies are importable.

setup_pythonpath() {
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

    local third_party_path="${PRIMUS_PATH}/third_party"
    local third_party_pythonpath=""
    while IFS= read -r dir; do
        third_party_pythonpath+="${dir}:"
    done < <(find "${third_party_path}" -mindepth 1 -maxdepth 1 -type d -exec realpath {} \;)
    third_party_pythonpath="${third_party_pythonpath%:}"

    export PYTHONPATH="${site_packages}:${PRIMUS_PATH}:${third_party_pythonpath}:${PYTHONPATH}"
}

setup_pythonpath

# ----------- Distributed Launch -----------
# Launch distributed training via torchrun using configured arguments.
# Logs are captured via tee. Exit code is preserved for upstream control flow.

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

if [[ -n "$LOCAL_RANKS_FILTER" ]]; then
    DISTRIBUTED_ARGS+=(--local-ranks-filter "$LOCAL_RANKS_FILTER")
fi


# Launch distributed training using torchrun and tee logs
torchrun "${DISTRIBUTED_ARGS[@]}" examples/${BACKEND}/pretrain.py --exp $EXP "$@" 2>&1 | tee $TRAIN_LOG
exit_code=${PIPESTATUS[0]}

if [ "${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}" -eq 1 ]; then
    echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-1]: HipBlasLT gemm shape dump is finished, " \
         "please set PRIMUS_HIPBLASLT_TUNING_STAGE to 2, " \
         "and tune the gemm with a single node."
fi

exit $exit_code
