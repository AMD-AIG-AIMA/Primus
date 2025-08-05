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

    EXP                           # Path to experiment config file (required)
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

    EXP=examples/megatron/exp_pretrain.yaml bash examples/run_pretrain.sh

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
        echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
    fi
}

LOG_INFO_RANK0() {
    if [ "$NODE_RANK" -ne 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
        fi
    fi
}

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*";
}

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}


LOG_INFO_RANK0 "==========Training cluster info=========="
LOG_INFO_RANK0 "MASTER_ADDR: $MASTER_ADDR"
LOG_INFO_RANK0 "MASTER_PORT: $MASTER_PORT"
LOG_INFO_RANK0 "NNODES: $NNODES"
LOG_INFO_RANK0 "NODE_RANK: $NODE_RANK"
LOG_INFO_RANK0 "GPUS_PER_NODE: $GPUS_PER_NODE"
LOG_INFO_RANK0 ""

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}

pip install -r "$PRIMUS_PATH/requirements.txt"  --quiet

# -------------------- EXP Check --------------------
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


TRAIN_LOG=${TRAIN_LOG:-"output/log_torchrun_pretrain_$(basename "$EXP" .yaml).txt"}

LOG_INFO_RANK0 "==========Training info=========="
LOG_INFO_RANK0 "EXP: $EXP"
LOG_INFO_RANK0 "TRAIN_LOG: $TRAIN_LOG"
LOG_INFO_RANK0 "PRIMUS_PATH: $PRIMUS_PATH"
LOG_INFO_RANK0 "DATA_PATH: $DATA_PATH"
LOG_INFO_RANK0 "HF_HOME: $HF_HOME"
LOG_INFO_RANK0 ""

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

LOG_INFO_RANK0 "==========NCCL and Network Settings=========="
LOG_INFO_RANK0 "NCCL_DEBUG: $NCCL_DEBUG"
LOG_INFO_RANK0 "NCCL_CHECKS_DISABLE: $NCCL_CHECKS_DISABLE"
LOG_INFO_RANK0 "NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
LOG_INFO_RANK0 "NCCL_CROSS_NIC: $NCCL_CROSS_NIC"
LOG_INFO "NCCL_IB_HCA: $NCCL_IB_HCA"
LOG_INFO "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
LOG_INFO "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
LOG_INFO ""

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

LOG_INFO_RANK0 "==========AMD-specific GPU optimizations=========="
LOG_INFO_RANK0 "HSA_ENABLE_SDMA: $HSA_ENABLE_SDMA"
LOG_INFO_RANK0 "HSA_NO_SCRATCH_RECLAIM: $HSA_NO_SCRATCH_RECLAIM"
LOG_INFO_RANK0 "RCCL_MSCCL_ENABLE: $RCCL_MSCCL_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_ENABLE: $RCCL_MSCCLPP_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_FORCE_ENABLE: $RCCL_MSCCLPP_FORCE_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_THRESHOLD: $RCCL_MSCCLPP_THRESHOLD"
LOG_INFO_RANK0 "MSCCLPP_DISABLE_CHANNEL_CACHE: $MSCCLPP_DISABLE_CHANNEL_CACHE"
LOG_INFO_RANK0 "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: $TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"
LOG_INFO_RANK0 ""

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

LOG_INFO_RANK0 "==========Performance tuning=========="
LOG_INFO_RANK0 "GPU_MAX_HW_QUEUES: $GPU_MAX_HW_QUEUES"
LOG_INFO_RANK0 "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
LOG_INFO_RANK0 "TORCH_NCCL_HIGH_PRIORITY: $TORCH_NCCL_HIGH_PRIORITY"
LOG_INFO_RANK0 "NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
LOG_INFO_RANK0 "NVTE_USE_CAST_TRANSPOSE_TRITON: $NVTE_USE_CAST_TRANSPOSE_TRITON"
LOG_INFO_RANK0 "NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE: $NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE"
if [[ "$PATCH_TE_FLASH_ATTN" == "1" ]]; then
    LOG_INFO_RANK0 'Patching _flash_attn_max_version in attention.py...'
    sed -i 's/_flash_attn_max_version = PkgVersion(\".*\")/_flash_attn_max_version = PkgVersion(\"3.0.0.post1\")/' \
        /opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py
    LOG_INFO_RANK0 'Patch complete.'
fi
LOG_INFO_RANK0 ""

# ----------------- Rebuild nbxt -----------------
export REBUILD_BNXT=${REBUILD_BNXT:-0}
export PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}

if [[ "$REBUILD_BNXT" == "1" && -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    LOG_INFO "Rebuilding bnxt from $PATH_TO_BNXT_TAR_PACKAGE ..." && \
    tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/ && \
    mv /tmp/libbnxt_re-* /tmp/libbnxt && \
    mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox && \
    cd /tmp/libbnxt/ && sh ./autogen.sh && ./configure && \
    make -C /tmp/libbnxt clean all install && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf && \
    ldconfig && \
    cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/ && \
    cd "${PRIMUS_PATH}" && \
    LOG_INFO "Rebuilding libbnxt done."
else
  LOG_INFO "Skip bnxt rebuild. REBUILD_BNXT=$REBUILD_BNXT, PATH_TO_BNXT_TAR_PACKAGE=$PATH_TO_BNXT_TAR_PACKAGE"
fi



# -------------------- HipBLASLt Tuning --------------------
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
            LOG_ERROR "GEMM tuning finished. Set PRIMUS_HIPBLASLT_TUNING_STAGE=3 and re-run training."
            exit 0
            ;;
        3)
            TUNE_FILE="$TUNE_LOG_PATH/gemm_tune/$RESULT_FILE"
            [[ ! -f "$TUNE_FILE" ]] && error_exit "Missing tuning result: $TUNE_FILE"
            export HIPBLASLT_TUNING_OVERRIDE_FILE=$TUNE_FILE
            ;;
    esac

    if [ "$NODE_RANK" = "0" ]; then
        LOG_INFO "========== Training tuning info =========="
        LOG_INFO "TE_HIPBLASLT_TUNING: $TE_HIPBLASLT_TUNING"
        LOG_INFO "TE_HIPBLASLT_TUNING_RUN_COUNT: $TE_HIPBLASLT_TUNING_RUN_COUNT"
        LOG_INFO "TE_HIPBLASLT_TUNING_ALGO_COUNT: $TE_HIPBLASLT_TUNING_ALGO_COUNT"
        LOG_INFO "PRIMUS_HIPBLASLT_TUNING_STAGE: ${PRIMUS_HIPBLASLT_TUNING_STAGE}"
        LOG_INFO "HIPBLASLT_LOG_MASK: ${HIPBLASLT_LOG_MASK}"
        LOG_INFO "HIPBLASLT_LOG_FILE: ${HIPBLASLT_LOG_FILE}"
        LOG_INFO "HIPBLASLT_LOG_LEVEL: ${HIPBLASLT_LOG_LEVEL}"
        LOG_INFO "HIPBLASLT_TUNING_OVERRIDE_FILE: ${HIPBLASLT_TUNING_OVERRIDE_FILE}"
        if [ "$STAGE" -eq 1 ]; then
            LOG_INFO "Dump HipBLASLt shapes, make sure train_iters is set to a very small value."
        fi
        LOG_INFO ""
    fi
}

handle_hipblaslt_tuning

# -------------------- Python Path Setup --------------------

check_dir_nonempty() {
    local dir_path=$1
    local name=$2
    if [[ ! -d "$dir_path" || -z "$(ls -A "$dir_path")" ]]; then
        echo "[ERROR] $name ($dir_path) does not exist or is empty."
        echo "        Please ensure Primus is properly initialized."
        echo
        echo "        If not yet cloned, run:"
        echo "            git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git"
        echo
        echo "        Or if already cloned, initialize submodules with:"
        echo "            git submodule update --init --recursive"
        echo
        exit 1
    fi
}


setup_pythonpath() {
    # Get site-packages directory for current Python environment
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

    local third_party_path="${PRIMUS_PATH}/third_party"
    local third_party_pythonpath=""

    # Define backend names that can be overridden via environment variables
    local CUSTOM_BACKENDS=("megatron" "torchtitan")
    declare -A CUSTOM_BACKEND_PATHS

    # Load backend paths from environment variables (e.g., MEGATRON_PATH)
    for backend in "${CUSTOM_BACKENDS[@]}"; do
        # Convert backend name to uppercase and append _PATH (e.g., MEGATRON_PATH)
        env_var_name="$(echo "${backend}_path" | tr '[:lower:]' '[:upper:]')"
        backend_path="${!env_var_name}"
        if [[ -n "$backend_path" ]]; then
            check_dir_nonempty "$env_var_name" "$backend_path"
            CUSTOM_BACKEND_PATHS["$backend"]="$backend_path"
        fi
    done

    declare -A DIR_TO_BACKEND=(
        ["Megatron-LM"]="megatron"
        ["torchtitan"]="torchtitan"
    )
    # Collect third_party paths, excluding overridden backends
    while IFS= read -r dir; do
        base_name=$(basename "$dir")
        base_name="${DIR_TO_BACKEND[$base_name]}"
        if [[ -n "${CUSTOM_BACKEND_PATHS[$base_name]}" ]]; then
            continue
        fi
        third_party_pythonpath+="${dir}:"
    done < <(find "${third_party_path}" -mindepth 1 -maxdepth 1 -type d -exec realpath {} \;)

    third_party_pythonpath="${third_party_pythonpath%:}"  # Remove trailing colon

    # Start building final PYTHONPATH
    local full_pythonpath="${site_packages}:${PRIMUS_PATH}:${third_party_pythonpath}:${PRIMUS_PATH}/third_party/torchtitan/torchtitan/experiments/deepseek_v3/"

    # Prepend custom backend paths if defined
    for backend in "${CUSTOM_BACKENDS[@]}"; do
        custom_path="${CUSTOM_BACKEND_PATHS[$backend]}"
        [[ -n "$custom_path" ]] && full_pythonpath="${custom_path}:${full_pythonpath}"
    done

    export PYTHONPATH="${full_pythonpath}:${PYTHONPATH}"
}

setup_pythonpath

LOG_INFO "prepare dataset"
PRIMUS_PATCH_ARGS_FILE=$(mktemp /tmp/primus_patch_args.XXXXXX.yaml)
trap 'rm -f "$PRIMUS_PATCH_ARGS_FILE"' EXIT

SCRIPT="$PRIMUS_PATH/examples/scripts/prepare_experiment.py"
if ! python3 "$SCRIPT" --exp "$EXP" --data_path "$DATA_PATH" --patch_args "$PRIMUS_PATCH_ARGS_FILE"; then
    LOG_ERROR "$SCRIPT failed, aborting."
    exit 1
fi

# ---------- Parse optional patch args ----------
TRAIN_EXTRA_ARGS=""
TORCHRUN_EXTRA_ARGS=""

if [[ -f "$PRIMUS_PATCH_ARGS_FILE" ]]; then
    LOG_INFO_RANK0 "Loading patch args from $PRIMUS_PATCH_ARGS_FILE"
    source_yaml_args() {
        local file=$1
        local key=$2
        grep -E "^${key}:" "$file" | cut -d':' -f2- | xargs
    }

    TRAIN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" train_args)
    TORCHRUN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" torchrun_args)

    if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TRAIN args: $TRAIN_EXTRA_ARGS"
    fi

    if [[ -n "$TORCHRUN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TORCHRUN args: $TORCHRUN_EXTRA_ARGS"
    fi
else
    LOG_INFO_RANK0 "No patch args file found at $PRIMUS_PATCH_ARGS_FILE, skipping patch args."
fi

# -------------------- Launch Training --------------------
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)


CMD="torchrun ${DISTRIBUTED_ARGS[*]} $TORCHRUN_EXTRA_ARGS primus/train.py --config $EXP $TRAIN_EXTRA_ARGS $*"

LOG_INFO "Launching distributed training with command: $CMD"

eval "$CMD" 2>&1 | tee "$TRAIN_LOG"
exit_code=${PIPESTATUS[0]}

if [ "${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}" -eq 1 ]; then
    LOG_INFO "[PRIMUS_HIPBLASLT_TUNING_STAGE-1]: HipBlasLT gemm shape dump is finished, " \
         "please set PRIMUS_HIPBLASLT_TUNING_STAGE to 2, " \
         "and tune the gemm with a single node."
fi

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
