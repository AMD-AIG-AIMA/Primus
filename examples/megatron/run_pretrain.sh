#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

# available models: primus/configs/models/megatron
export MODEL_CONFIG=${MODEL_CONFIG:-deepseek_v2_lite}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/third_party/Megatron-LM-20250324}
[[ ! -d "${MEGATRON_PATH}" || -z "$(ls -A "${MEGATRON_PATH}")" ]] && {
    echo "Error: MEGATRON_PATH (${MEGATRON_PATH}) does not exist or is empty"
    exit 1
}

# model config
export MODEL_CONFIG_FILE=$PRIMUS_PATH/primus/configs/models/megatron/${MODEL_CONFIG}.yaml
TOKENIZER_TYPE=$(grep "^tokenizer_type:" "$MODEL_CONFIG_FILE" | awk -F ': ' '{print $2}')
export TOKENIZER_TYPE
TOKENIZER_MODEL=$(grep "^tokenizer_model:" "$MODEL_CONFIG_FILE" | awk -F ': ' '{print $2}')
export TOKENIZER_MODEL
if [[ ! -f "${MODEL_CONFIG_FILE}" ]]; then
    echo "Error: Missing model config file: $MODEL_CONFIG_FILE."
    exit 1
fi

# env
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=1
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
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_DEBUG="" # VERSION, WARN, INFO, DEBUG

export DATA_PATH=${DATA_PATH:-"/apps/tas/0_public/data"}
export HF_HOME=${HF_HOME:-"${DATA_PATH}"/huggingface}
export TOKENIZED_DATA_PATH=${TOKENIZED_DATA_PATH:-${DATA_PATH}/bookcorpus/${TOKENIZER_TYPE}/bookcorpus_text_sentence}
if [ "$NODE_RANK" = "0" ]; then
    # dataset
    if [[ ! -f "${TOKENIZED_DATA_PATH}.done" ]]; then
        echo "prepare dataset..."
        ${HF_TOKEN:?Environment variable HF_TOKEN must be set}
        bash ./examples/scripts/prepare_dataset.sh ${DATA_PATH} ${TOKENIZER_TYPE} ${TOKENIZER_MODEL}
        touch ${TOKENIZED_DATA_PATH}.done
        echo "prepate dataset success..."
    fi
else
    while [[ ! -f "${TOKENIZED_DATA_PATH}.done" ]]; do
        echo "[NODE-$NODE_RANK] Waiting for dataset to be ready..."
        sleep 30
    done
fi

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training cluster info=========="
    echo "[NODE-$NODE_RANK] MASTER_ADDR: $MASTER_ADDR"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NNODES: $NNODES"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] GPUS_PER_NODE: $GPUS_PER_NODE"
    echo "[NODE-$NODE_RANK] HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
    echo ""

    echo "==========Training experiment info=========="
    echo "[NODE-$NODE_RANK] MODEL_CONFIG: $MODEL_CONFIG"
    echo "[NODE-$NODE_RANK] PRIMUS_PATH: $PRIMUS_PATH"
    echo "[NODE-$NODE_RANK] MEGATRON_PATH: $MEGATRON_PATH"
    echo "[NODE-$NODE_RANK] HF_HOME: $HF_HOME"
    echo "[NODE-$NODE_RANK] TOKENIZED_DATA_PATH: $TOKENIZED_DATA_PATH"
    echo "[NODE-$NODE_RANK] MODEL_CONFIG_FILE: $MODEL_CONFIG_FILE"
    echo "[NODE-$NODE_RANK] TOKENIZER_TYPE: $TOKENIZER_TYPE"
    echo "[NODE-$NODE_RANK] TOKENIZER_MODEL: $TOKENIZER_MODEL"
    echo "[NODE-$NODE_RANK] RUN_ENV: $RUN_ENV"
    echo ""
else
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
    echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
fi

# gemm tuning, https://github.com/ROCm/TransformerEngine
export TE_HIPBLASLT_TUNING=0
if [ "$TE_HIPBLASLT_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
    unset TE_HIPBLASLT_TUNING_RUN_COUNT
    unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi
export NVTE_CK_USES_BWD_V3=1

# 0: default
# 1: dump gemm shape
# 2: tuning gemm kernel
# 3: train with tuned kernel
export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}

mkdir -p output
TUNE_HIPBLASLT_LOG_PATH=${PRIMUS_PATH}/output/tune_hipblaslt/${MODEL_CONFIG}
TUNE_HIPBLASLT_RESULT_FILE="tune_hipblas_gemm_results.txt"
if [ "$PRIMUS_HIPBLASLT_TUNING_STAGE" -eq 1 ]; then
    if [ "$TE_HIPBLASLT_TUNING" -eq 1 ]; then
        echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-1]: Please disable TE_HIPBLASLT_TUNING when dump shape."
        exit 1
    fi

    mkdir -p $TUNE_HIPBLASLT_LOG_PATH/gemm_shape
    DUMP_FILE=${TUNE_HIPBLASLT_LOG_PATH}/gemm_shape/dump_hipblaslt_gemm_shape_${NODE_RANK}.txt

    # Note that HIPBLASLT_TUNING_OVERRIDE_FILE must be unset; even if it's empty,
    # it will still disable the shape dump feature.
    unset HIPBLASLT_TUNING_OVERRIDE_FILE
    export HIPBLASLT_LOG_MASK=32
    export HIPBLASLT_LOG_FILE=${HIPBLASLT_LOG_FILE:-${DUMP_FILE}}
elif [ "$PRIMUS_HIPBLASLT_TUNING_STAGE" -eq 2 ]; then
    mkdir -p $TUNE_HIPBLASLT_LOG_PATH/gemm_tune

    python ${PRIMUS_PATH}/examples/offline_tune/offline_tune_gemm.py \
        --dump-shape-path-or-file ${TUNE_HIPBLASLT_LOG_PATH}/gemm_shape \
        --tune-result-path ${TUNE_HIPBLASLT_LOG_PATH}/gemm_tune/${TUNE_HIPBLASLT_RESULT_FILE} \
        --num-devices 8

    echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-2]: HipBlasLT gemm tuning is finished, " \
         "please set PRIMUS_HIPBLASLT_TUNING_STAGE to 3, " \
         "and rerun the pretrain with more nodes."
    exit 0
elif [ "$PRIMUS_HIPBLASLT_TUNING_STAGE" -eq 3 ]; then
    if [ "$TE_HIPBLASLT_TUNING" -eq 1 ]; then
        echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-3]: Please disable TE_HIPBLASLT_TUNING at the shape dump stage."
        exit 1
    fi

    TUNE_FILE=${TUNE_HIPBLASLT_LOG_PATH}/gemm_tune/${TUNE_HIPBLASLT_RESULT_FILE}

    unset HIPBLASLT_LOG_MASK
    unset HIPBLASLT_LOG_FILE
    # 0: nolog, 1: error, 2: warning, 3: debug, 4: trace
    export HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-0}
    export HIPBLASLT_TUNING_OVERRIDE_FILE=${HIPBLASLT_TUNING_OVERRIDE_FILE:-${TUNE_FILE}}

    if [[ ! -f "${HIPBLASLT_TUNING_OVERRIDE_FILE}" ]]; then
        echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-3]: Missing hipblaslt tuning result file: ${HIPBLASLT_TUNING_OVERRIDE_FILE}."
        exit 1
    fi
fi

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training tuning info=========="
    echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING: $TE_HIPBLASLT_TUNING"
    echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_RUN_COUNT: $TE_HIPBLASLT_TUNING_RUN_COUNT"
    echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_ALGO_COUNT: $TE_HIPBLASLT_TUNING_ALGO_COUNT"
    echo "[NODE-$NODE_RANK] NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
    echo "[NODE-$NODE_RANK] PRIMUS_HIPBLASLT_TUNING_STAGE: ${PRIMUS_HIPBLASLT_TUNING_STAGE}"
    echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_MASK: ${HIPBLASLT_LOG_MASK}"
    echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_FILE: ${HIPBLASLT_LOG_FILE}"
    echo "[NODE-$NODE_RANK] HIPBLASLT_LOG_LEVEL: ${HIPBLASLT_LOG_LEVEL}"
    echo "[NODE-$NODE_RANK] HIPBLASLT_TUNING_OVERRIDE_FILE: ${HIPBLASLT_TUNING_OVERRIDE_FILE}"
    if [ "$PRIMUS_HIPBLASLT_TUNING_STAGE" -eq 1 ]; then
        echo "[NODE-$NODE_RANK] Dump HipBLASLt shapes, make sure train_iters is set to a very small value."
    fi
    echo ""
fi


TRAIN_LOG=output/log_torchrun_pretrain_${MODEL_CONFIG}.txt
if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training logging info=========="
    echo "[NODE-$NODE_RANK] TRAIN_LOG: $TRAIN_LOG"
    echo ""
fi

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=${SITE_PACKAGES}:${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

# build helper_cpp of megatron
pushd "${MEGATRON_PATH}/megatron/core/datasets" && make && popd || exit 1
# env

torchrun "${DISTRIBUTED_ARGS[@]}" examples/megatron/pretrain.py \
    --exp examples/megatron/exp_pretrain.yaml 2>&1 | tee $TRAIN_LOG

if [ "$PRIMUS_HIPBLASLT_TUNING_STAGE" -eq 1 ]; then
    echo "[PRIMUS_HIPBLASLT_TUNING_STAGE-1]: HipBlasLT gemm shape dump is finished, " \
         "please set PRIMUS_HIPBLASLT_TUNING_STAGE to 2, " \
         "and tune the gemm with a single node."
fi
