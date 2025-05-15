#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

MODEL_CONFIG=${MODEL_CONFIG:-llama2_7B}
DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}


BENCH_WORK_PATH=$(pwd)
PRIMUS_PATH="$BENCH_WORK_PATH/../../"
# PRIMUS_PATH=$(pwd)
# BENCH_WORK_PATH="$PRIMUS_PATH/benchmark/megatron"

MODEL_LOG_DIR="${BENCH_WORK_PATH}/logs/${MODEL_CONFIG}"
mkdir -p "$MODEL_LOG_DIR"

##################################################
# TODO: more model
ITERS=30
if [[ "$MODEL_CONFIG" == "llama2_7B" ]]; then
    DP=8
    TP=1
    EP=1
    PP=1
    MBS=6
    GBS=384
    NNODES=1
    SEQ_LENGTH=4096
    YAML_PATH="$BENCH_WORK_PATH/configs/benchmark_pretrain_base.yaml"

    echo "▶️ Running with MODEL=$MODEL_CONFIG, NNODES=$NNODES, TP=$TP, PP=$PP, EP=$EP, MBS=$MBS, GBS=$GBS, SEQ_LENGTH=$SEQ_LENGTH, ITERS=$ITERS"
    cd "$PRIMUS_PATH" || exit

    LOGFILE="${MODEL_LOG_DIR}/nodes${NNODES}_dp${DP}_tp${TP}_pp${PP}_ep${EP}_mbs${MBS}_gbs${GBS}_seqlen${SEQ_LENGTH}_iters${ITERS}.log"

    MODEL_CONFIG=$MODEL_CONFIG                                      \
        DATA_PATH=$DATA_PATH                                        \
        NNODES=$NNODES PRIMUS_TP=$TP PRIMUS_EP=$EP PRIMUS_PP=$PP    \
        PRIMUS_MBS=$MBS PRIMUS_GBS=$GBS                             \
        PRIMUS_SEQ_LENGTH=$SEQ_LENGTH PRIMUS_ITERS=$ITERS           \
        EXP=$YAML_PATH                                              \
        TRAIN_LOG=$LOGFILE                                          \
        bash ./examples/megatron/run_local_pretrain.sh

elif [[ "$MODEL_CONFIG" == "llama2_70B" ]]; then
    DP=8
    TP=1
    EP=1
    PP=1
    MBS=4   # 4
    GBS=256 # 256
    NNODES=1
    SEQ_LENGTH=4096
    RECOMPUTE_GRANULARITY=full
    RECOMPUTE_METHOD=block
    RECOMPUTE_NUM_LAYERS=80
    YAML_PATH="$BENCH_WORK_PATH/configs/benchmark_pretrain_fsdp.yaml"
    CUDA_DEVICE_MAX_CONNECTIONS=8

    echo "▶️ Running with MODEL=$MODEL_CONFIG, NNODES=$NNODES, TP=$TP, PP=$PP, EP=$EP, MBS=$MBS, GBS=$GBS, SEQ_LENGTH=$SEQ_LENGTH, ITERS=$ITERS"
    cd "$PRIMUS_PATH" || exit

    LOGFILE="${MODEL_LOG_DIR}/nodes${NNODES}_dp${DP}_tp${TP}_pp${PP}_ep${EP}_mbs${MBS}_gbs${GBS}_seqlen${SEQ_LENGTH}_iters${ITERS}.log"

    CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS        \
        MODEL_CONFIG=$MODEL_CONFIG                                  \
        DATA_PATH=$DATA_PATH                                        \
        NNODES=$NNODES PRIMUS_TP=$TP PRIMUS_EP=$EP PRIMUS_PP=$PP    \
        PRIMUS_MBS=$MBS PRIMUS_GBS=$GBS                             \
        PRIMUS_SEQ_LENGTH=$SEQ_LENGTH PRIMUS_ITERS=$ITERS           \
        PRIMUS_RECOMPUTE_GRANULARITY=$RECOMPUTE_GRANULARITY         \
        PRIMUS_RECOMPUTE_METHOD=$RECOMPUTE_METHOD                   \
        PRIMUS_RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS           \
        EXP=$YAML_PATH                                              \
        TRAIN_LOG=$LOGFILE                                          \
        bash ./examples/megatron/run_local_pretrain.sh

elif [[ "$MODEL_CONFIG" == "llama3_8B" ]]; then
    DP=8
    TP=1
    EP=1
    PP=1
    MBS=3
    GBS=192
    NNODES=1
    SEQ_LENGTH=8192
    YAML_PATH="$BENCH_WORK_PATH/configs/benchmark_pretrain_base.yaml"

    echo "▶️ Running with MODEL=$MODEL_CONFIG, NNODES=$NNODES, TP=$TP, PP=$PP, EP=$EP, MBS=$MBS, GBS=$GBS, SEQ_LENGTH=$SEQ_LENGTH, ITERS=$ITERS"
    cd "$PRIMUS_PATH" || exit

    LOGFILE="${MODEL_LOG_DIR}/nodes${NNODES}_dp${DP}_tp${TP}_pp${PP}_ep${EP}_mbs${MBS}_gbs${GBS}_seqlen${SEQ_LENGTH}_iters${ITERS}.log"

    MODEL_CONFIG=$MODEL_CONFIG                                      \
        DATA_PATH=$DATA_PATH                                        \
        NNODES=$NNODES PRIMUS_TP=$TP PRIMUS_EP=$EP PRIMUS_PP=$PP    \
        PRIMUS_MBS=$MBS PRIMUS_GBS=$GBS                             \
        PRIMUS_SEQ_LENGTH=$SEQ_LENGTH PRIMUS_ITERS=$ITERS           \
        PRIMUS_MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH                  \
        EXP=$YAML_PATH                                              \
        TRAIN_LOG=$LOGFILE                                          \
        bash ./examples/megatron/run_local_pretrain.sh

elif [[ "$MODEL_CONFIG" == "llama3_70B" ]]; then
    DP=8
    TP=1
    EP=1
    PP=1
    MBS=2
    GBS=128
    NNODES=1
    SEQ_LENGTH=8192
    RECOMPUTE_GRANULARITY=full
    RECOMPUTE_METHOD=block
    RECOMPUTE_NUM_LAYERS=80
    YAML_PATH="$BENCH_WORK_PATH/configs/benchmark_pretrain_fsdp.yaml"
    CUDA_DEVICE_MAX_CONNECTIONS=8

    echo "▶️ Running with MODEL=$MODEL_CONFIG, NNODES=$NNODES, TP=$TP, PP=$PP, EP=$EP, MBS=$MBS, GBS=$GBS, SEQ_LENGTH=$SEQ_LENGTH, ITERS=$ITERS"
    cd "$PRIMUS_PATH" || exit

    LOGFILE="${MODEL_LOG_DIR}/nodes${NNODES}_dp${DP}_tp${TP}_pp${PP}_ep${EP}_mbs${MBS}_gbs${GBS}_seqlen${SEQ_LENGTH}_iters${ITERS}.log"

    CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS        \
        MODEL_CONFIG=$MODEL_CONFIG                                  \
        DATA_PATH=$DATA_PATH                                        \
        NNODES=$NNODES PRIMUS_TP=$TP PRIMUS_EP=$EP PRIMUS_PP=$PP    \
        PRIMUS_MBS=$MBS PRIMUS_GBS=$GBS                             \
        PRIMUS_SEQ_LENGTH=$SEQ_LENGTH PRIMUS_ITERS=$ITERS           \
        PRIMUS_MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH                  \
        PRIMUS_RECOMPUTE_GRANULARITY=$RECOMPUTE_GRANULARITY         \
        PRIMUS_RECOMPUTE_METHOD=$RECOMPUTE_METHOD                   \
        PRIMUS_RECOMPUTE_NUM_LAYERS=$RECOMPUTE_NUM_LAYERS           \
        EXP=$YAML_PATH                                              \
        TRAIN_LOG=$LOGFILE                                          \
        bash ./examples/megatron/run_local_pretrain.sh
        
else
    echo "Unknown MODEL_CONFIG: $MODEL_CONFIG"
    exit 1
fi
##################################################
