#!/bin/bash
# shellcheck disable=all

# export HF_TOKEN=""
export NCCL_IB_HCA=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export CPUS_PER_TASK=96
export HSA_NO_SCRATCH_RECLAIM=1 # change to 0
export NVTE_CK_USES_BWD_V3=1 # change to 0
export CLEAN_DOCKER_CONTAINER=1

# export EXP=examples/megatron/configs/llama3.1_405B-pretrain.yaml


ALL_NODES=(
pdfc-aig-000010
pdfc-aig-000011
pdfc-aig-000012
pdfc-aig-000013
pdfc-aig-000014
pdfc-aig-000015
pdfc-aig-000016
pdfc-aig-000017
)

#==============================================================================================70b_fsdp2
# export EXP=examples/megatron/configs/llama3.1_70B-pretrain-wenx.yaml
# export NNODES=8

# SELECTED_NODES=("${ALL_NODES[@]:0:$NNODES}")
# export NODELIST=$(IFS=, ; echo "${SELECTED_NODES[*]}")

# MBS=1
# GBS=256
# TP=1
# ETP=1
# PP=1
# EP=1
# CP=1
# VPP=1
# TOPK=8
# ITER=10
# OPTIMIZER=adam
# RECOMPUTE_LAYERS=12
# CONFIG="Llama3.1_70B-fsdp2-GC-TRUE.BF16.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.nodes$NNODES.$OPTIMIZER.ITER$ITER"
# echo "config: $CONFIG"


# if [ $VPP -gt 1 ]; then
#     export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
# fi

# export PRIMUS_WORKSPACE=output/llama_overlap
# export PRIMUS_USER=wenx
# export PRIMUS_GROUP="date-$(date +%Y%m%d)"
# export PRIMUS_EXP_NAME=$CONFIG
# mkdir -p $PRIMUS_WORKSPACE

# LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
# mkdir -p $LOG_DIR
# LOG_FILE=$LOG_DIR/training.log
# echo $LOG_FILE

# export PRIMUS_PPDUMP_FILE=$LOG_DIR/pp_dump

# # tune gemm
# export HIPBLASLT_TUNING_OVERRIDE_FILE=gemm_tuning_results/gemm_tune_result.txt

# EXPORT_CONFIG=$LOG_DIR/config.yaml
# bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
#                                       --global_batch_size $GBS \
#                                       --tensor_model_parallel_size $TP \
#                                       --pipeline_model_parallel_size $PP \
#                                       --mock_data True \
#                                       --manual_gc True \
#                                       --manual_gc_interval 1 \
#                                       --context_parallel_size $CP \
#                                       --optimizer $OPTIMIZER \
#                                       --recompute_num_layers $RECOMPUTE_LAYERS \
#                                       --dump_pp_data False \
#                                       --profile True \
#                                       --disable_profiler_activity_cpu False \
#                                       --use_pytorch_profiler True \
#                                       --profile_step_start 4 \
#                                       --profile_step_end 5 \
#                                       ${VPP_CONFIG} \
#                                       --train_iters $ITER 2>&1 | tee $LOG_FILE
#                                     #   --cp_comm_type a2a \


#==============================================================================================70b_tpppvpp
export EXP=examples/megatron/configs/llama3.1_70B-pretrain-wenx-tppp.yaml
export NNODES=8

SELECTED_NODES=("${ALL_NODES[@]:0:$NNODES}")
export NODELIST=$(IFS=, ; echo "${SELECTED_NODES[*]}")

MBS=1
GBS=256
TP=2
ETP=1
PP=4
EP=1
CP=1
VPP=5
TOPK=8
ITER=8
NUM_LAYERS=80
OPTIMIZER=adam
TP_COMM_OVERLAP=True
RECOMPUTE_LAYERS=0
CONFIG="Llama3.1_70B-tppp-Layer${NUM_LAYERS}-GC-TRUE.BF16.MBS$MBS.GBS$GBS.TP$TP.ASyncTP${TP_COMM_OVERLAP}.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.nodes$NNODES.$OPTIMIZER.ITER$ITER"
echo "config: $CONFIG"


if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

export PRIMUS_WORKSPACE=output/llama_overlap
export PRIMUS_USER=wenx
export PRIMUS_GROUP="date-$(date +%Y%m%d)"
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE

LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

export PRIMUS_PPDUMP_FILE=$LOG_DIR/pp_dump

# tune gemm
# export HIPBLASLT_TUNING_OVERRIDE_FILE=gemm_tuning_results/gemm_tune_result.txt

EXPORT_CONFIG=$LOG_DIR/config.yaml
bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --pipeline_model_parallel_size $PP \
                                      --tp_comm_overlap ${TP_COMM_OVERLAP} \
                                      --num_layers ${NUM_LAYERS} \
                                      --mock_data True \
                                      --manual_gc True \
                                      --manual_gc_interval 1 \
                                      --context_parallel_size $CP \
                                      --optimizer $OPTIMIZER \
                                      --recompute_num_layers $RECOMPUTE_LAYERS \
                                      --dump_pp_data False \
                                      --profile True \
                                      --disable_profiler_activity_cpu False \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 4 \
                                      --profile_step_end 5 \
                                      ${VPP_CONFIG} \
                                      --train_iters $ITER 2>&1 | tee $LOG_FILE
                                    #   --fp8 hybrid \
                                    #   --cp_comm_type a2a \
