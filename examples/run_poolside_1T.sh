export NCCL_IB_HCA=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export CPUS_PER_TASK=96
export HSA_NO_SCRATCH_RECLAIM=1 # change to 0
export NVTE_CK_USES_BWD_V3=1 # change to 0

export EXP="examples/megatron/configs/deepseek_1T-pretrain.yaml"

# ALL_NODES=(
# pdfc-aig-000000
# pdfc-aig-000001
# pdfc-aig-000002
# pdfc-aig-000003
# pdfc-aig-000004
# pdfc-aig-000005
# pdfc-aig-000006
# pdfc-aig-000007
# pdfc-aig-000008
# pdfc-aig-000009
# pdfc-aig-000010
# pdfc-aig-000011
# pdfc-aig-000012
# pdfc-aig-000013
# pdfc-aig-000014
# pdfc-aig-000015
# pdfc-aig-000016
# pdfc-aig-000017
# pdfc-aig-000018
# pdfc-aig-000019
# pdfc-aig-000020
# pdfc-aig-000021
# pdfc-aig-000022
# pdfc-aig-000023
# pdfc-aig-000024
# pdfc-aig-000025
# pdfc-aig-000026
# pdfc-aig-000027
# pdfc-aig-000028
# pdfc-aig-000029
# pdfc-aig-00000A
# pdfc-aig-00000B
# pdfc-aig-00000C
# pdfc-aig-00000D
# pdfc-aig-00000E
# pdfc-aig-00000F
# pdfc-aig-00000G
# pdfc-aig-00000H
# pdfc-aig-00000I
# pdfc-aig-00000J
# pdfc-aig-00000K
# pdfc-aig-00000L
# pdfc-aig-00000M
# pdfc-aig-00000N
# pdfc-aig-00000O
# pdfc-aig-00000P
# pdfc-aig-00000Q
# pdfc-aig-00000R
# pdfc-aig-00000S
# pdfc-aig-00000T
# pdfc-aig-00000U
# pdfc-aig-00000V
# pdfc-aig-00000W
# pdfc-aig-00001J
# pdfc-aig-00001K
# pdfc-aig-00001L
# pdfc-aig-00001M
# pdfc-aig-00001N
# pdfc-aig-00001O
# pdfc-aig-00001P
# pdfc-aig-00001Q
# pdfc-aig-00001R
# pdfc-aig-00001S
# pdfc-aig-00001T
# pdfc-aig-00001U
# pdfc-aig-00001V
# pdfc-aig-00001W
# pdfc-aig-00001X
# )

# ALL_NNODES=69

ALL_NODES=(
pdfc-aig-000000
pdfc-aig-000001
pdfc-aig-000003
pdfc-aig-000004
pdfc-aig-000005
pdfc-aig-000006
pdfc-aig-000007
pdfc-aig-000008
)

# slurm number of nodes
ALL_NNODES=8

# the real number of nodes to run
export NNODES=8

SELECTED_NODES=("${ALL_NODES[@]:0:$ALL_NNODES}")
export NODELIST=$(IFS=, ; echo "${SELECTED_NODES[*]}")

MBS=1
TP=1
ETP=1

GBS=128
PP=8
EP=8
VPP=1
TOPK=8
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0
BALANCE=True


CONFIG="Turbo-attn-gg-deepep.mockdata.gc.BF16.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE"
echo "config: $CONFIG"


if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

export PRIMUS_WORKSPACE=output/poolside/1T
export PRIMUS_USER=wenx
export PRIMUS_GROUP="date-$(date +%Y%m%d)"
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE


LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
export PRIMUS_PPDUMP_FILE=$LOG_DIR/pp_dump/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml
bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --expert_tensor_parallel_size $ETP \
                                      --pipeline_model_parallel_size $PP \
                                      --expert_model_parallel_size $EP \
                                      --moe_router_force_load_balancing $BALANCE \
                                      --moe_router_topk $TOPK \
                                      --manual_gc True \
                                      --manual_gc_interval 1 \
                                      --optimizer $OPTIMIZER \
                                      --cp_comm_type a2a \
                                      --recompute_num_layers $RECOMPUTE_LAYERS \
                                      --moe_use_legacy_grouped_gemm True \
                                      --use_turbo_token_dispatcher_fp8_alltoall null \
                                      --dump_pp_data True \
                                      --moe_router_num_groups 1 \
                                      --moe_router_group_topk 1 \
                                      ${VPP_CONFIG} \
                                      --attn_warmup True \
                                      --profile True \
                                      --record_shapes True \
                                      --disable_profiler_activity_cpu False \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 5 \
                                      --profile_step_end 6 \
                                      --export-config $EXPORT_CONFIG \
                                      --num_layers 32 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE

                                    #   --recompute_layer_ids_start $RECOMPUTE_ID_START \
                                    #   --moe_permute_fusion True \
                                    #   --use_turbo_token_dispatcher_fp8_alltoall False \
