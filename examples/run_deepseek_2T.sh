export NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export CPUS_PER_TASK=96
export HSA_NO_SCRATCH_RECLAIM=1 # change to 0
export NVTE_CK_USES_BWD_V3=1 # change to 0

export EXP="examples/megatron/configs/deepseek_2T-pretrain.yaml"

ALL_NODES=(uswocpm2m-059-003)

# ALL_NNODES=69

# slurm number of nodes
ALL_NNODES=200

# the real number of nodes to run
export NNODES=96

# SELECTED_NODES=("${ALL_NODES[@]:8:$NNODES}")
# export NODELIST=$(IFS=, ; echo "${SELECTED_NODES[*]}")

export NODELIST=uswocpm2m-059-[007,027,041-042,047,049,060,062,065,069-070,074,076-078,089,104,110,118,121-122,124,132-133,138,140-141,144-145,149,151,159,172,175,181,186,188,192,195-196,200-201,209-210,217,219,230,233-234,236,263,275,281,304,306,309,313,330,332,335,351,366,396,410,418,421,442-444,446,457,465,483,500-503,511,518,520,564,567,571,578-579,605,621,631,639,650,660,666-667,677,681,684]

MBS=1
TP=1
ETP=1

GBS=768
PP=24
EP=8
CP=1
VPP=4
OPTIMIZER=adam
RECOMPUTE_LAYERS=96
RECOMPUTE_ID_START=0
BALANCE=True


CONFIG="Turbo-attn-gg-nodeepep.turbo.mockdata.gc.BF16.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE"
echo "config: $CONFIG"


if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

export PRIMUS_WORKSPACE=output/deepseek/2T
export PRIMUS_USER=qyy
export PRIMUS_GROUP="date-$(date +%Y%m%d-%H%M%S)"
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE


LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
export DUMP_PP_DIR=$LOG_DIR/pp_dump/
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
                                      --context_parallel_size $CP \
                                      --moe_router_force_load_balancing $BALANCE \
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
                                      --num_layers 96 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE

                                    #   --recompute_layer_ids_start $RECOMPUTE_ID_START \
                                    #   --moe_permute_fusion True \
                                    #   --use_turbo_token_dispatcher_fp8_alltoall False \
