#!/bin/bash
# shellcheck disable=SC2086


avaliable model configs:
# deepseek_v2_lite, deepseek_v2
# deepseek_v3, deepseek_v3_17B, deepseek_v3_45B
export MODEL_CONFIG=deepseek_v2_lite
echo "MODEL_CONFIG: $MODEL_CONFIG"

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/../Megatron-LM}
echo "PRIMUS_PATH: $PRIMUS_PATH"
echo "MEGATRON_PATH: $MEGATRON_PATH"

# check megatron path
[[ -z "${MEGATRON_PATH}" ]] && {
    echo "MEGATRON_PATH path is not set"
    exit 1
}

# data
mkdir -p "${PRIMUS_PATH}"/data/deepseek-datasets
export HF_HOME="${PRIMUS_PATH}"/data/huggingface
export DATA_PATH="${PRIMUS_PATH}"/data/deepseek-datasets/mmap_deepseekv2_datasets_text_document
echo "HF_HOME: $HF_HOME"
echo "DATA_PATH: $DATA_PATH"
if [[ ! -f "${DATA_PATH}.bin" || ! -f "${DATA_PATH}.idx" ]]; then
    echo "Error: Missing required deepseek files. \
          Please follow the README.md and download ${DATA_PATH}.bin and ${DATA_PATH}.idx."
    exit 1
fi

# network envs
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export HSA_ENABLE_SDMA=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
# export AMD_LOG_LEVEL=3
# export AMD_SERIALIZE_KERNEL=3
# export HSA_NO_SCRATCH_RECLAIM=1

export GEMM_TUNING=0
export NVTE_CK_USES_BWD_V3=1
echo "GEMM_TUNING: $GEMM_TUNING"
echo "NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"

# gemm tuning, https://github.com/ROCm/TransformerEngine
if [ "$GEMM_TUNING" -eq 1 ]; then
   export TE_HIPBLASLT_TUNING_RUN_COUNT=10
   export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
   unset TE_HIPBLASLT_TUNING_RUN_COUNT
   unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi

# cluster node envs
RUN_ENV="${RUN_ENV:-torchrun}"
echo "RUN_ENV: $RUN_ENV"
if [ "$RUN_ENV" = "torchrun" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
elif [ "$RUN_ENV" = "slurm" ]; then
    # use the first node as the master node
    node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t node_array <<< "$node_list"
    HEAD_NODE=${node_array[0]}

    export SLURM_MASTER_ADDR=$HEAD_NODE
    export SLURM_MASTER_PORT=29509
    export SLURM_WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

    echo "[NODE-$SLURM_NODEID] NODELIST=${node_array[*]}"
    echo "[NODE-$SLURM_NODEID] NODENAME=$SLURMD_NODENAME"
    echo "[NODE-$SLURM_NODEID] SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR"
    echo "[NODE-$SLURM_NODEID] SLURM_MASTER_PORT=$SLURM_MASTER_PORT"
    echo "[NODE-$SLURM_NODEID] SLURM_NNODES=$SLURM_NNODES"
    echo "[NODE-$SLURM_NODEID] SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
    echo "[NODE-$SLURM_NODEID] SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE"
    echo "[NODE-$SLURM_NODEID] SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
    echo "[NODE-$SLURM_NODEID] SLURM_PROCID: $SLURM_PROCID"

    export MASTER_ADDR=${SLURM_MASTER_ADDR}
    export MASTER_PORT=${SLURM_MASTER_PORT}
    export NNODES=${SLURM_NNODES}
    export NODE_RANK=${SLURM_NODEID}
    export GPUS_PER_NODE=$((SLURM_WORLD_SIZE / SLURM_NNODES))
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus

echo "[NODE-$NODE_RANK] MASTER_ADDR: $MASTER_ADDR"
echo "[NODE-$NODE_RANK] MASTER_PORT: $MASTER_PORT"
echo "[NODE-$NODE_RANK] NNODES: $NNODES"
echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
echo "[NODE-$NODE_RANK] GPUS_PER_NODE: $GPUS_PER_NODE"
echo "[NODE-$NODE_RANK] HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo ""

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

mkdir -p output
TRAIN_LOG=output/log_torchrun_pretrain_${MODEL_CONFIG}.txt

if [ "$RUN_ENV" = "torchrun" ]; then
    SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    export PYTHONPATH=${SITE_PACKAGES}:${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}

    # build helper_cpp of megatron
    pushd "${MEGATRON_PATH}/megatron/core/datasets" && make && popd || exit 1

    torchrun "${DISTRIBUTED_ARGS[@]}" examples/deepseek/pretrain.py \
        --exp examples/deepseek/exp_pretrain.yaml \
        2>&1 | tee $TRAIN_LOG

elif [ "$RUN_ENV" = "slurm" ]; then
    export DOCKER_IMAGE="docker.io/rocm/megatron-lm:latest"
    #   podman pull $DOCKER_IMAGE;
    echo "[NODE-$NODE_RANK] stop all podmann containers..."
    podman stop -a && \
    module load rocm && \
    podman run \
        --rm \
        --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
        --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
        --env SLURM_PROCID=$SLURM_PROCID \
        --env SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE \
        --env SLURM_NODEID=$SLURM_NODEID \
        --env SLURM_NNODES=$SLURM_NNODES \
        --env MASTER_ADDR=${MASTER_ADDR} \
        --env MASTER_PORT=${MASTER_PORT} \
        --env NNODES=${NNODES} \
        --env NODE_RANK=${NODE_RANK} \
        --env GPUS_PER_NODE=${GPUS_PER_NODE} \
        --env HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
        --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        --env GPU_MAX_HW_QUEUES=$GPU_MAX_HW_QUEUES \
        --env TORCH_NCCL_HIGH_PRIORITY=$TORCH_NCCL_HIGH_PRIORITY \
        --env NCCL_CHECKS_DISABLE=$NCCL_CHECKS_DISABLE \
        --env NCCL_IB_HCA=$NCCL_IB_HCA \
        --env NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX \
        --env NCCL_CROSS_NIC=$NCCL_CROSS_NIC \
        --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
        --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
        --env GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
        --env CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
        --env NCCL_PROTO=$NCCL_PROTO \
        --env RCCL_MSCCL_ENABLE=$RCCL_MSCCL_ENABLE \
        --env HF_HOME=$HF_HOME \
        --env DATA_PATH=$DATA_PATH \
        --env MODEL_CONFIG=$MODEL_CONFIG \
        --env TE_HIPBLASLT_TUNING_RUN_COUNT=$TE_HIPBLASLT_TUNING_RUN_COUNT \
        --env TE_HIPBLASLT_TUNING_ALGO_COUNT=$TE_HIPBLASLT_TUNING_ALGO_COUNT \
        --env NVTE_CK_USES_BWD_V3=$NVTE_CK_USES_BWD_V3 \
        --ipc=host --network=host \
        --device=/dev/kfd --device=/dev/dri  \
        --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
        --security-opt seccomp=unconfined --group-add video \
        --privileged --device=/dev/infiniband \
        -v $MEGATRON_PATH:$MEGATRON_PATH \
        -v $PRIMUS_PATH:$PRIMUS_PATH \
    $DOCKER_IMAGE /bin/bash -c \
        "echo $(date) && \
        pip install loguru wandb && \
        cd ${MEGATRON_PATH}/megatron/core/datasets && make && \
        cd $PRIMUS_PATH && \
        PYTHONPATH=${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH} \
        torchrun \
            --nproc_per_node ${GPUS_PER_NODE} \
            --nnodes ${NNODES} \
            --node_rank ${NODE_RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            examples/deepseek/pretrain.py \
            --exp examples/deepseek/exp_pretrain.yaml \
            2>&1 | tee $TRAIN_LOG && \
        echo $(date)"
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
