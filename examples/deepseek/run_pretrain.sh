#!/bin/bash
# shellcheck disable=SC2086

# available models: primus/configs/models/megatron
export MODEL_CONFIG=${MODEL_CONFIG:-deepseek_v2_lite}

# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/../Megatron-LM}
[[ ! -d "${MEGATRON_PATH}" ]] && {
    echo "Error: MEGATRON_PATH (${MEGATRON_PATH}) does not exist"
    exit 1
}

# data
mkdir -p "${PRIMUS_PATH}"/data/deepseek-datasets
export HF_HOME="${PRIMUS_PATH}"/data/huggingface
export DATA_PATH="${PRIMUS_PATH}"/data/deepseek-datasets/mmap_deepseekv2_datasets_text_document
if [[ ! -f "${DATA_PATH}.bin" || ! -f "${DATA_PATH}.idx" ]]; then
    echo "Error: Missing required deepseek files. \
          Please follow the README.md and download ${DATA_PATH}.bin and ${DATA_PATH}.idx."
    exit 1
fi

# cluster envs
RUN_ENV="${RUN_ENV:-torchrun}"
if [ "$RUN_ENV" = "torchrun" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
elif [ "$RUN_ENV" = "slurm" ]; then
    node_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    mapfile -t node_array <<<"$node_list"
    HEAD_NODE=${node_array[0]}
    random_port=$(shuf -i 1024-65535 -n 1)

    export SLURM_MASTER_ADDR=$HEAD_NODE
    export SLURM_MASTER_PORT=$random_port
    export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-8}
    export SLURM_WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

    if [ "$SLURM_NODEID" = "0" ]; then
        echo "==========Slurm cluster info=========="
        echo "[SLURM-NODE-$SLURM_NODEID] NODELIST=${node_array[*]}"
        echo "[SLURM-NODE-$SLURM_NODEID] NODENAME=$SLURMD_NODENAME"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_MASTER_PORT=$SLURM_MASTER_PORT"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_NNODES=$SLURM_NNODES"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_WORLD_SIZE=$SLURM_WORLD_SIZE"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
        echo "[SLURM-NODE-$SLURM_NODEID] SLURM_PROCID: $SLURM_PROCID"
        echo ""
    fi

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

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training experiment info=========="
    echo "[NODE-$NODE_RANK] MODEL_CONFIG: $MODEL_CONFIG"
    echo "[NODE-$NODE_RANK] PRIMUS_PATH: $PRIMUS_PATH"
    echo "[NODE-$NODE_RANK] MEGATRON_PATH: $MEGATRON_PATH"
    echo "[NODE-$NODE_RANK] HF_HOME: $HF_HOME"
    echo "[NODE-$NODE_RANK] DATA_PATH: $DATA_PATH"
    echo "[NODE-$NODE_RANK] RUN_ENV: $RUN_ENV"
    echo ""
fi

export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export HSA_ENABLE_SDMA=0
NCCL_IB_HCA=$(bash "${PRIMUS_PATH}"/examples/scripts/get_nccl_ib_hca.sh)
export NCCL_IB_HCA
IP_INTERFACE=$(bash "${PRIMUS_PATH}"/examples/scripts/get_ip_interface.sh)
export IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IP_INTERFACE}}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${IP_INTERFACE}}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export NCCL_CHECKS_DISABLE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1

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
else
    echo "[NODE-$NODE_RANK] NCCL_IB_HCA: $NCCL_IB_HCA"
    echo "[NODE-$NODE_RANK] IP_INTERFACE: $IP_INTERFACE"
    echo "[NODE-$NODE_RANK] NODE_RANK: $NODE_RANK"
fi

# gemm tuning, https://github.com/ROCm/TransformerEngine
export GEMM_TUNING=0
if [ "$GEMM_TUNING" -eq 1 ]; then
    export TE_HIPBLASLT_TUNING_RUN_COUNT=10
    export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
    unset TE_HIPBLASLT_TUNING_RUN_COUNT
    unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi
export NVTE_CK_USES_BWD_V3=1

if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training tuning info=========="
    echo "[NODE-$NODE_RANK] GEMM_TUNING: $GEMM_TUNING"
    echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_RUN_COUNT: $TE_HIPBLASLT_TUNING_RUN_COUNT"
    echo "[NODE-$NODE_RANK] TE_HIPBLASLT_TUNING_ALGO_COUNT: $TE_HIPBLASLT_TUNING_ALGO_COUNT"
    echo "[NODE-$NODE_RANK] NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
    echo ""
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

mkdir -p output
TRAIN_LOG=output/log_torchrun_pretrain_${MODEL_CONFIG}.txt
if [ "$NODE_RANK" = "0" ]; then
    echo "==========Training logging info=========="
    echo "[NODE-$NODE_RANK] TRAIN_LOG: $TRAIN_LOG"
    echo ""
fi

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
    bash "${PRIMUS_PATH}"/examples/scripts/docker_podman_proxy.sh pull $DOCKER_IMAGE && \
    bash "${PRIMUS_PATH}"/examples/scripts/docker_podman_proxy.sh run --rm \
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
        --device=/dev/kfd --device=/dev/dri \
        --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
        --security-opt seccomp=unconfined --group-add video \
        --privileged --device=/dev/infiniband \
        -v $MEGATRON_PATH:$MEGATRON_PATH \
        -v $PRIMUS_PATH:$PRIMUS_PATH \
        $DOCKER_IMAGE /bin/bash -c \
            "echo '[NODE-${NODE_RANK}]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
            pip install -q loguru wandb && \
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
            echo '[NODE-${NODE_RANK}]: end time=$(date +"%Y.%m.%d %H:%M:%S")'"
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
