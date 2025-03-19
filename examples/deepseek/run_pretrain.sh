#!/bin/bash

# python path
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export MEGATRON_PATH=${PRIMUS_PATH}/../Megatron-LM
export PYTHONPATH=${SITE_PACKAGES}:${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}

# check the path
[[ -z "${MEGATRON_PATH}" ]] && {
    echo "MEGATRON_PATH path is not set"
    exit 1
}
# build helper_cpp
pushd "${MEGATRON_PATH}/megatron/core/datasets" && make && popd || exit 1

# avaliable model configs:
# deepseek_v2_lite, deepseek_v2
# deepseek_v3, deepseek_v3_17B, deepseek_v3_45B
export MODEL_CONFIG=deepseek_v2_lite

# network envs
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export HSA_ENABLE_SDMA=0
export GLOO_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export AMD_LOG_LEVEL=3
# export AMD_SERIALIZE_KERNEL=3
# export HSA_NO_SCRATCH_RECLAIM=1

# cluster node envs
RUN_ENV="${RUN_ENV:-torchrun}"
if [ "$RUN_ENV" = "torchrun" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}
    export NNODES=${NNODES:-1}
    export NODE_RANK=${NODE_RANK:-0}
    export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
elif [ "$RUN_ENV" = "slurm" ]; then
    export MASTER_ADDR=${SLURM_MASTER_ADDR}
    export MASTER_PORT=${SLURM_MASTER_PORT}
    export NNODES=$SLURM_NNODES
    export NODE_RANK=${SLURM_NODEID}
    export GPUS_PER_NODE=$((SLURM_WORLD_SIZE / SLURM_NNODES))
    echo "Error: SLURM mode is not implemented yet!"
    exit 1
else
    echo "Error: Unknown RUN_ENV value: $RUN_ENV"
    exit 1
fi
gpus=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES=$gpus

echo "RUN_ENV: $RUN_ENV"
echo "PRIMUS_PATH: $PRIMUS_PATH"
echo "MEGATRON_PATH: $MEGATRON_PATH"
echo "SITE_PACKAGES: $SITE_PACKAGES"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
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

torchrun "${DISTRIBUTED_ARGS[@]}" examples/deepseek_v3/pretrain.py \
    --exp examples/deepseek_v3/exp_pretrain.yaml \
    2>&1 | tee $TRAIN_LOG
