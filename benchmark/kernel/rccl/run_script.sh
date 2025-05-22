#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export HSA_ENABLE_SDMA=0
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens51f0
export GLOO_SOCKET_IFNAME=ens51f0
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Reducing to 1 ensures no PCIE traffic (even on single node)
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
#export NCCL_DEBUG=INFO

PRIMUS_ROOT_PATH="$(pwd)/../../.."
MEGATRON_PATH="$PRIMUS_ROOT_PATH/third_party/Megatron-LM"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

# Setting
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}

#########################################################################
torchrun --master_addr "$MASTER_ADDR"   \
        --master_port "$MASTER_PORT"    \
        --nnodes=1                      \
        --node_rank=0                   \
        --nproc_per_node=8              \
    ./benchmark_all2all.py --report-csv-path ./all2all_benchmark.csv

torchrun --nproc_per_node=2             \
    ./benchmark_send_recv.py --report-csv-path ./send_recv_benchmark.csv

torchrun --master_addr "$MASTER_ADDR"   \
        --master_port "$MASTER_PORT"    \
        --nnodes=1                      \
        --node_rank=0                   \
        --nproc_per_node=8              \
    ./benchmark_allreduce.py            \
        --allreduce-report-csv-path ./allreduce_benchmark.csv \
        --allgather-report-csv-path ./allgather_benchmark.csv \
        --reducescatter-report-csv-path ./reducescatter_benchmark.csv

#########################################################################
