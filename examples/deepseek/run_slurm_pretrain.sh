#!/bin/bash
# shellcheck disable=SC2086

# salloc --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation --exclusive --mem=0 -N 8
# salloc --nodelist=gpu-56 --exclusive --mem=0 -N 8

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
echo "Current script path: $SCRIPT_DIR"

export RUN_ENV=slurm
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

srun -N 2 \
     --gres=gpu:8 \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=64 \
     bash ${SCRIPT_DIR}/run_pretrain.sh
