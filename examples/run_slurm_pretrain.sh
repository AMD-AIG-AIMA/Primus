#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

#SBATCH --job-name=mlperf-andyye              # Job name
#SBATCH --nodes=7                        # Number of nodes
#SBATCH --ntasks-per-node=1               # One task (process) per node
#SBATCH --cpus-per-task=224                # Adjust based on your node's CPU count
#SBATCH --gres=gpu:8                      # Assuming 8 GPUs per node, adjust if different
#SBATCH --mem=0                           # Use all available memory
#SBATCH --time=00-03:00:00                   # Maximum runtime in DD-HH:MM:SS
#SBATCH --output=slurm_log/%x-%j.out                # Standard output log
#SBATCH --error=slurm_log/%x-%j.err                 # Standard error log
#SBATCH --partition=amd-rccl
#SBATCH --account=amd-rccl
#SBATCH --exclusive 
#SBATCH --exclude=useocpm2m-401-[086,036,037,028,067-069,052,082,102,105,106]
#	#SBATCH --nodelist=useocpm2m-401-[075,122-127] # add this if you need a fixed groups of nodes 


if [[ "$1" == "--help" || "$1" == "-h" ]]; then
cat <<EOF
Usage: run_slurm_pretrain.sh

Launches a Primus distributed pretraining task on a Slurm cluster using Docker.

Requirements:
  - Slurm job scheduler with 'srun'
  - Docker or Podman runtime (for container execution)

Optional Environment Variables:
  NNODES          Number of nodes to use [default: 1]
  MASTER_PORT     Master port [default: 12345]
  LOG_DIR         Directory for log output [default: ./output]

Example:
  export DATA_PATH=/mnt/data
  export EXP=examples/megatron/exp_pretrain.yaml
  NNODES=2 bash run_slurm_pretrain.sh
EOF
exit 0
fi

export MASTER_PORT=${MASTER_PORT:-12345}
export NNODES=${NNODES:-8}

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

export LOG_DIR=${LOG_DIR:-"./output"}
LOG_FILE="${LOG_DIR}/log_slurm_pretrain.txt"
mkdir -p "$LOG_DIR"

srun -N "${NNODES}" \
     --exclusive \
     --ntasks-per-node=1 \
     --cpus-per-task=128 \
     -t 03:00:00 \
     --exclude=useocpm2m-401-[086,036,037,028,067,068,069,052,080,102,123,124,125,005] \
     bash -c "
          CONTAINERS=\$(docker ps -a --format \"{{.Names}}\" || echo \"\")
          echo \"Containers...\"
          echo \${CONTAINERS}

          if [ -n \"\${CONTAINERS}\" ]; then
              echo \"Found containers to clean up: \${CONTAINERS}\"
              for container in \${CONTAINERS}; do
                  echo \"Stopping and removing container: \${container}\"
                  docker stop \${container} 2>/dev/null || true
                  docker rm \${container}
              done
          else
              echo \"No existing containers to clean up\"
          fi
          sleep 30
          echo \"Running docker ps\"
          docker ps
          echo \"Running docker ps -a\"
          docker ps -a

          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"========== Slurm cluster info ==========\"
              echo \"SLURM_NODELIST: \${node_array[*]}\"
              echo \"SLURM_NNODES: \${SLURM_NNODES}\"
              echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
              echo \"\"
          fi
          export MASTER_ADDR=\${node_array[0]}
          export MASTER_PORT=\${MASTER_PORT}
          export NNODES=\${SLURM_NNODES}
          export NODE_RANK=\${SLURM_PROCID}
          export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
          bash ${SCRIPT_DIR}/run_local_pretrain.sh \"\$@\" 2>&1 | tee ${LOG_FILE}
     " bash "$@"


# srun -N "${NNODES}" \
#      --exclusive \
#      --ntasks-per-node=1 \
#      --cpus-per-task=128 \
#      -t 03:00:00 \
#      --exclude=useocpm2m-401-[086,036,037,028,067,068,069,052,102,123,124,125,005] \
#      bash -c '
#      CONTAINERS=$(docker ps -a --format "{{.Names}}" || echo "")
#      echo "Containers..."
#      echo $CONTAINERS

#      if [ -n "$CONTAINERS" ]; then
#          echo "Found containers to clean up: $CONTAINERS"
#          for container in $CONTAINERS; do
#              echo "Stopping and removing container: $container"
#              docker stop $container 2>/dev/null || true
#              docker rm $container
#          done
#      else
#          echo "No existing containers to clean up"
#      fi
#      sleep 20
#      echo "Running docker ps"
#      docker ps
#      echo "Running docker ps -a"
#      docker ps -a



# srun -N "${NNODES}" \
#      --exclusive \
#      --ntasks-per-node=1 \
#      --cpus-per-task=128 \
#      -t 03:00:00 \
#      --exclude=useocpm2m-401-[086,036,037,028,067,068,069,052,102,123,124,125,005] \
#      bash -c "
#           readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
#           if [ \"\$SLURM_NODEID\" = \"0\" ]; then
#               echo \"========== Slurm cluster info ==========\"
#               echo \"SLURM_NODELIST: \${node_array[*]}\"
#               echo \"SLURM_NNODES: \${SLURM_NNODES}\"
#               echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
#               echo \"\"
#           fi
#           export MASTER_ADDR=\${node_array[0]}
#           export MASTER_PORT=\${MASTER_PORT}
#           export NNODES=\${SLURM_NNODES}
#           export NODE_RANK=\${SLURM_PROCID}
#           export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
#           bash ${SCRIPT_DIR}/run_local_pretrain.sh \"\$@\" 2>&1 | tee ${LOG_FILE}
#      " bash "$@"

    #  --exclude=useocpm2m-401-[086,036,037,028,069,052,102] \
    #  --exclude=useocpm2m-401-[086,036,037,028,067,068,069,052,102,123,124,125,005] \
