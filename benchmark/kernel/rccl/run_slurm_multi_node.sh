#!/bin/bash
#SBATCH --job-name=all2all
#SBATCH --output=logs/slurm/all2all-2node-job.%j.out
#SBATCH --nodes=2                         # Number of nodes, Adjust as necessary
#SBATCH --ntasks-per-node=1                  # One task per GPU -> total 8 tasks per node
#SBATCH --cpus-per-task=96                  # assign all CPUs to the job
#SBATCH --gres=gpu:8                         # Request 8 GPUs per node
#SBATCH --time=01:00:00                      # Adjust as necessary
#SBATCH --reservation=gpu-40_gpu-41_gpu-43_gpu-44_gpu-46_gpu-47_gpu-50_gpu-55_reservation # modify based on your reservation settings
#SBATCH --nodelist=gpu-40,gpu-43
echo "get first node"
# Get the list of nodes and the first node (master node)
# node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
COORDINATOR_IP=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_node=$COORDINATOR_IP
# node_array=(${node_list})
# master_node=${node_array[0]}

# Set environment variables for distributed training
export SLURM_MASTER_ADDR=$master_node
export SLURM_MASTER_PORT=29565

# Optional: Print out the values for debugging
echo "MASTER_ADDR=$SLURM_MASTER_ADDR"
echo "MASTER_PORT=$SLURM_MASTER_PORT"

# Define the Docker image
# export NCCL_IB_HCA=$(rdma link -j | python3 -c "import sys, json; links=json.load(sys.stdin);names=[links[i]['ifname'] for i in range(8)]; print(*names,sep=',')")
NCCL_IB_HCA_LIST=$(bash "../../../examples/scripts/get_nccl_ib_hca.sh")
echo "NCCL_IB_HCA_LIST: $NCCL_IB_HCA_LIST"
# export NCCL_IB_HCA=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export NCCL_IB_HCA=$NCCL_IB_HCA_LIST
echo "NCCL_IB_HCA: $NCCL_IB_HCA"


export DOCKER_IMAGE="docker.io/rocm/megatron-lm:v25.5_py310"
# Pull docker image
podman pull $DOCKER_IMAGE # change podman to docker if you are using docker
# Setup your keys for HF and WADNB
# export HF_TOKEN="your hf key"
# export WANDB_API_KEY="wandb key"
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export TIME_STAMP=${CURRENT_TIME}
echo "TIME_STAMP=$TIME_STAMP"

# Define the mount points
export TITAN_DIR=${PWD}                                # change this path to Megatron-LM inside the docker
export CONTAINER_DIR=${HOME}
export HOST_MOUNT=${HOST_MOUNT:=${HOME}}               # change this path to host dir intend to be attached to the docker
export CONTAINER_MOUNT=${CONTAINER_MOUNT:=${HOME}}     # change this path to development workspace path inside the docker

podman ps -aq | xargs -r podman rm -f
# Run the Docker container with the script
srun bash -c ' podman run --rm \
 --env SLURM_MASTER_ADDR=$SLURM_MASTER_ADDR \
 --env SLURM_MASTER_PORT=$SLURM_MASTER_PORT \
 --env SLURM_PROCID=$SLURM_PROCID \
 --env SLURM_NODEID=$SLURM_NODEID \
 --env SLURM_NNODES=$SLURM_NNODES \
 --ipc=host --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE  --cap-add=CAP_SYS_ADMIN  \
 --security-opt seccomp=unconfined --group-add video --privileged --device=/dev/infiniband \
 -v $HOST_MOUNT:$CONTAINER_MOUNT \
 $DOCKER_IMAGE /bin/bash -c \
 "echo $(date) ; \
 export NNODES=$SLURM_NNODES ; \
 export NODE_RANK=$SLURM_NODEID ;  \
 export NCCL_IB_HCA=$NCCL_IB_HCA ; \
 cd $TITAN_DIR; \
 bash run_script.sh \
 echo $(date)
 "
 '
