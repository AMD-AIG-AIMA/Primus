# Primus

## Overview
Primus is a training framework that supports different training and inference backends. It is designed for pretraining, posttraining, and reinforcement learning tasks.

## Setup Environment
Use the following command to create a container:
```bash
# pull the public docker image
docker pull rocm/megatron-lm:latest

# create a container
docker run -d \
  --name=dev_username \
  --network=host\
  --ipc=host  \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  rocm/megatron-lm:latest sleep infinity

# get into the container
docker exec -it dev_username bash
```


Use the following command to clone the repo:
- [ ] Set Megatron-LM as a submodule repo
```bash
mkdir workspace && cd workspace
git clone git@github.com:AMD-AIG-AIMA/Primus.git
git clone git@github.com:NVIDIA/Megatron-LM.git
# version 20250324
cd Megatron-LM && git checkout d61821b7174bac690afbad9134bcb4983521052f
```

## Setup Primus
```bash
# Install the required dependencies using:
pip install -r requirements.txt

# setup the pre-commit for your repo
cd workspace/Primus && pre-commit install
```

## Examples
```bash
cd workspace/Primus
# deepseek pretrain (default use deepseek_v2_lite model)
./examples/deepseek/run_pretrain.sh
```



