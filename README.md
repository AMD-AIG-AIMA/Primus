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
# If you are cloning the repository for the first time:
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git
# If you've already cloned primus without submodules, run the following commands:
git submodule update --init --recursive
```

## Setup Primus
```bash
cd workspace/Primus
# Install the required dependencies using:
pip install -r requirements.txt
# setup the pre-commit for your repo
pre-commit install
```

## Examples
```bash
cd workspace/Primus
# megatron pretrain
./examples/megatron/run_pretrain.sh
```
