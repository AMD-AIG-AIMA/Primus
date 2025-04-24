# Primus

## Overview
Primus is a training framework that supports different training and inference backends. It is designed for pretraining, posttraining, and reinforcement learning tasks.

## What's New
- **[2025/04/18]** [Preflight](./tools/preflight/README.md) sanity check.
- **[2025/04/14]** Support HipblasLT auto tuning.
- **[2025/04/09]** Add llama2/llama3/deepseek-v2/deepseek-v3 [megatron models](https://github.com/AMD-AIG-AIMA/Primus/tree/main/primus/configs/models/megatron).
- **[2025/03/04]** Megatron trainer.

## Contents
- [Primus](#primus)
  - [Overview](#overview)
  - [What's New](#whats-new)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Setup Docker](#setup-docker)
    - [Setup Primus](#setup-primus)
  - [Examples](#examples)
    - [Megatron Pretrain](#megatron-pretrain)
- [TODOs](#todos)


## Setup
### Setup Docker

We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following command to start a container on your machine for training:

```bash
# pull the latest docker image
docker pull rocm/megatron-lm:latest

# launch an instance of training container,
docker run -d \
  --name=dev_primus \
  --network=host\
  --ipc=host  \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /path/to/workspace:/workspace \
  -v /path/to/data:/workspace/data \
  rocm/megatron-lm:latest sleep infinity

# get into the container
docker exec -it dev_primus bash
```

### Setup Primus

Use the following command to clone the repo:
```bash
# If you are cloning the repository for the first time:
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git
# If you've already cloned primus without submodules, run the following command:
git submodule update --init --recursive

cd Primus
# Install the required dependencies
pip install -r requirements.txt
# Setup the pre-commit for your repo
pre-commit install
```



## Examples

### [Megatron Pretrain](./examples/megatron/README.md)

- Supported Models

| Model            | Huggingface Config | Megatron Config |
| ---------------- | ------------------ | --------------- |
| llama2_7B        | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)         | [llama2_7B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama2_7B.yaml)               |
| llama2_70B       | [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)       | [llama2_70B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama2_70B.yaml)             |
| llama3_8B        | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)     | [llama3_8B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama3_8B.yaml)               |
| llama3_70B       | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)   | [llama3_70B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama3_70B.yaml)             |
| llama3.1_8B      | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           | [llama3.1_8B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama3.1_8B.yaml)           |
| llama3.1_70B     | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)         | [llama3.1_70B.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/llama3.1_70B.yaml)         |
| deepseek_v2_lite | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | [deepseek_v2_lite.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/deepseek_v2_lite.yaml) |
| deepseek_v2      | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)           | [deepseek_v2.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/deepseek_v2.yaml)           |
| deepseek_v3      | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)           | [deepseek_v3.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/primus/configs/models/megatron/deepseek_v3.yaml)           |



# TODOs
- [ ] Primus-RL (training and inference modules, rlhf/onlinedpo/grpo/...)
- [ ] support more models
