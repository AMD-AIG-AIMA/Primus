# 🚀 Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference. It is designed to support **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and ROCm-optimized components.


## 🆕 What's New

- **[2025/04/18]** Added [Preflight](./tools/preflight/README.md) cluster sanity checker.
- **[2025/04/14]** Added HipblasLT autotuning support.
- **[2025/04/09]** Expanded support for LLaMA2, LLaMA3, DeepSeek-V2/V3 in [Megatron model configs](https://github.com/AMD-AIG-AIMA/Primus/tree/main/primus/configs/models/megatron).
- **[2025/03/04]** Introduced Megatron trainer module.


## 📖 Contents

- [🚀 Primus](#-primus)
  - [🆕 What's New](#-whats-new)
  - [📖 Contents](#-contents)
  - [⚙️ Setup](#️-setup)
    - [Setup Docker](#setup-docker)
    - [Setup Primus](#setup-primus)
  - [🔹 Examples](#-examples)
    - [Megatron Pretrain](#megatron-pretrain)
  - [📝 TODOs](#-todos)

---

## ⚙️ Setup

### Setup Docker

We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following commands to pull and launch the container:

```bash
# Pull the latest Docker image
docker pull rocm/megatron-lm:latest

# Launch the container
docker run -d \
  --name=dev_primus \
  --network=host \
  --ipc=host  \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64G \
  -v /path/to/workspace:/workspace \
  rocm/megatron-lm:latest sleep infinity

# Access the container
docker exec -it dev_primus bash
```

### Setup Primus

Clone the repository and install dependencies:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git

# Or initialize submodules if already cloned
git submodule update --init --recursive

cd Primus

# Install Python dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

---

## 🔹 Examples

### [Megatron Pretrain](./examples/megatron/README.md)

Primus supports pretraining of various large language models with Megatron. Supported configurations include:

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

---

## 📝 TODOs

- [ ] Support for Primus-RL (training/inference modules for RLHF, OnlineDPO, GRPO, etc.)
- [ ] Add support for more model architectures and backends
