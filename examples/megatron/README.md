
This guide demonstrates how to perform pretraining using **Megatron** within the **Primus** framework.
It supports both **single-node** and **multi-node** training, and includes **HipblasLT auto-tuning** for optimal performance.

---

## 📚 Table of Contents

- [📚 Table of Contents](#-table-of-contents)
- [🖥️ Single Node Training](#️-single-node-training)
  - [Setup Docker](#setup-docker)
  - [Setup Primus](#setup-primus)
  - [Run Pretraining](#run-pretraining)
    - [🚀 Quick Start Mode](#-quick-start-mode)
    - [🧑‍🔧 Interactive Mode](#-interactive-mode)
- [🌐 Multi-node Training](#-multi-node-training)
- [🔧 HipblasLT Auto Tuning](#-hipblaslt-auto-tuning)
  - [Stage 1: Dump GEMM Shape](#stage-1-dump-gemm-shape)
  - [Stage 2: Tune GEMM Kernel](#stage-2-tune-gemm-kernel)
  - [Stage 3: Train with Tuned Kernel](#stage-3-train-with-tuned-kernel)
- [✅ Supported Models](#-supported-models)
  - [🏃‍♂️ How to Run a Supported Model](#️-how-to-run-a-supported-model)

---

## 🖥️ Single Node Training

### Setup Docker
We recommend using the official [rocm/megatron-lm Docker image](https://hub.docker.com/r/rocm/megatron-lm) to ensure a stable and compatible training environment. Use the following commands to pull and launch the container:

```bash
# Pull the latest Docker image
docker pull docker.io/rocm/megatron-lm:v25.5_py310

```

---

### Setup Primus
Clone the repository and install dependencies:

```bash
# Clone with submodules
cd /workspace
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

### Run Pretraining
Use the `run_pretrain.sh` script to start training.

#### 🚀 Quick Start Mode

Use this mode for **rapid iteration or validation** of a model config.
You do not need to enter the Docker container. Just set the config and run.

```bash
# Example for llama2_7B
EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash ./examples/megatron/run_local_pretrain.sh

```

---

#### 🧑‍🔧 Interactive Mode

This mode is recommended for **development, debugging**, or running **custom workflows**.
You will manually enter the container and execute training inside.

```bash
# Launch the container
bash tools/docker/start_container.sh

# Access the container
docker exec -it dev_primus bash

# install required packages
cd Primus && pip install -r requirements.txt

# Example for llama2_7B
EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash ./examples/megatron/run_pretrain.sh

```

---

## 🌐 Multi-node Training

Multi-node training is launched via **SLURM**.
Specify the number of nodes and the model config:

```bash
export DOCKER_IMAGE="docker.io/rocm/megatron-lm:v25.5_py310"
export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash
export NUM_NODES=8
bash ./examples/megatron/run_slurm_pretrain.sh
```

## 🔧 HipblasLT Auto Tuning

HipblasLT tuning is divided into three stages and controlled via the environment variable `PRIMUS_HIPBLASLT_TUNING_STAGE`:

```bash
# default 0 means no tuning
export PRIMUS_HIPBLASLT_TUNING_STAGE=${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}
```

---

### Stage 1: Dump GEMM Shape
In this stage, GEMM shapes used during training are collected.
It is recommended to reduce `train_iters` for faster shape generation.

```bash
# Output will be stored to:
# ./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_shape

export PRIMUS_HIPBLASLT_TUNING_STAGE=1
export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml
NUM_NODES=1 bash ./examples/megatron/run_slurm_pretrain.sh
```

---

### Stage 2: Tune GEMM Kernel

This stage performs kernel tuning based on the dumped GEMM shapes using the [offline_tune tool](https://github.com/AMD-AIG-AIMA/Primus/tree/main/examples/offline_tune).
It typically takes 10–30 minutes depending on model size and shape complexity.


```bash
# Output will be stored to:
# ./output/tune_hipblaslt/${PRIMUS_MODEL}/gemm_tune/tune_hipblas_gemm_results.txt

export PRIMUS_HIPBLASLT_TUNING_STAGE=2
export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml
NUM_NODES=1 bash ./examples/megatron/run_slurm_pretrain.sh
```

---

### Stage 3: Train with Tuned Kernel

In this final stage, the tuned kernel is loaded for efficient training:

```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=3
export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml
NUM_NODES=1 bash ./examples/megatron/run_slurm_pretrain.sh
```

## ✅ Supported Models

The following models are supported out of the box via provided configuration files:

| Model            | Huggingface Config | Megatron Config |
| ---------------- | ------------------ | --------------- |
| llama2_7B        | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)         | [llama2_7B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/config/llama2_7B-pretrain.yaml)               |
| llama2_70B       | [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)       | [llama2_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/llama2_70B-pretrain.yaml)             |
| llama3_8B        | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)     | [llama3_8B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/llama3_8B-pretrain.yaml)               |
| llama3_70B       | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)   | [llama3_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/llama3_70B-pretrain.yaml)             |
| llama3.1_8B      | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)           | [llama3.1_8B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/llama3.1_8B-pretrain.yaml)           |
| llama3.1_70B     | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)         | [llama3.1_70B-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/llama3.1_70B-pretrain.yaml)         |
| deepseek_v2_lite | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | [deepseek_v2_lite-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/deepseek_v2_lite-pretrain.yaml) |
| deepseek_v2      | [deepseek-ai/DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)           | [deepseek_v2-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/deepseek_v2-pretrain.yaml)           |
| deepseek_v3      | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)           | [deepseek_v3-pretrain.yaml](https://github.com/AMD-AIG-AIMA/Primus/blob/main/examples/megatron/configs/deepseek_v3-pretrain.yaml)           |


---

### 🏃‍♂️ How to Run a Supported Model

Use the following command pattern to start training with a selected model configuration:

```bash
EXP=examples/megatron/configs/<model_config>.yaml bash ./examples/megatron/run_local_pretrain.sh
```

For example, to run the llama2_7B model quickly:

```bash
EXP=examples/megatron/configs/llama2_7B-pretrain.yaml bash ./examples/megatron/run_local_pretrain.sh
```


For multi-node training via SLURM, use:

```bash
export EXP=examples/megatron/configs/llama2_7B-pretrain.yaml
export NUM_NODES=8
bash ./examples/megatron/run_slurm_pretrain.sh
```
