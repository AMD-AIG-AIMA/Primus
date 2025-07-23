# How to run deepseek model pretraining on MI355X

## 1. Pull the Docker image for MI355/MI300

```bash
podman pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

## 2. Get the Primus repo

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
git checkout dev/mi355_benchmark
mkdir -p data
```

## 3. Run the training

Use the following command run the deepseek-v2-lite with megatron backend. The training configuration is defined in `examples/megatron/configs/deepseek_v2_lite-pretrain_MI355.yaml`

```bash
export HF_TOKEN="your_huggingface_token"
export DOCKER_IMAGE=rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
export NNODES=1
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
EXP=examples/megatron/configs/deepseek_v2_lite-pretrain_MI355.yaml bash ./examples/run_local_pretrain.sh
```

The pytorch trace will be saved at  `Primus/output` and the throughput information will print at the console directly.
