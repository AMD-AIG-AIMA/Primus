# DeepSeekV3 GEMM kernel benchmark


1. Pull the Docker image.

```bash
docker pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

2. Start the container.

```bash
docker run -it --device /dev/dri --device /dev/kfd \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G \
    -w /workspace/Megatron-LM \
    --name training_benchmark \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

3. Rebuild hipblaslt with latest rocm-library

```bash
export MAX_JOBS=128
export PYTORCH_ROCM_ARCH="gfx950" #use "gfx950" for MI355X/MI350X, "gfx942" for MI300X/MI325X
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipblaslt/
apt remove -y hipblaslt
MAX_JOBS=${MAX_JOBS} ./install.sh -idc --architecture ${PYTORCH_ROCM_ARCH} --skip_rocroller
ln -s /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/libhipblaslt.so.0
```

4. Clone Primus

```bash
git clone https://github.com/AMD-AIG-AIMA/Primus.git -b dev/zhangrb
```

5. Execute GEMM perf for DeepSeekV3 proxy model

We already dumped the DeepSeekV3 FP8 GEMM shapes with MBS=1, SEQ=4096 and EP=8 configuration. You can input it to `offline_tune_gemm.py` script directly.

```bash
python3 offline_tune_gemm.py                                                      \
    --dump-shape-path-or-file DeepSeekV3_mbs1_seq4096_ep8_fp8_gemms.txt           \
    --tune-result-path DeepSeekV3_mbs1_seq4096_ep8_fp8_gemms_tune_results.txt     \
    --reports-result-path DeepSeekV3_mbs1_seq4096_ep8_fp8_gemms_tune_reports.csv  \
    --num-devices 1
```

You will get a file named `DeepSeekV3_mbs1_seq4096_ep8_fp8_gemms_tune_reports.csv` which includes performance data of gemm kernels.

It should be similar to the table below:

| m     | n     | k     | batch_count | lda  | ldb  | ldc  | ldd  | alpha | beta | dtype_a | dtype_b | dtype_c | trans_a | trans_b | tflops      | kernel_name |
|-------|-------|-------|-------------|------|------|------|------|--------|------|----------|----------|----------|----------|----------|-------------|-------------|
| 7168  | 4096  | 4096  | 1           | 4096 | 4096 | 7168 | 7168 | 1      | 0    | f8_r     | bf8_r    | bf16_r   | T        | N        | 2549.893 | ... |
| 7168  | 4096  | 18432 | 1           | 18432| 18432| 7168 | 7168 | 1      | 0    | f8_r     | f8_r     | bf16_r   | T        | N        | 2911.221 | ... |
