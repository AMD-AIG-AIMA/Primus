# How to run GEMM kernel benchmark on AMD MI355/350


## 1. Pull the Docker image for MI355/MI300

```bash
podman pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

If using docker environment, please change `podman` to `docker`

## 2. Start the container.

```bash
podman run -it --device /dev/dri --device /dev/kfd \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME \
    -w /workspace/Megatron-LM \
    --name gemm_benchmark \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

## 3. Rebuild the latest hipblaslt with rocm-library

```bash
export MAX_JOBS=128
export PYTORCH_ROCM_ARCH="gfx950" #use "gfx950" for MI355X/MI350X, "gfx942" for MI300X/MI325X
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipblaslt/
apt remove -y hipblaslt
MAX_JOBS=${MAX_JOBS} ./install.sh -idc --architecture ${PYTORCH_ROCM_ARCH} --skip_rocroller
ln -s /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/libhipblaslt.so.0
```

## 4. Run GEMM benchmark with provided shapes


Copy the following content in to a file `hipblaslt_bench_gemm.sh` and then run `bash hipblaslt_bench_gemm.sh` in the command line.

```bash
#!/bin/bash
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 4096 -n 65536 -k 14336 --lda 14336 --ldb 14336 --ldc 4096 --ldd 4096
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 4096 -n 4096 -k 65536 --lda 65536 --ldb 65536 --ldc 4096 --ldd 4096
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 32768 -k 28672 --lda 28672 --ldb 28672 --ldc 8192 --ldd 8192
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 57344 -k 32768 --lda 32768 --ldb 32768 --ldc 8192 --ldd 8192
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 32768 -k 8192 --lda 8192 --ldb 8192 --ldc 8192 --ldd 8192
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 8192 -k 32768 --lda 32768 --ldb 32768 --ldc 8192 --ldd 8192
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 4096 -n 32768 -k 14336 --lda 14336 --ldb 14336 --ldc 4096 --ldd 4096
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 57344 -n 32768 -k 8192 --lda 8192 --ldb 8192 --ldc 57344 --ldd 57344
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 32768 -k 57344 --lda 57344 --ldb 57344 --ldc 8192 --ldd 8192
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 28672 -n 8192 -k 32768 --lda 32768 --ldb 32768 --ldc 28672 --ldd 28672
hipblaslt-bench --api_method c --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0 --alpha 1.000000 --beta 0.000000 --transA T --transB N --batch_count 1 --scaleA 1 --scaleB 1 --a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --rotating 512 --iters 1000 --cold_iters 200 -m 8192 -n 10240 -k 32768 --lda 32768 --ldb 32768 --ldc 8192 --ldd 8192

```

The following is an output exmple that with M,N,K=4096,65536,14336 FP8 GEMM, the measured throughput is 3273TFlops.

```bash
hipBLASLt version: 100100
hipBLASLt git version: 319de70737
Query device success: there are 8 devices. (Target device ID is 0)
Device ID 0 : AMD Radeon Graphics gfx950:sramecc+:xnack-
with 309.2 GB memory, max. SCLK 2400 MHz, max. MCLK 2000 MHz, compute capability 9.5
maxGridDimX 2147483647, sharedMemPerBlock 163.8 KB, maxThreadsPerBlock 1024, warpSize 64

Rotating buffer 512 MiB. Needed Size: 1464 MiB. Needed block count: 1 (Capped to max iters: 1000)
Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,aux_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us
    T,N,0,1,4096,65536,14336,1,14336,58720256,0,14336,939524096,4096,268435456,4096,268435456,f8_r,f8_r,bf16_r,bf16_r,f32_r,1,1,0,0,0,none,0,f32_r,bf16_r,512,3.2733e+06,608.035,2351.32
```
