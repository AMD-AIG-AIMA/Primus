# Large Model Training Benchmark
## 1. Overview
This repository provides tools for benchmarking the training performance of large language models (LLMs). For each supported model, we provide a recommended parallelism strategy as the default configuration. Users are also encouraged to experiment with different parallel strategies to explore the performance impact under various settings.


## 2. Hardware & Software

| Software Component | Version |
|--------------------|---------|
| Primus             |         |
| ROCm               |         |
| Python             |         |
| PyTorch            |         |
| Transformer Engine |         |
| Flash Attention    |         |
| hipBLASLt          |         |
| Triton             |         |

## 3. Benchmark

### 3.1 Supported Models

* llama-2-7B
* llama-2-70B
* llama-3-8B
* llama-3-70B
* DeepSeek-V2-Lite
* DeepSeek-V2
* DeepSeek-V3


### 3.2 Base Perf

| Model       | Paralle Strategy<br>(DP/TP/PP/EP/CP) | GBS/MBS | SeqLen | Nodes | Hardware | TFLOP/s/GPU  | Step Time(s) | Memory Usages(%)  |
|-------------|--------------------------------------|---------|--------|-------|----------|--------------|--------------|-------------------|
| Llama2-7B   | 8/1/1/1/1                            | 128/4   |  4096  |   1   |  MI300X  |              |              |                   |
| Llama2-70B  | 8/1/1/1/1                            | 128/4   |  4096  |   1   |  MI300X  |              |              |                   |
| Llama3-8B   | 8/1/1/1/1                            | 128/4   |  8192  |   1   |  MI300X  |              |              |                   |
| Llama3-70B  | 8/1/1/1/1                            | 128/4   |  8192  |   1   |  MI300X  |              |              |                   |

### 3.3 Gemm Tune Perf

...


## 4. How to Run

...
