# Large Model Training Benchmark
## 1. Overview
This repository provides tools for benchmarking the training performance of large language models (LLMs). For each supported model, we provide a recommended parallelism strategy as the default configuration. Users are also encouraged to experiment with different parallel strategies to explore the performance impact under various settings.


## 2. How to Run

First, run the model you want to test using the following command:
```bash
export DATA_PATH=/PATH/TO/DATA
python3 benchmark/megatron/model/benchmark_model.py --model llama2_7B
```
The log results will be saved in the logs folder under the current directory.

Next, you can use the `benchmark_report.py` tool to process the logs and generate the benchmark CSV data.
```bash
python3 benchmark/megatron/model/benchmark_report.py --report-csv-path model_benchmark_llama2_7B.csv

```
