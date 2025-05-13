# Large Model Training Benchmark
## 1. Overview
This repository provides tools for benchmarking the training performance of large language models (LLMs). For each supported model, we provide a recommended parallelism strategy as the default configuration. Users are also encouraged to experiment with different parallel strategies to explore the performance impact under various settings.


## 2. How to Run

First, run the model you want to test using the following command:
```
MODEL_CONFIG=llama2_7B  \
DATA_PATH=/PATH/TO/DATA \
    bash benchmark_model.sh
```
The log results will be saved in the logs folder under the current directory.

Next, you can use the `benchmark_report.py` tool to process the logs and generate the benchmark CSV data.
```
python3 benchmark_report.py                 \
    --model llama2_7B                       \
    --benchmark-log-dir ./logs/llama2_7B/   \
    --report-csv-path model_benchmark_llama2_7B.csv

```
