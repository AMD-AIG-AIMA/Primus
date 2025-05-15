# Large Model Training Operator Benchmark

This benchmark focuses on evaluating the performance of key operators in large model training scenarios, including GEMM, Attention, and communication-related operators.


## How to run

### GEMM
Run the following command to benchmark GEMM operators. All GEMM-related information and performance data for each model will be automatically saved as separate CSV files under `/PATH/TO/DIR`.
```
python3 benchmark_gemm.py                               \
    --model-config-path /PATH/TO/model_configs.json     \
    --report-dir-path   /PATH/TO/DIR

```

### Attention
Run the following command to generate benchmark data for Attention operators:
```
python3 benchmark_attention.py                         \
    --shapes-json-path /PATH/TO/attention_shapes.json  \
    --report-csv-path  /PATH/TO/attention_benchmark.csv
```


### RCCL
This benchmark evaluates the performance of commonly used communication primitives in large model training, including AllReduce, AllGather, ReduceScatter, Point-to-Point (P2P), and All2All operations.

To run it, simply configure the IP and PORT, then execute the script. Benchmark results will be automatically generated as multiple CSV files.
```
bash run_script.sh
```
