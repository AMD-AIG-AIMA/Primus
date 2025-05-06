# Large Model Training Operator Benchmark

This benchmark focuses on evaluating the performance of key operators in large model training scenarios, including GEMM, Attention, and communication-related operators.


## How to run

### GEMM
...

### Attention
Run the following command to generate benchmark data for Attention.
```
python3 benchmark_attention.py                         \
    --shapes-json-path /PATH/TO/attention_shapes.json  \
    --report-csv-path /PATH/TO/attention_benchmark.csv
```


### RCCL
This benchmark focuses on evaluating the performance of commonly used communication primitives in mainstream large model training, including AllReduce, AllGather, ReduceScatter, Point-to-Point (P2P), and All2All operations.

Simply configure the IP and PORT in rccl/run_script.sh, then run the script to automatically generate multiple CSV-format benchmark results.
