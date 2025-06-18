# Large Model Checkpoint Benchmark
## 1. Overview
This directory provides a benchmark tool for checkpoint saving when training large language models with Primus(megatron-lm backend).

It requires the user to specify a Primus YAML config file. 

Since the performance of checkpoint saving is related to the number of processes, data_parallel_size, and other parallel settings, 
it also supports overwriting some of the parallel configurations in the YAML file during benchmarking.

In the current version, the final tool outputs the following checkpoint saving metrics.
```
{
    "world_size": "8",
    "data_parallel_size": "8",
    "ckpt_format": "torch",
    "ckpt_fully_parallel_save": "True",
    "async_save": "None",
    "save": "/mnt/apps_proxy/tas/limou/source/Primus/output/amd/root/exp_pretrain/checkpoints",
    "save_interval": "20",
    "optimizer": "adam",
    "use_distributed_optimizer": "True",
    "params_dtype": "torch.bfloat16",
    "main_params_dtype": "torch.float32",
    "exp_avg_dtype": "torch.float32",
    "exp_avg_sq_dtype": "torch.float32",
    "block_time": 844, # time in seconds, which blocks main training process (for async_save=True)
    "total_time": 844,
    "accurate": true,
    "num_saved": 2,
    "iter_folder_size": 676320105034, # 629.8 GB
    "write_bandwidth_in_mbps": 764.2051111849563 # 764 MB/s
}
```

## 2. How to Run

The entry file is ckpt_launch.py, of course you can also run ckpt_report.py separately if needed.

example:
```
export DATA_PATH=/PATH/TO/DATA
python3 benchmark/megatron/checkpoint/ckpt_launch.py \
    --yaml-config-path examples/megatron/configs/mixtral_8x7B_v0.1-pretrain.yaml \
    --nnodes 1
```
If you need to benchmark multiple different models, parallel strategies, and checkpoint modes, 
you can add a simple wrapper script around the tool to call it multiple times for statistics.
