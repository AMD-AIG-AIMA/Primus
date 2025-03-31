# Offline Tune


## 1. GEMM Tune

Use the `hipblaslt-bench` tool to perform GEMM tuning.

`hipblaslt-bench` is usually located under `/opt/rocm/bin`. However, if it's not available in some environments/docker, you'll need to reinstall hipblaslt.


### Install Hipblaslt (Optional)
You can reference: https://github.com/ROCm/hipBLASLt?tab=readme-ov-file#build-and-install

If only run MI300X, you can use the following command for a quick compilation, reducing the compilation time to under 2 hours.
```
./install.sh -idc --logic-yaml-filter gfx942/*/* -a gfx942 -j 256 --build_dir build
```


### Step 1: Dump Shape
* Set the Hipblaslt ENV.
* Run Train code.
* Unset ENV.
* The gemm shape will be dumped into `dump_gemm_shapes.txt`.
* Note: If just to dump shape, in most cases, there's no need to train for many iters—just a few should be enough, as each step uses the same shape.
```
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE=dump_gemm_shapes.txt

./run_your_code

unset HIPBLASLT_LOG_MASK
unset HIPBLASLT_LOG_FILE
```
### Step 2: Tuning
Run `offline_tune_gemm.py` and save tuned results in `tune_gemm_results.txt`
```
python3 offline_tune_gemm.py                            \
    --dump-shape-path /PATH/TO/dump_gemm_shapes.txt     \
    --tune-result-path /PATH/TO/tune_gemm_results.txt
```

### Step 3: Use tuned results to Train
* Set the results ENV.
* Start your tasks.
```
export HIPBLASLT_TUNING_OVERRIDE_FILE=tune_gemm_results.txt
./run_your_code
```

# Reference

https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html
