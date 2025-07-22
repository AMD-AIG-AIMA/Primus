#!/bin/bash
set -x

PWD=$(cd "$(dirname "$0")" || exit; pwd)

python3 "${PWD}"/offline_tune_gemm.py                                           \
    --dump-shape-path-or-file "${PWD}"/DeepSeekV3_mbs1_seq4096_ep8.txt          \
    --tune-result-path "${PWD}"/DeepSeekV3_mbs1_seq4096_ep8_tune_results.txt    \
    --reports-result-path "${PWD}"/DeepSeekV3_mbs1_seq4096_ep8_tune_reports.csv \
    --num-devices 1
