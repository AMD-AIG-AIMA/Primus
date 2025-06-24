#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# examples
# bash ./examples/megatron/prepare_dataset.sh ./data_path DeepSeekV2Tokenizer deepseek-ai/DeepSeek-V2

export DATA_PATH=$1
# Note: The same type of tokenizer uses the same tokenizer model.
# For example, `deepseek-ai/DeepSeek-V2` and `deepseek-ai/DeepSeek-V2-Lite` use
# the same tokenizer model. Therefore, the `tokenizer_type` is the same as `DeepSeekV2Tokenizer`,
# and the tokenized data path is also the same.
# Therefore, if you have already preprocessed the data using the same tokenizer model,
# you don't need to run this script again.
#
# tokenizer_type,       tokenizer_model
# DeepSeekV2Tokenizer,  deepseek-ai/DeepSeek-V2
# DeepSeekV2Tokenizer,  deepseek-ai/DeepSeek-V2-Lite
# DeepSeekV3Tokenizer,  deepseek-ai/DeepSeek-V3
# DeepSeekV3Tokenizer,  deepseek-ai/DeepSeek-V3-base
#
# available tokenizer types: Primus/primus/backends/megatron/training/tokenizer/tokenizer.py@build_tokenizer
# available tokenizer models: https://huggingface.co
export TOKENIZER_TYPE=$2 # DeepSeekV2Tokenizer
export TOKENIZER_MODEL=$3 # deepseek-ai/DeepSeek-V2-Lite


# framework path
PRIMUS_PATH=$(realpath "$(dirname "$0")/../..")
export PRIMUS_PATH
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/third_party/Megatron-LM}
export PYTHONPATH=${MEGATRON_PATH}:${PRIMUS_PATH}:${PYTHONPATH}
[[ ! -d "${MEGATRON_PATH}" ]] && {
    echo "Error: MEGATRON_PATH (${MEGATRON_PATH}) does not exist"
    exit 1
}
echo "HF_HOME: $HF_HOME"
echo "PRIMUS_PATH: $PRIMUS_PATH"
echo "MEGATRON_PATH: $MEGATRON_PATH"

# bookcorpus dataset
export DATASET=bookcorpus
DATASET_PATH="${DATA_PATH}/${DATASET}"
OUTPUT_PATH="$DATASET_PATH/${TOKENIZER_TYPE}"
export HF_HOME=${HF_HOME:-"${DATA_PATH}"/data/huggingface}
mkdir -p "$OUTPUT_PATH"

export TOKENIZED_DATA_PATH=${TOKENIZED_DATA_PATH:-"${OUTPUT_PATH}"/bookcorpus_text_sentence}
if [[ -f "${TOKENIZED_DATA_PATH}.bin" && -f "${TOKENIZED_DATA_PATH}.idx" ]]; then
    echo "Tokenized data files ${TOKENIZED_DATA_PATH}.bin and ${TOKENIZED_DATA_PATH}.idx exist, skip data preprocess"
    exit 0
fi

START_TIME=$(date +%s)
if [[ -f "${DATASET_PATH}/bookcorpus_megatron.json" ]]; then
    echo "Find the '${DATASET}' dataset: '${DATASET_PATH}'/bookcorpus_megatron.json, skip download."
else
    echo "Downloading '${DATASET}' dataset to '${DATASET_PATH}'..."
    python3 "${PRIMUS_PATH}"/examples/megatron/prepare_bookcorpus_megatron_dataset.py --out-dir "${DATASET_PATH}"
fi

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Download '${DATASET}' completed. Time: '${ELAPSED_TIME}' s"

START_TIME=$(date +%s)
python "${PRIMUS_PATH}"/examples/megatron/preprocess_data.py \
    --input "${DATASET_PATH}"/bookcorpus_megatron.json \
    --tokenizer-type "${TOKENIZER_TYPE}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --output-prefix "${OUTPUT_PATH}"/bookcorpus \
    --workers "$(nproc)" --split-sentences --partitions 2

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Dataset '${DATASET}' preprocess completed. Time: '${ELAPSED_TIME}' s"
