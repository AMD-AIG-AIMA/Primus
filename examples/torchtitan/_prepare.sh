#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Set PRIMUS_PATH to the root directory of the framework
PRIMUS_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)
echo "[INFO] PRIMUS_PATH is set to: ${PRIMUS_PATH}"

# Set TORCHTITAN_PATH to the default path unless explicitly provided
export TORCHTITAN_PATH=${TORCHTITAN_PATH:-${PRIMUS_PATH}/third_party/torchtitan}
echo "[INFO] TORCHTITAN_PATH is set to: ${TORCHTITAN_PATH}"

# Function to check if a directory exists and is not empty
check_dir_nonempty() {
    local dir_path=$1
    local name=$2
    if [[ ! -d "$dir_path" || -z "$(ls -A "$dir_path")" ]]; then
        echo "[ERROR] $name ($dir_path) does not exist or is empty."
        echo "        Please ensure Primus is properly initialized."
        echo
        echo "        If not yet cloned, run:"
        echo "            git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git"
        echo
        echo "        Or if already cloned, initialize submodules with:"
        echo "            git submodule update --init --recursive"
        echo
        exit 1
    fi
}

check_dir_nonempty "$TORCHTITAN_PATH" "TORCHTITAN_PATH"


echo "[INFO] pip install for torchtitan......"
pip install -r "${TORCHTITAN_PATH}/requirements.txt" --quiet

export DATA_PATH=${DATA_PATH:-"${PRIMUS_PATH}/data"}
echo "[INFO] DATA_PATH is set to: ${DATA_PATH}"

export HF_HOME=${HF_HOME:-"${DATA_PATH}/huggingface"}
echo "[INFO] HF_HOME is set to: ${HF_HOME}"

# Ensure EXP is set, otherwise exit with error
if [ -z "${EXP:-}" ]; then
    echo "Error: EXP must be specified (e.g., examples/megatron/exp_pretrain.yaml)."
    echo "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
    echo "[ERROR] The specified EXP file does not exist: ${EXP}"
    echo "        Primus will use the configuration in EXP to train the model."
    exit 1
fi
echo "[INFO] EXP is set to: ${EXP}"
echo ""

# TOKENIZER_PATH=$(grep "^[[:space:]]*tokenizer_path[[:space:]]*=" "$EXP" | awk -F '=' '{print $2}' | tr -d ' "')
# Step 1: Extract framework and model config
# eval $(python3 -c "
# import yaml
# with open('${EXP}', 'r') as f:
#     cfg = yaml.safe_load(f)
# framework = cfg['modules']['pre_trainer']['framework']
# model_file = cfg['modules']['pre_trainer']['model']
# print(f'FRAMEWORK={framework}')
# print(f'MODEL_FILE={model_file}')
# " 2>/dev/null)

if [ -z "$FRAMEWORK" ] || [ -z "$MODEL_FILE" ]; then
    echo "[ERROR] Failed to extract framework or model file from EXP yaml: ${EXP}"
    exit 1
fi

# Step 2: Build model config path
MODEL_CONFIG_PATH="${PRIMUS_PATH}/primus/configs/models/${FRAMEWORK}/${MODEL_FILE}"
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "[ERROR] Model config file not found: $MODEL_CONFIG_PATH"
    exit 1
fi

Step 3: Get tokenizer_path from model config
TOKENIZER_PATH=$(python3 -c "
import yaml
with open('${MODEL_CONFIG_PATH}', 'r') as f:
    model_cfg = yaml.safe_load(f)
print(model_cfg['model']['tokenizer_path'])
" 2>/dev/null)

if [ -z "$TOKENIZER_PATH" ]; then
    echo "[ERROR] Failed to extract tokenizer_path from model config: $MODEL_CONFIG_PATH"
    exit 1
fi

FULL_PATH="${DATA_PATH%/}/torchtitan/${TOKENIZER_PATH#/}/"
TOKENIZER_FILE="${FULL_PATH}/original/tokenizer.model"

export NODE_RANK=${NODE_RANK:-0}

if [ ! -f "$TOKENIZER_FILE" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "[ERROR] HF_TOKEN not set. Please export HF_TOKEN."
        exit 1
    fi

    echo "HF_TOKEN $HF_TOKEN"
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "[INFO] Downloading tokenizer to $FULL_PATH ..."
        mkdir -p "$FULL_PATH"
        python "${TORCHTITAN_PATH}/scripts/download_tokenizer.py" \
            --repo_id "${TOKENIZER_PATH}" \
            --tokenizer_path "original" \
            --local_dir "$FULL_PATH" \
            --hf_token "${HF_TOKEN}"
    else
        echo "[INFO] Rank $RANK waiting for tokenizer file $TOKENIZER_FILE ..."
        while [ ! -f "$TOKENIZER_FILE" ]; do
            sleep 5
        done
    fi
else
    echo "[INFO] Tokenizer file exists: $TOKENIZER_FILE"
fi

export TOKENIZER_PATH=$TOKENIZER_FILE
echo "[INFO] TOKENIZER_PATH is set to: ${TOKENIZER_PATH}"

export LOCAL_RANKS_FILTER="0"
