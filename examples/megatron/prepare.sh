#!/bin/bash
# shellcheck disable=SC2086
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

echo "-00-----------------------"

# Set PRIMUS_PATH to the root directory of the framework
PRIMUS_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)
echo "[INFO] PRIMUS_PATH is set to: ${PRIMUS_PATH}"

Set MEGATRON_PATH to the default path unless explicitly provided
export MEGATRON_PATH=${MEGATRON_PATH:-${PRIMUS_PATH}/third_party/Megatron-LM}
echo "[INFO] MEGATRON_PATH is set to: ${MEGATRON_PATH}"

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

check_dir_nonempty "$MEGATRON_PATH" "MEGATRON_PATH"

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

export NODE_RANK=${NODE_RANK:-0}

# ----------- Dataset Preparation -----------
# Prepare or validate the tokenized dataset used for training.
# Ensures consistency across nodes and performs preprocessing if necessary.

MODEL=$(grep "^[[:space:]]*model:" "$EXP" | awk -F ': ' '{print $2}')
# Check if MODEL is set, otherwise exit with error
if [ -z "$MODEL" ]; then
    echo "[ERROR] 'model' must be specified as one of the available model names in $EXP."
    echo "        Example model: llama2_7B.yaml"
    echo "        Available models can be found under: primus/configs/models/megatron/"
    exit 1
fi

if [[ "$MODEL" =~ ^\$\{([^}:]+)(:([^}]+))?\}\.yaml$ ]]; then
    VAR_NAME="${BASH_REMATCH[1]}"
    DEFAULT_VALUE="${BASH_REMATCH[3]}"

    if [[ -n "${!VAR_NAME}" ]]; then
        RESOLVED_MODEL="${!VAR_NAME}"
        echo "Using environment variable $VAR_NAME = ${!VAR_NAME}, resolved MODEL = $RESOLVED_MODEL"
    elif [[ -n "$DEFAULT_VALUE" ]]; then
        RESOLVED_MODEL="${DEFAULT_VALUE}"
        echo "Using default value for $VAR_NAME = $DEFAULT_VALUE, resolved MODEL = $RESOLVED_MODEL"
    else
        echo "[ERROR] Neither environment variable \$$VAR_NAME is set nor default value provided for MODEL."
        exit 1
    fi
    MODEL="${RESOLVED_MODEL}.yaml"
else
    # Not in ${VAL}.yaml or ${VAL:VALUE}.yaml format; must end with .yaml
    if [[ "$MODEL" != *.yaml ]]; then
        echo "[ERROR] MODEL must end with .yaml (given: $MODEL)"
        exit 1
    fi
    echo "Using literal MODEL = $MODEL"
fi

MODEL_CONFIG_FILE="$PRIMUS_PATH/primus/configs/models/megatron/${MODEL}"
if [[ ! -f "$MODEL_CONFIG_FILE" ]]; then
    echo "[ERROR] Model config file not found: $MODEL_CONFIG_FILE"
    echo "        Please make sure the file exists under primus/configs/models/megatron/"
    exit 1
fi


# Extract tokenizer_type and tokenizer_model from the model config file
TOKENIZER_TYPE=$(grep "^tokenizer_type:" "$MODEL_CONFIG_FILE" | awk -F ': ' '{print $2}')
TOKENIZER_MODEL=$(grep "^tokenizer_model:" "$MODEL_CONFIG_FILE" | awk -F ': ' '{print $2}')

# Ensure these variables are not empty
if [[ -z "$TOKENIZER_TYPE" ]]; then
    echo "[ERROR]: 'tokenizer_type' not found in ${MODEL_CONFIG_FILE}."
    exit 1
fi

if [[ -z "$TOKENIZER_MODEL" ]]; then
    echo "[ERROR]: 'tokenizer_model' not found in ${MODEL_CONFIG_FILE}."
    exit 1
fi
export TOKENIZER_TYPE
export TOKENIZER_MODEL

export TOKENIZED_DATA_PATH=${TOKENIZED_DATA_PATH:-${DATA_PATH}/bookcorpus/${TOKENIZER_TYPE}/bookcorpus_text_sentence}

echo "--${DATA_PATH}"
echo "--${TOKENIZER_TYPE}"
echo "--${TOKENIZER_MODEL}"
echo "--${TOKENIZED_DATA_PATH}"

if [[ "$NODE_RANK" == "0" && ! -f "${TOKENIZED_DATA_PATH}.done" ]]; then
    # Ensure HF_TOKEN is set; exit with error if not
    if [[ -z "${HF_TOKEN}" ]]; then
        echo "Error: Environment variable HF_TOKEN must be set."
        exit 1
    fi

    bash ./examples/scripts/prepare_dataset.sh "$DATA_PATH" "$TOKENIZER_TYPE" "$TOKENIZER_MODEL"
    touch "${TOKENIZED_DATA_PATH}.done"
    echo "Dataset preparation completed."

elif [[ "$NODE_RANK" != "0" ]]; then
    while [[ ! -f "${TOKENIZED_DATA_PATH}.done" ]]; do
        echo "Waiting for dataset..."
        sleep 30
    done
fi

# build helper_cpp of megatron
pushd "${MEGATRON_PATH}/megatron/core/datasets" && make && popd || exit 1