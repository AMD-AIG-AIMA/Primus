#!/bin/bash

set -euo pipefail

export NNODES="${NNODES:-1}"

# -------------------- EXP Check --------------------
if [ -z "${EXP:-}" ]; then
    echo "[ERROR] EXP must be specified (e.g., examples/megatron/exp_pretrain.yaml)." \
         "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# Ensure EXP file exists, otherwise exit with error
if [ ! -f "${EXP}" ]; then
    echo "[ERROR] The specified EXP file does not exist: ${EXP}" \
         "Primus will use the configuration in EXP to train the model."
    exit 1
fi

# -------------------- DATA_PATH Check --------------------
if [ -z "${DATA_PATH:-}" ]; then
    DATA_PATH="$(pwd)/data"
    echo "[WARNING] DATA_PATH not specified. Defaulting to: ${DATA_PATH}"

    if [ ! -d "${DATA_PATH}" ]; then
        echo "[WARNING] DATA_PATH does not exist. Creating: ${DATA_PATH}"
        mkdir -p "${DATA_PATH}"
    fi
fi

# Slurm Launch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export SCRIPT_DIR
export DATA_PATH
bash "${SCRIPT_DIR}/run_slurm_job.sh" pretrain --config "$EXP"

# sbatch
# SBATCH_ARGS=(--nodes="$NNODES")
# if [[ -n "${RESERVATION:-}" ]]; then
#     SBATCH_ARGS+=(--reservation="$RESERVATION")
# fi
# sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_slurm_job.sh" pretrain --config "$EXP"
