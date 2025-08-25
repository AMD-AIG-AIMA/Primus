#!/bin/bash

set -euo pipefail

export NNODES="${NNODES:-1}"

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")

bash "$PRIMUS_PATH/runner/primus-run.sh slurm" -- benchmark gemm "$@"


# sbatch
# SBATCH_ARGS=(--nodes="$NNODES")
# if [[ -n "${RESERVATION:-}" ]]; then
#     SBATCH_ARGS+=(--reservation="$RESERVATION")
# fi
# sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_slurm_job.sh" train pretrain --config "$EXP"  "sh $@"
