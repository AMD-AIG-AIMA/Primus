#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_usage() {
cat <<'EOF'
Primus Slurm Launcher

Usage:
    primus-cli slurm [srun|sbatch] [SLURM_FLAGS...] -- <primus-entry> [PRIMUS_ARGS...]

Examples:
    primus-cli slurm srun -N 4 -p AIG_Model -- container -- train pretrain --config exp.yaml
    primus-cli slurm sbatch --output=run.log -N 2 -- container -- benchmark gemm -M 4096 -N 4096 -K 4096

Notes:
    - [srun|sbatch] is optional; defaults to srun.
    - All SLURM_FLAGS before '--' are passed directly to Slurm (supports both --flag=value and --flag value).
    - Everything after the first '--' is passed to Primus entry (e.g. container, direct, etc.), and then to Primus CLI.
    - For unsupported or extra Slurm options, just pass them after '--' (they'll be ignored by this wrapper).

Debug:
    - Collected SLURM flags and primus arguments will be printed before launch.

EOF
}

# Show help if requested or if no args are given
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="$SCRIPT_DIR/primus-cli-slurm-entry.sh"

# 1. Detect srun/sbatch mode
LAUNCH_CMD="srun"   # Default launcher
if [[ "${1:-}" == "sbatch" || "${1:-}" == "srun" ]]; then
    LAUNCH_CMD="$1"
    shift
fi

# 2. Collect all SLURM flags until '--'
SLURM_FLAGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        shift
        break
    elif [[ "$1" == --* ]]; then
        SLURM_FLAGS+=("$1")
        if [[ "$1" != *=* && $# -gt 1 && "$2" != --* && "$2" != -* ]]; then
            SLURM_FLAGS+=("$2")
            shift
        fi
        shift
    elif [[ "$1" =~ ^-[A-Za-z]$ ]]; then
        SLURM_FLAGS+=("$1")
        if [[ $# -gt 1 && "$2" != --* && "$2" != -* ]]; then
            SLURM_FLAGS+=("$2")
            shift
        fi
        shift
    else
        break
    fi
done

# 3. Check for primus-run args
if [[ $# -eq 0 ]]; then
    print_usage
    exit 2
fi

# 4. Logging and launch
echo "[primus-cli-slurm] Executing: $LAUNCH_CMD ${SLURM_FLAGS[*]} $ENTRY $*"
exec "$LAUNCH_CMD" "${SLURM_FLAGS[@]}" "$ENTRY" "$@"
