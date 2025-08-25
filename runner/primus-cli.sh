#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# Check argument count, print usage if not enough arguments
[[ $# -ge 1 ]] || { echo "Usage: primus-cli {launch|slurm|direct} ..."; exit 2; }
mode="$1"; shift

case "$mode" in
    slurm)
        # ========= Slurm mode =========
        # All Slurm argument parsing and launch mode detection handled here
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        ENTRY="$SCRIPT_DIR/primus-cli-slurm.sh"
        SLURM_FLAGS=()

         # ==== New: detect sbatch or srun as second arg ====
        LAUNCH_MODE="srun"   # default
        if [[ "${1:-}" == "sbatch" || "${1:-}" == "srun" ]]; then
            LAUNCH_MODE="$1"
            shift
        fi

        # Parse CLI options for slurm flags and launch mode
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --output|--error|-p|-A|-q|-t|-J|-N)
                    SLURM_FLAGS+=("$1" "$2"); shift 2;;
                --nodes|--nodelist|--partition|--reservation|--qos|--time|--job-name)
                    SLURM_FLAGS+=("$1" "$2"); shift 2;;
                --output=*|--error=*|-p=*|-A=*|-q=*|-t=*|-J=*|-N=*)
                    SLURM_FLAGS+=("$1"); shift;;
                --nodes=*|--nodelist=*|--partition=*|--reservation=*|--qos=*|--time=*|--job-name=*)
                    SLURM_FLAGS+=("$1"); shift;;
                --) shift; break;;
                *)  break;;
            esac
        done

        [[ $# -gt 0 ]] || { echo "Usage: primus-cli slurm [sbatch|srun] [slurm-flags] -- <primus args>"; exit 2; }

        echo "[primus-cli] Executing: $LAUNCH_MODE ${SLURM_FLAGS[*]} $ENTRY -- $*"
        # exec "$LAUNCH_MODE" "${SLURM_FLAGS[@]}" "$ENTRY" -- "$@"
        ;;

    container)
        # Local launch mode: directly invoke primus-cli-launch.sh
        # This script handles environment setup and training start logic
        exec bash "$(dirname "$0")/primus-cli-container.sh" "$@"
        ;;

    direct)
        exec bash "$(dirname "$0")/primus-cli-direct.sh" "$@"
        ;;

    *)
        echo "Unknown mode: $mode"; exit 2
        ;;
esac
