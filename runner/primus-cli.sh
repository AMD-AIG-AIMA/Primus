#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_help() {
cat << EOF
Primus Unified Launcher CLI

Usage:
    primus-cli <mode> [mode-args] [-- primus-args]

Supported modes:
    slurm      Launch distributed training via Slurm cluster (supports sbatch/srun)
    container  Launch training inside a managed container (Docker/Podman/Singularity)
    direct     Directly launch training in the current environment (host or container)

Examples:
    # Launch via Slurm with srun (default), 4 nodes
    primus-cli slurm srun -N 4 -- benchmark gemm -M 4096 -N 4096 -K 4096

    # Launch via Slurm using sbatch, partition named AIG_Model
    primus-cli slurm sbatch -N 8 -p AIG_Model -- train pretrain --config exp.yaml

    # Run in managed container (single node or for debugging)
    primus-cli container -- benchmark gemm -M 4096 -N 4096 -K 4096

    # Run directly on host or inside an existing container
    primus-cli direct -- train pretrain --config exp.yaml

Advanced:
    # Use --container or --host flags with slurm to control entry script (optional)
    primus-cli slurm srun --container -N 2 -- benchmark gemm ...
    primus-cli slurm sbatch --host -N 1 -- train pretrain ...

Notes:
- [--] separates mode-specific flags from Primus CLI arguments.
       Everything after -- is passed to the Primus Python CLI (e.g., benchmark, train).
- For detailed Primus training and benchmarking options, run: primus-cli direct -- --help

EOF
}

# Show help if called with -h, --help, or no arguments
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

# # Check argument count, print usage if not enough arguments
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
        exec "$LAUNCH_MODE" "${SLURM_FLAGS[@]}" "$ENTRY" -- "$@"
        ;;

    container)
        # Container mode: dispatch to primus-cli-container.sh for containerized execution
        script_path="$(dirname "$0")/primus-cli-container.sh"
        echo "[primus-cli] Executing: bash $script_path $*"
        exec bash "$script_path" "$@"
        ;;

    direct)
        # Direct (host) mode: run workflow directly on the host, no container
        script_path="$(dirname "$0")/primus-cli-direct.sh"
        echo "[primus-cli] Executing: bash $script_path $*"
        exec bash "$script_path" "$@"
        ;;

    *)
        echo "Unknown mode: $mode"; exit 2
        ;;
esac
