#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

print_usage() {
cat <<EOF
Primus Slurm Launcher

Usage:
    primus-cli slurm [srun|sbatch] [SLURM_FLAGS...] -- <primus-run args>

Supported SLURM_FLAGS:
    --nodes, -N <num_nodes>           Number of nodes
    --partition, -P <name>            Partition/queue name
    --nodelist <nodes>                List of nodes
    --reservation <name>              Reservation name
    --account, -A <name>              Account name
    --qos, -q <qos>                   Quality of service
    --time, -t <time>                 Maximum job time (minutes or HH:MM:SS)
    --job-name, -J <name>             Job name
    --output <file>                   Write stdout to file
    --error <file>                    Write stderr to file

    # All above options also support --flag=value form (e.g. --output=run.log)

Examples:
    primus-cli slurm srun -N 4 -p AIG_Model -- container -- train pretrain --config exp.yaml
    primus-cli slurm sbatch --output=run.log -N 2 -- container -- benchmark gemm -M 4096 -N 4096 -K 4096

Notes:
    - [srun|sbatch] is optional, defaults to srun if not specified.
    - All SLURM_FLAGS before '--' are passed directly to Slurm.
    - Everything after '--' is passed to the per-node Primus entry (container/native, and Primus CLI args).
    - For unsupported or extra Slurm options, pass them after -- (they'll be ignored by our wrapper).

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

# 3. Check for primus-run args
if [[ $# -eq 0 ]]; then
    print_usage
    exit 2
fi

# 4. Logging and launch
echo "[primus-cli-slurm] Executing: $LAUNCH_CMD ${SLURM_FLAGS[*]} $ENTRY -- $*"
exec "$LAUNCH_CMD" "${SLURM_FLAGS[@]}" "$ENTRY" -- "$@"
