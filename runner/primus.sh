#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

[[ $# -ge 1 ]] || { echo "Usage: primus-run {launch|slurm|direct} ..."; exit 2; }
mode="$1"; shift
case "$mode" in
  launch)  exec bash "$(dirname "$0")/primus-run-launch.sh"  "$@" ;;
  slurm)  exec bash "$(dirname "$0")/primus-run-slurm.sh"  "$@" ;;
  direct) exec bash "$(dirname "$0")/primus-run-direct.sh" "$@" ;;
  *) echo "Unknown mode: $mode"; exit 2;;
esac
