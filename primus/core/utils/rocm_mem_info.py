###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import subprocess


def get_rocm_smi_mem_info(device_id: int):
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", f"-d={device_id}"], text=True)
    except FileNotFoundError:
        raise RuntimeError("rocm-smi not found, please ensure ROCm is installed and in PATH")

    # mem in Bytes
    total_mem, used_mem = None, None
    for line in out.splitlines():
        if "Total Memory" in line:
            total_mem = int(line.split(":")[-1].strip())
        elif "Total Used Memory" in line:
            used_mem = int(line.split(":")[-1].strip())

    assert total_mem is not None
    assert used_mem is not None
    free_mem = total_mem - used_mem

    return total_mem, used_mem, free_mem
