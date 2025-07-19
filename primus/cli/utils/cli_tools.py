###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import sys

def apply_env(envs):
    for kv in envs:
        k, v = kv.split("=", 1)
        os.environ[k] = v

def add_custom_paths(paths):
    for p in paths:
        if p and p not in sys.path:
            print(f"🔧 Adding --custom-path: {p}")
            sys.path.insert(0, p)