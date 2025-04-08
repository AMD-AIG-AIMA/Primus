###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


import os
import subprocess
import sys
import time
import unittest

from primus.core.utils import logger
from tests.utils import PrimusUT


class TestMegatronTrainer(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pretrain(self):
        shell_entry = "examples/megatron/run_pretrain.sh"
        do_print_at_runtime = False
        run_stdout = subprocess.PIPE if not do_print_at_runtime else sys.stdout
        run_stderr = subprocess.PIPE if not do_print_at_runtime else sys.stderr
        try:
            logger.info(f"Begin run {shell_entry}...")
            start = time.time()
            result = subprocess.run(
                ["bash", f"{shell_entry}"],
                check=True,
                stdout=run_stdout,
                stderr=run_stderr,
                text=True,
            )
            logger.info(f"End run {shell_entry}, time={time.time()-start:.3f} s")
            if not do_print_at_runtime:
                logger.info(f"Training log path: ut_out/logs/UT-{self.__class__.__name__}")

            logger.debug(f"Standard Output:\n {result.stdout}")
            logger.debug(f"Standard Error:\n {result.stderr}")
        except subprocess.CalledProcessError as e:
            os.environ["SCRIPT_ERROR"] = e.stderr.strip()
            assert False, f"Shell script failed: {os.environ['SCRIPT_ERROR']}"


if __name__ == "__main__":
    unittest.main(buffer=False)
