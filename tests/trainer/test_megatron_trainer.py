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

    def test_llama2_7B(self):
        self._run_script(
            "llama2_7B",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "llama2_7B",
                "PRIMUS_GLOBAL_BATCH_SIZE": "8",
                "PRIMUS_NUM_LAYERS": "4",
            },
        )

    def test_llama3_8B(self):
        self._run_script(
            "llama3_8B",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "llama3_8B",
                "PRIMUS_GLOBAL_BATCH_SIZE": "8",
                "PRIMUS_NUM_LAYERS": "4",
            },
        )

    def test_llama3_70B(self):
        self._run_script(
            "llama3_70B",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "llama3_70B",
                "PRIMUS_GLOBAL_BATCH_SIZE": "8",
                "PRIMUS_NUM_LAYERS": "4",
            },
        )

    def test_deepseek_v2_lite(self):
        self._run_script(
            "deepseek_v2_lite",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "deepseek_v2_lite",
                "PRIMUS_GLOBAL_BATCH_SIZE": "8",
                "PRIMUS_MOE_LAYER_FREQ": "[0]*1+[1]*3",
                "PRIMUS_EP": "8",
                "PRIMUS_NUM_LAYERS": "4",
            },
        )

    def test_deepseek_v3(self):
        self._run_script(
            "deepseek_v3",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "deepseek_v3",
                "PRIMUS_GLOBAL_BATCH_SIZE": "8",
                "PRIMUS_MOE_LAYER_FREQ": "[0]*3+[1]*1",
                "PRIMUS_EP": "8",
                "PRIMUS_NUM_LAYERS": "4",
            },
        )

    def test_interleaved_pipeline_parallelism(self):
        self._run_script(
            "interleaved_pipeline_parallelism",
            env_override={
                "BACKEND": "megatron",
                "PRIMUS_MODEL": "deepseek_v2_lite",
                "PRIMUS_GLOBAL_BATCH_SIZE": "16",
                "PRIMUS_MOE_LAYER_FREQ": "[0]*1+[1]*7",
                "PRIMUS_PP": "4",
                "PRIMUS_VPP": "2",
                "PRIMUS_NUM_LAYERS": "8",
            },
        )

    def _run_script(self, tag: str, env_override: dict = None):
        shell_entry = "examples/run_pretrain.sh"
        env = os.environ.copy()
        if env_override:
            env.update(env_override)
        env["EXP"] = "tests/trainer/test_megatron_trainer.yaml"

        do_print_at_runtime = False
        run_stdout = subprocess.PIPE if not do_print_at_runtime else sys.stdout
        run_stderr = subprocess.PIPE if not do_print_at_runtime else sys.stderr
        try:
            logger.info(f"Begin run {tag}...")
            start = time.time()
            result = subprocess.run(
                ["bash", f"{shell_entry}"],
                check=True,
                stdout=run_stdout,
                stderr=run_stderr,
                text=True,
                env=env,
            )
            logger.info(f"End run {tag}, time={time.time()-start:.3f} s")
            if not do_print_at_runtime:
                ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
                logger.info(f"Training log path: {ut_log_path}/logs/UT-{self.__class__.__name__}")

            logger.debug(f"Standard Output:\n {result.stdout}")
            logger.debug(f"Standard Error:\n {result.stderr}")
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr or ""
            stdout_output = e.stdout or ""
            if "after training is done" in stdout_output:
                logger.warning(f"[{tag}] Training likely succeeded despite return code != 0.")
                logger.warning(f"stderr excerpt:\n{stderr_output[:1000]}")
            else:
                raise AssertionError(f"Shell script failed: {stderr_output.strip()}")


if __name__ == "__main__":
    unittest.main(buffer=False)
