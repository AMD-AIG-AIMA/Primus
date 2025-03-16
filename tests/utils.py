###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


import os
import unittest

from xpipe.core.utils import logger


class XPipeUT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        ut_log_path = os.environ.get("UT_LOG_PATH", "output/ut")
        logger_cfg = logger.LoggerConfig(
            exp_root_path=ut_log_path,
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name=f"UT-{cls.__name__}",
            file_sink_level="DEBUG",
            stderr_sink_level="INFO",
            node_ip="localhost",
            rank=os.environ.get("RANK", 0),
            world_size=os.environ.get("WORLD_SIZE", 1),
        )
        logger.setup_logger(logger_cfg, is_head=False)

    def setUp(self):
        pass

    def tearDown(self):
        pass
