###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


import argparse
import unittest

from primus.core.launcher.parser import PrimusParser
from primus.core.utils import logger
from tests.utils import PrimusUT


class TestPrimusParser(PrimusUT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        self.config_parser = PrimusParser()
        self.cli_args = argparse.Namespace()

    def tearDown(self):
        pass

    def parse_config(self, cli_args: argparse.Namespace):
        exp_config = self.config_parser.parse(cli_args)
        return exp_config

    def test_exp_configs(self):
        exps = [
            "examples/megatron/exp_pretrain.yaml",
        ]

        for exp in exps:
            self.cli_args.exp = exp
            logger.info(f"test exp config: {exp}")
            logger.debug(f"============================")
            exp_config = self.parse_config(self.cli_args)
            logger.debug(f"exp config: \n{exp_config}")
            logger.debug(f"============================")


if __name__ == "__main__":
    unittest.main()
