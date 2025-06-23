###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


import argparse
import unittest
from types import SimpleNamespace

from primus.core.launcher.parser import (
    PrimusParser,
    _check_keys_exist,
    _deep_merge_namespace,
    _parse_kv_overrides,
)
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

    def test_parse_equal_format(self):
        args = ["--a=1", "--b.c=2", "--flag"]
        result = _parse_kv_overrides(args)
        expected = {"a": 1, "b": {"c": 2}, "flag": True}
        self.assertEqual(result, expected)

    def test_parse_space_format(self):
        args = ["--a", "1", "--b.c", "2", "--flag"]
        result = _parse_kv_overrides(args)
        expected = {"a": 1, "b": {"c": 2}, "flag": True}
        self.assertEqual(result, expected)

    def test_override_check_pass(self):
        # Simulate pre_trainer config
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2), flag=False)
        overrides = {"a": 123, "b": {"c": 999}, "flag": True}
        # Should not raise
        _check_keys_exist(ns, overrides)

    def test_override_check_fail(self):
        ns = SimpleNamespace(a=1)
        overrides = {"missing": 10}
        with self.assertRaises(AssertionError) as context:
            _check_keys_exist(ns, overrides)
        self.assertIn("Override key 'missing' does not exist", str(context.exception))

    def test_deep_merge_namespace(self):
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2), flag=False)
        overrides = {"a": 10, "b": {"c": 20}, "flag": True}
        _deep_merge_namespace(ns, overrides)
        self.assertEqual(ns.a, 10)
        self.assertEqual(ns.b.c, 20)
        self.assertEqual(ns.flag, True)


if __name__ == "__main__":
    unittest.main()
