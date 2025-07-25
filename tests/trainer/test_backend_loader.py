###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
import tempfile
import pytest
from unittest import mock

from primus.train import setup_backend_path, load_backend_trainer


def test_setup_backend_path_with_valid_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        sys_path_backup = list(sys.path)
        try:
            setup_backend_path(tmpdir)
            assert tmpdir in sys.path
        finally:
            sys.path = sys_path_backup  # restore


def test_setup_backend_path_with_invalid_path():
    with pytest.raises(FileNotFoundError):
        setup_backend_path("/non/existent/path/to/backend")


def test_setup_backend_path_with_list():
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        sys_path_backup = list(sys.path)
        try:
            setup_backend_path([tmpdir1, tmpdir2])
            assert tmpdir1 in sys.path
            assert tmpdir2 in sys.path
        finally:
            sys.path = sys_path_backup


def test_load_backend_trainer_megatron(monkeypatch):
    class DummyMegatron:
        pass

    monkeypatch.setitem(
        sys.modules,
        "primus.modules.trainer.megatron.pre_trainer",
        mock.Mock(MegatronPretrainTrainer=DummyMegatron)
    )
    trainer_cls = load_backend_trainer("megatron")
    assert trainer_cls is DummyMegatron


def test_load_backend_trainer_invalid():
    with pytest.raises(ValueError):
        load_backend_trainer("unknown_framework")
