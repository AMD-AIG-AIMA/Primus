###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def get_batch_func(self):
        raise NotImplementedError

    @abstractmethod
    def get_loss_func(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataset_and_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def get_forward_step_func(self):
        raise NotImplementedError
