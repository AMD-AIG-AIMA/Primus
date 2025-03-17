###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

from abc import ABC, abstractmethod

import torch
from megatron.core.models.gpt import GPTModel


class BaseTrainer(ABC):
    # def get_batch_func(self):
    @abstractmethod
    def get_batch(self, data_iterator):
        raise NotImplementedError

    # def get_loss_func(self):
    @abstractmethod
    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        raise NotImplementedError

    # def get_forward_step_func(self):
    @abstractmethod
    def forward_step(self, data_iterator, model: GPTModel):
        raise NotImplementedError
