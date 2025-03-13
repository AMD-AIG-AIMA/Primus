###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################


from .trainer import MegatronTrainer


class MegatronPretrainTrainer(MegatronTrainer):

    def get_batch_func(self):
        raise NotImplementedError

    def get_loss_func(self):
        raise NotImplementedError

    def build_dataset_and_tokenizer(self):
        raise NotImplementedError

    def get_forward_step_func(self):
        raise NotImplementedError
