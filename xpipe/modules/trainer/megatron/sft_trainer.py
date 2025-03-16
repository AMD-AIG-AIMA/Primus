from .trainer import MegatronTrainer


class MegatronSFTTrainer(MegatronTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "sft_trainer"
        super().__init__(*args, **kwargs)

    def get_batch_func(self):
        raise NotImplementedError

    def get_loss_func(self):
        raise NotImplementedError

    def build_dataset_and_tokenizer(self):
        raise NotImplementedError

    def get_forward_step_func(self):
        raise NotImplementedError
