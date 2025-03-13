import xpipe
from xpipe.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

if __name__ == "__main__":
    xpipe.init()

    trainer = MegatronPretrainTrainer(
        module_name="deepseek_v3_pretrain",
        xpipe_config=xpipe.get_xpipe_config(),
    )

    trainer.init()
    trainer.run()
