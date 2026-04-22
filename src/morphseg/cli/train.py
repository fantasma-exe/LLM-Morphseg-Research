import hydra

import torch.multiprocessing as mp

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Train a PyTorch Lightning model using a Hydra configuration.

    The function instantiates the model, datamodule, logger, and callbacks
    from the configuration, then runs training via a Lightning Trainer.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object describing the training setup.
        The expected configuration structure and available options are
        defined in ``configs/train.yaml``.
    """
    
    mp.set_start_method('spawn', force=True)

    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.model_name)

    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer, _recursive_=False)
    datamodule = hydra.utils.instantiate(
        cfg.datamodule, tokenizer=tokenizer, _recursive_=False
    )

    logger = hydra.utils.instantiate(cfg.logger)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume_from_checkpoint,
    )


if __name__ == "__main__":
    train()
