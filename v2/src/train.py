import hydra

import pytorch_lightning as L

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    logger = hydra.utils.instantiate(cfg.logger)

    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume_from_checkpoint,
    )


if __name__ == "__main__":
    train()
