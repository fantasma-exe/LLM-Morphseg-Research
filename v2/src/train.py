import hydra

import pytorch_lightning as L

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Train a PyTorch Lightning model using Hydra configuration.

    This function initializes the model, datamodule, logger, and callbacks
    based on the provided configuration, then trains the model using a
    Lightning Trainer.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing sections for the model, datamodule,
        logger, callbacks, trainer, and training settings.

    Notes
    -----
    - The function prints the full configuration in YAML format at startup.
    - If `cfg.training.resume_from_checkpoint` is set, training will resume
      from the specified checkpoint.
    """
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
