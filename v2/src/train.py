import hydra

import pytorch_lightning as L

from omegaconf import DictConfig, OmegaConf


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

    Notes
    -----
    - Prints the resolved configuration in YAML format at startup.
    - If a checkpoint path is specified in the configuration, training
      resumes from that checkpoint.
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
