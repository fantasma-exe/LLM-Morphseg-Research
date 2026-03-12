import hydra


from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer, _recursive_=False)
    datamodule = hydra.utils.instantiate(
        cfg.datamodule, tokenizer=tokenizer, _recursive_=False
    )

    logger = hydra.utils.instantiate(cfg.logger)

    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]

    print(OmegaConf.to_yaml(cfg.trainer))
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume_from_checkpoint,
    )

    # Temporary solution to connect inference and train
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        with open("last_best_ckpt.txt", "w") as f:
            f.write(best_path)
        print(f"Training finished. Best checkpoint saved at: {best_path}")


if __name__ == "__main__":
    train()
