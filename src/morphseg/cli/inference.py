import hydra

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def inference(cfg: DictConfig) -> None:
    """
    Run the inference pipeline defined in the Hydra configuration.

    The function initializes the tokenizer, instantiates the inference
    pipeline via Hydra, and executes it. Input and output handling are
    delegated to configurable strategies defined in the inference config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object for the inference run. The expected
        configuration structure and available options are defined in
        ``configs/inference.yaml``.

    """

    print(OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_cfg.model_name)

    ckpt_loader = hydra.utils.instantiate(cfg.inference.loader)

    pipeline = hydra.utils.instantiate(
        cfg.pipeline,
        predictor={
            "checkpoint_path": ckpt_loader.download_checkpoint(),
            "tokenizer": tokenizer,
        },
    )

    pipeline.run()


if __name__ == "__main__":
    inference()
