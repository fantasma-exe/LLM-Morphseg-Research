import hydra

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict")
def predict(cfg: DictConfig) -> None:
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
        ``configs/predict.yaml``.

    Notes
    -----
    - Prints the resolved inference configuration in YAML format.
    - Ensures that the tokenizer has a ``pad_token``. If it is not set,
      the ``eos_token`` is used instead.
    """
    print(OmegaConf.to_yaml(cfg.inference))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline = hydra.utils.instantiate(
        cfg.inference.pipeline,
        inference_cfg=cfg.inference,
        input_strategy=cfg.inference.input,
        output_strategy=cfg.inference.output,
        predictor={"tokenizer": tokenizer},
    )

    pipeline.run()


if __name__ == "__main__":
    predict()
