import hydra

from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(cfg: DictConfig) -> None:
    """
    Run an inference pipeline using a Hydra configuration.

    This function sets up the tokenizer, initializes the inference pipeline
    with specified input and output strategies, and executes the pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model and inference settings, including:
        - `cfg.model.cfg.model_name`: Name or path of the pre-trained model.
        - `cfg.inference.pipeline`: Pipeline configuration for inference.
        - `cfg.inference.input`: Input strategy configuration.
        - `cfg.inference.output`: Output strategy configuration.

    Notes
    -----
    - Prints the inference configuration in YAML format.
    - Ensures that the tokenizer has a `pad_token`; if not, it is set to
      the `eos_token`.
    """
    print(OmegaConf.to_yaml(cfg.inference))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline = hydra.utils.instantiate(
        cfg.inference.pipeline,
        input_strategy=cfg.inference.input,
        output_strategy=cfg.inference.output,
        predictor={"tokenizer": tokenizer},
    )

    pipeline.run()


if __name__ == "__main__":
    predict()
