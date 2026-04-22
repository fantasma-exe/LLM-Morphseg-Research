import hydra
import torch

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from morphseg.utils import get_device, dictconfig_to_dict


class Predictor:
    """
    Predictor for generating model outputs from text prompts.

    This class handles loading a pre-trained model and tokenizer,
    moving the model to the appropriate device, and generating predictions
    for a batch of prompts.

    Parameters
    ----------
    model_cfg : omegaconf.DictConfig
        Configuration used to instantiate the model.

    checkpoint_path : str
        Path to the model checkpoint file.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to the model.

    device_type : str
        The device type on which to run the model (e.g., 'cuda' or 'cpu').

    generation_kwargs : omegaconf.DictConfig
        Configuration of model generating.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        checkpoint_path: str,
        tokenizer: PreTrainedTokenizer,
        device_type: str,
        generation_kwargs: DictConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.generation_kwargs = dictconfig_to_dict(generation_kwargs)

        self.model = hydra.utils.instantiate(
            model_cfg, tokenizer=self.tokenizer, _recursive_=False
        )

        self.device = get_device(device_type)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.eval().to(self.device)

    def predict_batch(self, inputs: dict[str, torch.Tensor]) -> list[str]:
        """
        Generate predictions for a batch of prompts.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Typical batch passed to forward method.

        Returns
        -------
        list[str]
            Predicted outputs corresponding to each input.
        """

        batch = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = batch["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.model.generate(
                **batch,
                **self.generation_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_tokens = outputs[:, input_length:]
            decoded = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            return [text.strip() for text in decoded]
