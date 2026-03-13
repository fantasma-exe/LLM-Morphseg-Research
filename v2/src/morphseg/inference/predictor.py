import hydra
import torch

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer


class Predictor:
    """
    Predictor for generating model outputs from text prompts.

    This class handles loading a pre-trained model and tokenizer,
    moving the model to the appropriate device, and generating predictions
    for a batch of prompts.

    Parameters
    ----------
    model_cfg : DictConfig
        Configuration used to instantiate the model.

    checkpoint_path : str
        Path to the model checkpoint file.

    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        checkpoint_path: str,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self.tokenizer = tokenizer

        self.model = hydra.utils.instantiate(
            model_cfg, tokenizer=self.tokenizer, _recursive_=False
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.eval().freeze()
        self.model.to(self.device)

    def predict_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate predictions for a batch of prompts.

        Parameters
        ----------
        prompts : list of str
            A list of prompt strings to feed into the model.

        Returns
        -------
        list of str
            Predicted outputs corresponding to each prompt. If the model
            generates text containing "### Ответ:", only the part after
            this marker is returned; otherwise, the full output is returned.
        """
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=64,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            results = []
            for text in decoded:
                if "### Ответ:" in text:
                    results.append(text.split("### Ответ:")[1].strip())
                else:
                    results.append(text.strip())

            return results
