import hydra

import pytorch_lightning as L

from omegaconf import DictConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    BatchEncoding,
)


class BaseDataModule(L.LightningDataModule):
    """
    PyTorch Lightning data module for morphological segmentation training.

    The module loads a JSON dataset, converts each example into a prompt for
    instruction-style language model training, tokenizes the prompt using a
    Hugging Face tokenizer, and constructs a training DataLoader.

    The data pipeline follows the Hydra configuration pattern, where
    tokenizer, collator, dataset paths, and training parameters are defined
    in configuration files.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration describing dataset paths, preprocessing and
        dataloader parameters. Expected structure:

        - paths.train_path : str
            Path to JSON dataset.
        - training.batch_size : int
            Training batch size.
        - training.max_seq_length : int
            Maximum sequence length during tokenization.
        - collator : DictConfig
            Hydra config for data collator.

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer used to tokenize prompts.

    Attributes
    ----------
    _cfg : DictConfig
        Data configuration.

    _tokenizer : PreTrainedTokenizer
        Tokenizer used for text preprocessing.

    _data_collator : Callable
        Batch collation function instantiated via Hydra.

    _train_dataset : datasets.Dataset
        Tokenized training dataset.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer

        self.data_collator = hydra.utils.instantiate(
            cfg.collator, tokenizer=self.tokenizer
        )

    def _build_prompt(self, word: str, answer: str | None = None) -> str:
        """
        Build an instruction-style prompt for morphological segmentation.

        The prompt asks the model to split a word into morphemes and label
        each morpheme type.

        Parameters
        ----------
        word : str
            Word to be segmented.

        answer : str | None, optional
            Reference segmentation to append to the prompt.

        Returns
        -------
        str
            Formatted instruction prompt.
        """
        template = self.cfg.prompt_template
        prompt = template.format(word=word)

        if answer is not None:
            prompt += f"{answer}{self.tokenizer.eos_token}"

        return prompt

    def setup(self, stage: str | None = None) -> None:
        """
        Prepare the dataset for training.

        This method loads the raw dataset from JSON files and applies
        tokenization and prompt construction.

        Parameters
        ----------
        stage : str | None, optional
            Lightning stage indicator (e.g. "fit", "test").
            Currently unused.

        Returns
        -------
        None
        """

        data_files = {"train": self.cfg.path.train_path}

        val_path = self.cfg.paths.get("val_path")
        if val_path:
            data_files["val"] = self.cfg.path.val_path

        dataset = load_dataset("json", data_files=data_files)

        def tokenize_function(example: dict[str, str]) -> BatchEncoding:
            """
            Convert a raw dataset example into tokenized prompt format.

            Parameters
            ----------
            example : dict[str, str]
                Raw dataset example with structure::

                    {
                        "input": "word",
                        "output": "segmentation"
                    }

            Returns
            -------
            BatchEncoding
                Tokenized prompt with labels for causal LM training.
            """

            prompt = self._build_prompt(example["input"], example["output"])

            outputs = self.tokenizer(prompt, **self.cfg.tokenizer_kwargs)

            outputs["labels"] = outputs["input_ids"].copy()  # type: ignore

            return outputs

        self.train_dataset = dataset["train"].map(
            tokenize_function,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing train dataset",
            num_proc=self.cfg.get("num_proc", 1),
        )

        if "val" in dataset:
            self.val_dataset = dataset["val"].map(
                tokenize_function,
                remove_columns=dataset["val"].column_names,
                desc="Tokenizing val dataset",
                num_proc=self.cfg.get("num_proc", 1),
            )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the training dataset.
        """

        return DataLoader(
            self.train_dataset,  # type: ignore
            collate_fn=self.data_collator,
            **self.cfg.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader | None:
        """
        Create the validation DataLoader.

        Returns
        -------
        DataLoader | None
            PyTorch DataLoader for the validation dataset.
        """

        if not hasattr(self, "val_dataset"):
            return None

        return DataLoader(
            self.val_dataset,  # type: ignore
            collate_fn=self.data_collator,
            **self.cfg.dataloader_kwargs,
        )
