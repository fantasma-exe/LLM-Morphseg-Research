import hydra

import pytorch_lightning as L

from omegaconf import DictConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    BatchEncoding,
)


class MorphologyDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for morphological segmentation training.

    Loads a JSON dataset, converts each example into an instruction-style prompt,
    tokenizes the text, and provides PyTorch DataLoaders.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer used to tokenize prompts.

    paths : DictConfig
        Dictionary containing 'train_data' and 'val_data' file paths.

    prompt_template : str
        String template for the instruction prompt (must contain {word}).

    dataloader_kwargs : DictConfig
        Keyword arguments passed to the DataLoader (batch_size, num_workers, etc.).

    tokenizer_kwargs : DictConfig
        Keyword arguments for the tokenizer (max_length, padding, etc.).

    num_proc : int, optional
        Number of processes for data processing, by default 1.

    collator_cfg : DictConfig, optional
        Hydra configuration for the data collator. If None, a default setup
        should be handled manually.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        paths: DictConfig,
        prompt_template: str,
        dataloader_kwargs: DictConfig,
        tokenizer_kwargs: DictConfig,
        num_proc: int = 1,
        collator_cfg: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer
        self.paths = paths
        self.prompt_template = prompt_template
        self.dataloader_kwargs = dataloader_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.num_proc = num_proc

        if collator_cfg:
            self.data_collator = hydra.utils.instantiate(
                collator_cfg, tokenizer=self.tokenizer
            )
        else:
            self.data_collator = None

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
        template = self.prompt_template
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

        data_files = {"train": self.paths.train_path}

        val_path = self.paths.get("val_path")
        if val_path:
            data_files["val"] = self.paths.val_path

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

            outputs = self.tokenizer(prompt, **self.tokenizer_kwargs)  # type: ignore

            outputs["labels"] = outputs["input_ids"].copy()  # type: ignore

            return outputs

        self.train_dataset = dataset["train"].map(
            tokenize_function,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing train dataset",
            num_proc=self.num_proc,
        )

        if "val" in dataset:
            self.val_dataset = dataset["val"].map(
                tokenize_function,
                remove_columns=dataset["val"].column_names,
                desc="Tokenizing val dataset",
                num_proc=self.num_proc,
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
            **self.dataloader_kwargs,  # type: ignore
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
            **self.dataloader_kwargs,  # type: ignore
        )
