import hydra
import os

import pytorch_lightning as L

from omegaconf import DictConfig
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from morphseg.utils import dictconfig_to_dict, get_datamodule_hash


class MorphologyDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for morphological segmentation tasks.

    This module loads datasets from JSON files, applies tokenization using a
    provided tokenizer, caches the tokenized dataset to disk, and prepares PyTorch DataLoaders 
    for training and validation.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to encode input and target texts.

    data_paths : omegaconf.DictConfig
        Mapping of dataset splits to file paths (e.g., {"train": ..., "val": ...}).
        
    cache_dir : str
        Base directory where tokenized datasets will be cached.

    prompt_template : str
        Template string used to construct the input prompt. Should contain
        a placeholder for the input word (e.g., "{word}").

    train_dataloader_cfg : omegaconf.DictConfig
        Hydra сonfiguration for the training DataLoader.

    val_dataloader_cfg : omegaconf.DictConfig
        Hydra сonfiguration for the validation DataLoader.

    tokenizer_header_cfg : omegaconf.DictConfig
        Tokenizer arguments for encoding the input (prompt).

    tokenizer_target_cfg : omegaconf.DictConfig
        Tokenizer arguments for encoding the target output.

    num_proc : int, default=1
        Number of processes used for dataset preprocessing.

    collator_cfg : omegaconf.DictConfig | None, default=None
        Hydra config for instantiating a data collator.
        
    logic_version : str
        Version of datamodule implementation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_paths: DictConfig,
        cache_dir: str,
        prompt_template: str,
        train_dataloader_cfg: DictConfig,
        val_dataloader_cfg: DictConfig,
        tokenizer_header_cfg: DictConfig,
        tokenizer_target_cfg: DictConfig,
        num_proc: int = 1,
        collator_cfg: DictConfig | None = None,
        logic_version: str = "v2_version"
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.num_proc = num_proc
        self.logic_version = logic_version

        self.data_files = dictconfig_to_dict(data_paths, resolve=True)
        self.tokenizer_header_kwargs = dictconfig_to_dict(tokenizer_header_cfg)
        self.tokenizer_target_kwargs = dictconfig_to_dict(tokenizer_target_cfg)
        self.train_cfg = dictconfig_to_dict(train_dataloader_cfg)
        self.val_cfg = dictconfig_to_dict(val_dataloader_cfg)
        
        self.cache_id = get_datamodule_hash(self.data_files, tokenizer.name_or_path, self.prompt_template, logic_version)
        self.cache_dir = os.path.join(cache_dir, self.cache_id)

        if collator_cfg is not None:
            self.data_collator = hydra.utils.instantiate(
                collator_cfg, tokenizer=self.tokenizer
            )
        else:
            self.data_collator = None
            
    def _tokenize_fn(self, example: dict[str, str]) -> dict:
        """
        Tokenize a single dataset example.

        Constructs the input sequence as:
            [BOS] + formatted prompt + target + [EOS]

        Parameters
        ----------
        example : dict[str, str]
            Dictionary with keys:
            - "input": source word
            - "output": target segmentation

        Returns
        -------
        dict
            Dictionary containing:
            - input_ids : list[int]
            - labels : list[int]
            - attention_mask : list[int]
        """
        
        header_text = f"{self.tokenizer.bos_token}{self.prompt_template.format(word=example['input'])}"
        target_text = f"{example['output']}{self.tokenizer.eos_token}"

        header_ids = self.tokenizer(
            header_text,
            **self.tokenizer_header_kwargs,
        )["input_ids"]
        target_ids = self.tokenizer(
            target_text,
            **self.tokenizer_target_kwargs,
        )["input_ids"]

        input_ids = header_ids + target_ids  # type: ignore
        labels = [-100] * len(header_ids) + target_ids  # type:ignore
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_raw_text": header_text,
        }
        
    def prepare_data(self) -> None:
        """
        Preprocess and cache the dataset.
        """
        
        if os.path.exists(self.cache_dir):
            return
        
        raw_dataset = load_dataset("json", data_files=self.data_files)
        
        tokenized_dataset = raw_dataset.map(
            self._tokenize_fn,
            remove_columns=raw_dataset["train"].column_names,
            num_proc=self.num_proc,
            desc="Tokenizing dataset",
        )
        
        tokenized_dataset.save_to_disk(self.cache_dir)

    def setup(self, stage: str | None = None) -> None:
        """
        Load and preprocess datasets.

        This method loads JSON datasets using Hugging Face Datasets, applies
        tokenization, and prepares train and validation splits.

        Parameters
        ----------
        stage : str | None, default=None
            Stage identifier used by Lightning ("fit", "validate", etc.).
            If None, all relevant datasets are prepared.
        """

        tokenized_dataset = load_from_disk(self.cache_dir)

        if stage == "fit" or stage is None:
            self.train_ds = tokenized_dataset["train"]
            if "val" in tokenized_dataset:
                self.val_ds = tokenized_dataset["val"]


    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for the training dataset.
        """

        return DataLoader(
            self.train_ds,  # type: ignore
            collate_fn=self.data_collator,
            **self.train_cfg,
        )

    def val_dataloader(self) -> DataLoader | None:
        """
        Create the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader | None
            DataLoader for the validation dataset.
        """

        if not hasattr(self, "val_ds"):
            return None

        return DataLoader(self.val_ds, collate_fn=self.data_collator, **self.val_cfg)  # type:ignore