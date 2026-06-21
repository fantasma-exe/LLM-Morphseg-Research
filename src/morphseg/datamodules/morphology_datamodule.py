import hydra
import os

import pytorch_lightning as L

from omegaconf import DictConfig
from datasets import load_dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from functools import partial

from morphseg.utils import (
    dictconfig_to_dict,
    get_datamodule_hash,
)


class MorphologyDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_paths: DictConfig,
        cache_dir: str,
        prompt_template: str,
        train_dataloader_cfg: DictConfig,
        val_dataloader_cfg: DictConfig,
        tokenizer_prompt_cfg: DictConfig,
        tokenizer_full_text_cfg: DictConfig,
        train_collator_cfg: DictConfig,
        val_collator_cfg: DictConfig,
        logic_version: str = "v2_version",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.logic_version = logic_version

        self.data_files = dictconfig_to_dict(data_paths, resolve=True)
        self.tokenizer_prompt_kwargs = dictconfig_to_dict(tokenizer_prompt_cfg)
        self.tokenizer_full_text_kwargs = dictconfig_to_dict(tokenizer_full_text_cfg)
        self.train_cfg = dictconfig_to_dict(train_dataloader_cfg)
        self.val_cfg = dictconfig_to_dict(val_dataloader_cfg)

        self.cache_id = get_datamodule_hash(
            self.data_files,
            tokenizer.name_or_path,
            self.prompt_template,
            self.logic_version,
        )
        self.cache_dir = os.path.join(cache_dir, self.cache_id)

        self.train_collator = hydra.utils.instantiate(
            train_collator_cfg,
            tokenizer=self.tokenizer,
        )
        self.val_collator = hydra.utils.instantiate(
            val_collator_cfg, tokenizer=self.tokenizer
        )

    def _tokenize_train(self, example: dict[str, str]) -> dict:
        word, target = example["input"].lstrip(), example["output"].strip()

        prompt_text = self.prompt_template.format(word=word)
        prompt_part = f"{self.tokenizer.bos_token}{prompt_text}"

        tokenized_prompt = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            **self.tokenizer_prompt_kwargs,
        )

        full_text = f"{prompt_part}{target}{self.tokenizer.eos_token}"

        tokenized_full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            **self.tokenizer_full_text_kwargs,
        )

        input_ids = tokenized_full["input_ids"]
        prompt_len = len(tokenized_prompt["input_ids"])  # type: ignore
        labels = [-100] * prompt_len + input_ids[prompt_len:]  # type: ignore

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized_full["attention_mask"],
        }

    def _tokenize_validation(self, example: dict[str, str]) -> dict:
        word, target = example["input"].lstrip(), example["output"].strip()

        prompt_text = self.prompt_template.format(word=word)
        prompt_part = f"{self.tokenizer.bos_token}{prompt_text}"

        tokenized_prompt = self.tokenizer(
            prompt_part,
            add_special_tokens=False,
            **self.tokenizer_prompt_kwargs,
        )

        return {
            **tokenized_prompt,
            "target": target,
            "word": word,
        }

    def prepare_data(self) -> None:
        if os.path.exists(self.cache_dir):
            return

        raw_dataset = load_dataset("json", data_files=self.data_files)

        train_dataset = raw_dataset["train"].map(
            self._tokenize_train,
            remove_columns=raw_dataset["train"].column_names,
            desc="Tokenizing train dataset",
        )
        val_dataset = raw_dataset["val"].map(
            self._tokenize_validation,
            remove_columns=raw_dataset["val"].column_names,
            desc="Tokenizing validation dataset",
        )

        raw_dataset["train"] = train_dataset
        raw_dataset["val"] = val_dataset

        tokenized_dataset = DatasetDict(
            {
                "train": train_dataset,
                "val": val_dataset,
            }
        )

        tokenized_dataset.save_to_disk(self.cache_dir)

    def setup(self, stage: str | None = None) -> None:
        tokenized_dataset = load_from_disk(self.cache_dir)

        if stage == "fit" or stage == "test":
            self.train_ds = tokenized_dataset["train"]
            self.val_ds = tokenized_dataset["val"]

    def train_dataloader(self) -> DataLoader:
        train_dl = DataLoader(
            self.train_ds,  # type: ignore
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                collator=self.train_collator,
                padding_side="right",
            ),
            **self.train_cfg,
        )

        return train_dl

    def val_dataloader(self) -> list[DataLoader]:
        loss_dl = DataLoader(
            self.train_ds,  # type:ignore
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                collator=self.train_collator,
                padding_side="right",
            ),
            **self.val_cfg,
        )
        gen_dl = DataLoader(
            self.val_ds,  # type: ignore
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                collator=self.val_collator,
                padding_side="left",
            ),
            **self.val_cfg,
        )

        return [loss_dl, gen_dl]

    def test_dataloader(self) -> DataLoader:
        test_dl = DataLoader(
            self.val_ds,  # type: ignore
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                collator=self.val_collator,
                padding_side="left",
            ),
            **self.val_cfg,
        )

        return test_dl


def collate_fn(batch, tokenizer, collator, padding_side):
    tokenizer.padding_side = padding_side
    return collator(batch)
