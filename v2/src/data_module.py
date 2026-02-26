import pytorch_lightning as L

from omegaconf import DictConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    BatchEncoding,
)


class DataMoudle(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self._cfg = cfg
        self._tokenizer = tokenizer

        self._data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=False
        )

    def _build_prompt(self, word: str, answer: str | None = None) -> str:
        """
        Build a prompt for morphological segmentation.

        The prompt instructs the model to split the given word into morphemes
        and specify the type of each morpheme. If an answer is provided, it is
        appended to the prompt.

        Parameters
        ----------
        word : str
            Input word to be segmented.
        answer : str or None, optional
            Reference answer to append to the prompt.

        Returns
        -------
        str
            Formatted prompt string.
        """

        prompt = f"""
            ### Инструкция:
            Раздели слово на морфемы и укажи тип каждой морфемы

            ### Слово:
            {word}

            ### Ответ:\n
        """

        if answer is not None:
            prompt += answer

        return prompt

    def setup(self, stage: str | None = None) -> None:
        """
        Create train dataset: load dataset, then tokenize it.

        Parameters
        ----------
        stage : str | None
            Is not used.

        Returns
        -------
        None
        """
        dataset = load_dataset("json", data_files=self._cfg.paths.train_path)

        def tokenize_function(example: dict[str, str]) -> BatchEncoding:
            """
            Create prompt and tokenize it.

            Parameters
            ----------
            example: dict[str, str]
                raw json:
                    {
                    "input": "....",
                    "output": "..."
                    }

            Returns
            -------
            BatchEncoding
                dict[str, list[int]] wrapper.
            """
            prompt = self._build_prompt(example["input"], example["output"])

            outputs = self._tokenizer(
                prompt,
                truncation=True,
                max_length=self._cfg.training.max_seq_length,
                padding=False,
                add_special_tokens=True,
            )

            outputs["labels"] = outputs["input_ids"].copy()  # type: ignore

            return outputs

        self._train_dataset = dataset["train"].map(
            tokenize_function,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create train dataloader before training.

        Returns
        -------
        None
        """
        return DataLoader(
            self._train_dataset,  # type: ignore
            batch_size=self._cfg.training.batch_size,
            shuffle=True,
            collate_fn=self._data_collator,
            pin_memory=True,
        )
