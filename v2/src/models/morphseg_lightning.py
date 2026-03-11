import torch
import hydra

import typing as tp
import pytorch_lightning as L

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer
from omegaconf import DictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils.metrics import (
    morpheme_precision,
    morpheme_f1,
    morpheme_recall,
    char_accuracy,
    word_accuracy,
)


class MorphSegModule(L.LightningModule):
    """
    PyTorch Lightning module for training a causal language model with optional
    4-bit quantization and LoRA parameter-efficient fine-tuning.

    The module loads a pretrained transformer model from Hugging Face,
    optionally applies 4-bit quantization via BitsAndBytes, prepares the model
    for k-bit training, and injects LoRA adapters using PEFT. Optimizer and
    scheduler are instantiated via Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration describing the model setup. The expected structure
        and available options are defined in
        ``configs/model/morphseg_lightning.yaml``.

    optimizer : DictConfig
        Hydra configuration used to instantiate the optimizer.

    scheduler : DictConfig
        Hydra configuration used to instantiate the learning rate scheduler.

    scheduler_config : DictConfig
        Additional Lightning scheduler configuration (e.g. interval, frequency,
        monitor).

    Attributes
    ----------
    model : torch.nn.Module
        The underlying Hugging Face causal language model with LoRA adapters applied.

    cfg : DictConfig
        Model configuration.

    optimizer_cfg : DictConfig
        Hydra configuration used to instantiate the optimizer.

    scheduler_cfg : DictConfig
        Hydra configuration used to instantiate the scheduler.

    scheduler_config : DictConfig
        Additional scheduler configuration passed to Lightning.
    """

    def __init__(
        self,
        cfg: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        scheduler_config: DictConfig,
        lora: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.cfg = cfg
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.scheduler_config = scheduler_config

        bnb_cfg = None
        if cfg.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.name,
            quantization_config=bnb_cfg,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        )

        if cfg.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(**lora)  # type: ignore

        self.model = get_peft_model(self.model, lora_cfg)
        self.tokenizer = tokenizer
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None) -> tp.Any:
        """
        Forward pass through the language model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape `(batch_size, sequence_length)`.

        attention_mask : torch.Tensor
            Attention mask indicating valid tokens.

        labels : torch.Tensor, optional
            Target token IDs for language modeling. If provided,
            the model will compute and return the loss.

        Returns
        -------
        transformers.modeling_outputs.CausalLMOutput
            Model output containing logits and optionally the loss.
        """

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx) -> tp.Any:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : dict
            Batch containing:

            - input_ids : torch.Tensor
            - attention_mask : torch.Tensor
            - labels : torch.Tensor

        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed training loss.
        """

        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : dict
            Batch containing:

            - input_ids : torch.Tensor
            - attention_mask : torch.Tensor
            - labels : torch.Tensor

        batch_idx : int
            Index of the current batch.

        Returns
        -------
        None
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention=batch["attention_mash"],
            max_new_token=64,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        preds_raw = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  # type: ignore
        golds_raw = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        def extract_answer(text: str) -> str:
            if "### Ответ:" in text:
                return text.split("### Ответ:")[1].strip()
            return text.strip()

        preds = [extract_answer(p) for p in preds_raw]
        golds = [extract_answer(g) for g in golds_raw]

        self.validation_step_outputs.append({"preds": preds, "golds": golds})

    def on_validation_epoch_end(self) -> None:
        all_preds = [
            p for batch in self.validation_step_outputs for p in batch["preds"]
        ]
        all_golds = [
            g for batch in self.validation_step_outputs for g in batch["golds"]
        ]

        metrics = {
            "morpheme_precision_full": morpheme_precision(all_preds, all_golds),
            "morpheme_recall_full": morpheme_recall(all_preds, all_golds),
            "morpheme_f1_full": morpheme_f1(all_preds, all_golds),
            "morpheme_precision_root": morpheme_precision(
                all_preds, all_golds, allowed_types={"ROOT"}
            ),
            "morpheme_recall_root": morpheme_recall(
                all_preds, all_golds, allowed_types={"ROOT"}
            ),
            "morpheme_f1_root": morpheme_f1(
                all_preds, all_golds, allowed_types={"ROOT"}
            ),
            "char_level_accuracy": char_accuracy(all_preds, all_golds),
            "word_accuracy": word_accuracy(all_preds, all_golds),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> tp.Any:
        """
        Instantiate optimizer and scheduler using Hydra configuration.

        Returns
        -------
        dict
            Dictionary compatible with PyTorch Lightning optimizer configuration.
            Contains instantiated optimizer and learning rate scheduler.
        """

        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=self.parameters(),
        )

        scheduler = hydra.utils.instantiate(
            self.scheduler_cfg,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **self.scheduler_config,
            },
        }
