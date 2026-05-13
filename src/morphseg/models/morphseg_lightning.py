import hydra
import torch
import gc
import copy

import pytorch_lightning as L
import typing as tp

from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer

from morphseg.utils import (
    MorphemeMetrics,
    dictconfig_to_dict,
)


class MorphSegModule(L.LightningModule):
    """
    PyTorch Lightning module for training a causal language model with optional
    quantization and LoRA parameter-efficient fine-tuning.

    The module loads a pretrained transformer model from Hugging Face, optionally
    applies quantization using BitsAndBytes, prepares the model for k-bit training
    when required, and injects LoRA adapters using PEFT. Optimizer and scheduler
    are instantiated from Hydra configuration during training.

    Parameters
    ----------
    model_cfg : DictConfig
        Hydra configuration describing how the base Hugging Face model should be
        loaded (e.g. model name, dtype, trust_remote_code flag).

    log_cfg : DictConfig
        Hydra configuration that controls when and what to log.

    quantization_cfg : DictConfig | None
        Optional configuration used to construct a ``BitsAndBytesConfig`` that
        controls model quantization. If enabled, this configuration is passed to
        ``AutoModelForCausalLM.from_pretrained``. If disabled or ``None``, the
        model is loaded without quantization.

    optimizer_cfg : DictConfig
        Hydra configuration used to instantiate the optimizer.

    scheduler_cfg : DictConfig
        Hydra configuration used to instantiate the learning rate scheduler.

    scheduler_settings : DictConfig
        Additional Lightning scheduler configuration (e.g. interval, frequency,
        monitor).

    lora_cfg : DictConfig
        Configuration for PEFT LoRA adapters (rank, alpha, dropout, target modules,
        etc.).

    tokenizer : PreTrainedTokenizer
        Hugging Face tokenizer used for preprocessing and generation.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        log_cfg: DictConfig,
        lora_cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
        quantization_cfg: DictConfig | None = None,
        optimizer_cfg: DictConfig | None = None,
        scheduler_cfg: DictConfig | None = None,
        scheduler_settings: DictConfig | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model_cfg = copy.deepcopy(model_cfg)
        self.log_cfg = log_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.scheduler_settings = scheduler_settings
        self.training_step_cnt = 0

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        torch_dtype = dtype_map.get(self.model_cfg.torch_dtype, torch.float32)

        bnb_cfg = None
        if quantization_cfg is not None and quantization_cfg.get(
            "use_quantization", False
        ):
            quant_kwargs = dictconfig_to_dict(quantization_cfg)
            quant_kwargs.pop("enabled", None)

            if "bnb_4bit_compute_dtype" in quant_kwargs:
                quant_kwargs["bnb_4bit_compute_dtype"] = dtype_map[
                    quant_kwargs["bnb_4bit_compute_dtype"]
                ]

            bnb_cfg = BitsAndBytesConfig(**quant_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_cfg.model_name,
            quantization_config=bnb_cfg,
            trust_remote_code=self.model_cfg.trust_remote_code,
            dtype=torch_dtype,
        )

        if bnb_cfg is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.model_cfg.get(
                    "use_grad_checkpointing", False
                ),
            )

        lora = LoraConfig(**dictconfig_to_dict(lora_cfg))
        self.model = get_peft_model(self.model, lora)

        if bnb_cfg is None and self.model_cfg.get("use_grad_checkpointing", False):
            self.model.gradient_checkpointing_enable()  # type: ignore
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()  # type: ignore

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        typing.Any
            Model output containing logits and optionally the loss.
        """

        return self.model(
            input_ids=input_ids,
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

        self.training_step_cnt += 1

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        self.log(
            "train/loss",
            loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self._log_memory("train")

        if self.training_step_cnt % self.model_cfg.clean_memory_every_n_step == 0:
            gc.collect()
            torch.cuda.empty_cache()

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
        """

        with torch.inference_mode():
            outputs = self(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )

        val_loss = outputs.loss
        self.log("val/loss", val_loss.item(), prog_bar=True, on_epoch=True)

        if batch_idx < self.log_cfg.limit_val_batches:
            prompt_raw_texts = batch["prompt_raw_texts"]

            self.tokenizer.padding_side = "left"
            prompt_encodings = self.tokenizer(
                prompt_raw_texts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=True,
            ).to(self.model.device)  # type: ignore
            self.tokenizer.padding_side = "right"

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=prompt_encodings["input_ids"],
                    attention_mask=prompt_encodings["attention_mask"],
                    max_new_tokens=self.model_cfg.max_tokens_val_generation,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_only = generated_ids[:, prompt_encodings["input_ids"].shape[1] :]  # type: ignore
            preds = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)  # type: ignore

            clean_preds = [p.split("\n")[0].strip() for p in preds]
            clean_targets = [t.strip() for t in batch["targets"]]
            print(f"== CLEAN PREDS == {clean_preds}")
            self.validation_step_outputs.append(
                {
                    "preds": clean_preds,
                    "targets": clean_targets,
                    "words": batch["words"],
                }
            )

        self._log_memory("val")

    def on_validation_epoch_end(self) -> None:
        if not self.validation_step_outputs:
            return

        all_preds = [
            p for batch in self.validation_step_outputs for p in batch["preds"]
        ]
        all_targets = [
            t for batch in self.validation_step_outputs for t in batch["targets"]
        ]
        all_words = [
            w for batch in self.validation_step_outputs for w in batch["words"]
        ]

        print("\n" + "=" * 50)
        print(f"Epoch {self.current_epoch + 1} - Sample Generations")
        for i in range(min(self.log_cfg.num_print_sample, len(all_preds))):
            print(f"Target : {all_targets[i]}")
            print(f"Predict: {all_preds[i]}")
            print("-" * 50)

        metric_calculator = MorphemeMetrics()
        metrics = metric_calculator.compute(all_preds, all_targets, all_words)

        self.log_dict(metrics, prog_bar=True)
        self.validation_step_outputs.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_save_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        """
        Modify the checkpoint before saving so that it contains only LoRA weights.

        This hook filters the ``state_dict`` and keeps only parameters whose names
        contain the substring ``"lora"``. All other model weights are removed.
        As a result, the saved checkpoint stores only the LoRA adapter parameters,
        which significantly reduces checkpoint size and allows later loading
        on top of the base pretrained model.

        Parameters
        ----------
        checkpoint : dict[str, Any]
            The checkpoint dictionary created by PyTorch Lightning during saving.
            The ``state_dict`` entry is modified in-place to keep only LoRA weights.
        """

        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if "lora" in k
        }

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

        if self.trainer.max_steps != -1:
            total_steps = self.trainer.max_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches

        scheduler = hydra.utils.call(
            self.scheduler_cfg,
            optimizer=optimizer,
            total_steps=total_steps,
            _recursive_=False,
        )

        if self.scheduler_settings is None:
            ss = {}
        else:
            ss = self.scheduler_settings

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **ss,
            },
        }

    def _log_memory(self, mode: tp.Literal["train", "val"]) -> None:
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        self.log(
            f"debug/{mode}/vram_allocated", allocated, on_step=True, on_epoch=False
        )
        self.log(f"debug/{mode}/vram_reserved", reserved, on_step=True, on_epoch=False)
