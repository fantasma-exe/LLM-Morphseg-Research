import hydra
import torch
import gc
import copy

import pytorch_lightning as L
import typing as tp

from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import Optimizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer

from morphseg.utils import (
    MorphemeMetrics,
    dictconfig_to_dict,
)


class MorphSegModule(L.LightningModule):
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
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model_cfg = copy.deepcopy(model_cfg)
        self.log_cfg = log_cfg
        self.lora_cfg = dictconfig_to_dict(lora_cfg)
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

        lora = LoraConfig(**self.lora_cfg)
        self.model = get_peft_model(self.model, lora)

        if bnb_cfg is None and self.model_cfg.get("use_grad_checkpointing", False):
            self.model.gradient_checkpointing_enable()  # type: ignore
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()  # type: ignore

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=False)
            print(
                f"Missing params: {len(missing)}, Unexpected params: {len(unexpected)}",
                f"Keys: {unexpected[: min(50, len(unexpected))]}",
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None) -> tp.Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def on_load_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        sd = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            k: v
            for k, v in sd.items()
            if not (
                k.endswith(".absmax")
                or k.endswith(".quant_map")
                or ".quant_state." in k
                or k.endswith(".nested_absmax")
                or k.endswith(".nested_quant_map")
            )
        }

    def training_step(self, batch, batch_idx) -> tp.Any:
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
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].size(0),
        )

        self._log_memory("train")

        if self.training_step_cnt % self.model_cfg.clean_memory_every_nsteps == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx) -> None:
        if dataloader_idx == 0:
            outputs = self(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )

            val_loss = outputs.loss

            self.log(
                "val_loss",
                val_loss.item(),
                prog_bar=True,
                on_epoch=True,
                batch_size=batch["input_ids"].size(0),
                add_dataloader_idx=False,
            )

            self._log_memory("val", postfix="_loss")

        elif (self.current_epoch + 1) % self.log_cfg.val_every_nepochs == 0 or (
            self.current_epoch + 1
        ) == self.log_cfg.max_epochs:
            gen_data = self._generate(batch)

            preds = self.tokenizer.batch_decode(gen_data, skip_special_tokens=True)
            clean_preds = [p.split("\n")[0].strip() for p in preds]

            self.validation_step_outputs.append(
                {
                    "preds": clean_preds,
                    "targets": batch["target"],
                    "words": batch["word"],
                }
            )

            self._log_memory("val", postfix="_generation")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self) -> tp.Any:
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

    def _log_memory(self, mode: tp.Literal["train", "val"], postfix: str = "") -> None:
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        self.log(
            f"debug/{mode}/vram_allocated{postfix}",
            allocated,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
        )
        self.log(
            f"debug/{mode}/vram_reserved{postfix}",
            reserved,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        trainable_grad_norm = [
            p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None
        ]

        if trainable_grad_norm:
            total_norm = torch.norm(torch.stack(trainable_grad_norm), 2)

            self.log("grad_norm_l2", total_norm.item(), on_step=True, on_epoch=False)

    def _generate(self, batch):
        input_ids = batch["input_ids"]

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.model_cfg.max_tokens_val_generation,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )

        prompt_len = input_ids.shape[1]
        gen_only = generated_ids[:, prompt_len:]

        return gen_only

    def test_step(self, batch, batch_idx) -> None:
        gen_data = self._generate(batch)

        preds = self.tokenizer.batch_decode(gen_data, skip_special_tokens=True)

        clean_preds = [p.split("\n")[0].split(" ")[0].strip() for p in preds]

        self.test_step_outputs.append(
            {
                "preds": clean_preds,
                "targets": batch["target"],
                "words": batch["word"],
            }
        )

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return

        all_preds = [p for batch in self.test_step_outputs for p in batch["preds"]]
        all_targets = [t for batch in self.test_step_outputs for t in batch["targets"]]
        all_words = [w for batch in self.test_step_outputs for w in batch["words"]]

        metric_calculator = MorphemeMetrics()
        metrics = metric_calculator.compute(all_preds, all_targets, all_words)

        metrics = {
            f"test_{metric_name}": value for metric_name, value in metrics.items()
        }

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.test_step_outputs.clear()
