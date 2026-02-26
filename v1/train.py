import torch
import shutil

from datetime import datetime
from typing import Any
from pathlib import Path
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

from prompts import build_prompt
from config import load_config
from naming import normalize_model_name

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    """
    Create output directory structure for a given model.

    The following directory structure is created if it does not exist::

        outputs/
            <normalized_model_name>/
                runs/
                    <run_id>/
                        train/
                            checkpoints/
                            final_model/

    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded training configuration.

    Returns
    -------
    tuple[Path, Path, Path]
        Tuple containing:
        - run_dir
        - checkpoints_dir
        - final_model_dir
    """
    model_name = cfg["model"]["name"]
    base_output_dir = Path(cfg["paths"]["output_dir"])

    model_dir = base_output_dir / normalize_model_name(model_name)
    run_dir = model_dir / "runs" / run_id
    checkpoints_dir = run_dir / "train" / "checkpoints"
    final_model_dir = run_dir / "train" / "final_model"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, checkpoints_dir, final_model_dir


def _init_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, PreTrainedTokenizer]:
    """
    Initialize tokenizer and LoRA-adapted language model.

    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded training configuration.

    Returns
    -------
    tuple[Any, PreTrainedTokenizer]
        Tuple containing the initialized model and tokenizer.
    """
    model_name = cfg["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, quantization_config=bnb_config, dtype=torch.bfloat16
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def _prepare_dataset(cfg: dict[str, Any]) -> Dataset:
    """
    Load and preprocess the training dataset.

    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded training configuration.

    Returns
    -------
    DatasetDict
        HuggingFace DatasetDict containing the processed training split.
    """
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["paths"]["train_data"],
        },
    )

    def format_example(ex: dict[str, str]) -> dict[str, str]:
        return {"text": build_prompt(ex["input"], ex["output"])}

    dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
    )

    return dataset


def _run_train(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    cfg: dict[str, Any],
    checkpoints_dir: Path,
    final_model_dir: Path,
) -> None:
    """
    Run supervised fine-tuning (SFT) training loop.

    Parameters
    ----------
    model : Any
        LoRA-wrapped language model.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    dataset : Dataset
        Prepared training dataset.
    cfg : dict[str, Any]
        Loaded training configuration.
    checkpoints_dir : Path
        Directory for saving intermediate checkpoints.
    final_model_dir : Path
        Directory for saving the final trained model.

    Returns
    -------
    None
    """
    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["grad_accum"],
        learning_rate=cfg["training"]["lr"],
        num_train_epochs=cfg["training"]["epochs"],
        bf16=cfg["training"]["bf16"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        args=args,
    )
    resume_ckpt = cfg["training"].get("resume_checkpoint")

    if resume_ckpt is not None:
        resume_ckpt = str(checkpoints_dir / resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))


def main() -> None:
    """
    Entry point for training script.
    """
    cfg = load_config("configs/train.yaml")

    run_dir, checkpoints_dir, final_model_dir = _setup_dirs(cfg)
    model, tokenizer = _init_model_and_tokenizer(cfg)
    dataset = _prepare_dataset(cfg)
    _run_train(model, tokenizer, dataset, cfg, checkpoints_dir, final_model_dir)

    shutil.copy(
        "configs/train.yaml",
        run_dir / "train.yaml",
    )


if __name__ == "__main__":
    main()
