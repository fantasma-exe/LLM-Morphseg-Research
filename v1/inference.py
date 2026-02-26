import json
import torch
import shutil

from typing import Any, Callable
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import PeftModel
from datasets import load_dataset, Dataset

from prompts import build_prompt
from metrics_registry import METRICS
from config import load_config
from naming import normalize_model_name


def _setup_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    """
    Create directory structure for inference outputs.

    The following directory structure is created if it does not exist::

        outputs/
            <normalized_model_name>/
                runs/
                    <run_id>/
                        inference/
                            predictions.jsonl
                            metrics.json

    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded inference configuration. Must contain:
        - paths.run_id
        - paths.output_dir
        - model.name

    Returns
    -------
    tuple[Path, Path, Path]
        Tuple containing:
        - predictions_path
        - metrics_path
        - inference_dir

    Raises
    ------
    FileNotFoundError
        If the specified run_id folder does not exist.
    ValueError
        If user did not specify run_id in confing file.
    """
    try:
        run_id = cfg["paths"]["run_id"]
    except KeyError:
        raise ValueError("Specify 'run_id' of a trained model in cfg['paths']")

    base_output_dir = Path(cfg["paths"]["output_dir"])
    model_dir = base_output_dir / normalize_model_name(cfg["model"]["name"])
    run_dir = model_dir / "runs" / run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    infer_dir = run_dir / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = infer_dir / "predictions.jsonl"
    metrics_path = infer_dir / "metrics.json"

    return predictions_path, metrics_path, infer_dir


def _get_train_path(cfg: dict[str, Any]) -> Path:
    """
    Parameters
    ----------
    cfg : dict[str, Any]
        Loaded inference configuration.

    Returns
    -------
    Path
        Path where stored training data.

    Raises
    ------
    ValueError
        If user did not specify run_id in confing file.
    """
    try:
        run_id = cfg["paths"]["run_id"]
    except KeyError:
        raise ValueError("Specify 'run_id' of a trained model in cfg['paths']")
    base_output_dir = Path(cfg["paths"]["output_dir"])
    model_dir = normalize_model_name(cfg["model"]["name"])
    return base_output_dir / model_dir / "runs" / run_id / "train"


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
    train_path = _get_train_path(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        train_path / cfg["model"]["lora_path"],
    )
    model.eval()

    return model, tokenizer


def _generate_single_response(model: Any, tokenizer: PreTrainedTokenizer, prompt: str, gen_cfg: dict[str, Any]) -> str:
    """
    Generate a single model response for a given prompt.

    The function uses generation parameters from the configuration
    and returns only the decoded answer part of the model output.

    Parameters
    ----------
    model : Any
        LoRA-adapted language model.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    prompt : str
        Input prompt string.
    gen_cfg : dict[str, Any]
        Generation configuration.

    Returns
    -------
    str
        Generated model answer.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg["do_sample"],
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Ответ:")[-1].strip()


def _run_infer_loop(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    cfg: dict[str, Any],
    metric_fns: dict[str, Callable],
    metrics_path: Path,
    predictions_path: Path,
) -> None:
    """
    Run inference on the entire test dataset and compute evaluation metrics.

    The function generates predictions for all examples, aggregates
    evaluation metrics over the dataset, and saves both the metrics
    and per-example predictions to disk.

    Parameters
    ----------
    model : Any
        LoRA-adapted language model.
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the model.
    dataset : Dataset
        Test dataset.
    cfg : dict[str, Any]
        Loaded inference configuration.
    metric_fns : dict[str, Callable]
        Dictionary mapping metric names to metric functions.
    metrics_path : Path
        Path to save aggregated metrics.
    predictions_path : Path
        Path to save per-example predictions.

    Returns
    -------
    None
    """
    predictions = []
    preds = []
    golds = []

    gen_cfg = cfg["generation"]

    for ex in dataset:
        word = ex["input"]
        gold = ex["output"]

        prompt = build_prompt(word, None)
        pred = _generate_single_response(model, tokenizer, prompt, gen_cfg)

        preds.append(pred)
        golds.append(gold)

        predictions.append(
            {
                "word": word,
                "gold": gold,
                "pred": pred,
            }
        )

    final_metrics = {}
    for name, metric_fn in metric_fns.items():
        final_metrics[name] = metric_fn(preds, golds)

    final_metrics.update(
        {
            "num_examples": len(golds),
            "model": cfg["model"]["name"],
            "lora_path": cfg["model"]["lora_path"],
        }
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in predictions:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    """
    Entry point for inference pipeline.
    """
    cfg = load_config("configs/infer.yaml")

    predictions_path, metrics_path, infer_dir = _setup_dirs(cfg)
    model, tokenizer = _init_model_and_tokenizer(cfg)

    metric_fns = {name: METRICS[name] for name in cfg["metrics"]}
    dataset = load_dataset("json", data_files={"test": cfg["paths"]["test_data"]})["test"]

    _run_infer_loop(model, tokenizer, dataset, cfg, metric_fns, metrics_path, predictions_path)

    shutil.copy("configs/infer.yaml", infer_dir / "infer.yaml")


if __name__ == "__main__":
    main()
