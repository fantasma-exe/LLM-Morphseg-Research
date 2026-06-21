import hashlib
import json
import torch
import os

import typing as tp

from pathlib import Path


def get_device(device_type: str) -> torch.device:
    if "cuda" == device_type and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available on your system.")

    return torch.device(device_type)


def get_datamodule_hash(
    data_files: dict[str, str],
    tokenizer_name: str,
    prompt_template: str,
    logic_version: str,
) -> str:
    fingerprint = {
        "tokenizer": tokenizer_name,
        "template": prompt_template,
        "files": {},
        "logic_versoion": logic_version,
    }

    for key, path in data_files.items():
        if path and os.path.exists(path):
            fingerprint["files"][key] = os.path.getmtime(path)

    fingerprint_str = json.dumps(fingerprint, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]


def save_run_info(run_info: dict[str, tp.Any], save_path: str) -> None:
    file_name = Path(save_path) / "run_info.json"
    with open(file_name, "w") as f:
        json.dump(run_info, f, indent=2)


def load_checkpoint_path(load_path: str, ckpt_key: str = "last_ckpt") -> str:
    file_name = Path(load_path) / "run_info.json"
    with open(file_name, "r") as f:
        info = json.load(f)

    ckpt_path = Path(info["train"][ckpt_key])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return str(ckpt_path)


def load_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)
