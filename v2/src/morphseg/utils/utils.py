import hashlib
import json
import torch
import os


def get_device(device_type: str) -> torch.device:
    if "cuda" in device_type and not torch.cuda.is_available():
        raise RuntimeError(
            "Cuda is not available. Found no NVIDIA driver on your system."
        )

    return torch.device(device_type)


def get_datamodule_hash(data_files: dict[str, str], tokenizer_name: str, prompt_template: str) -> str:
    fingerprint = {
        "tokenizer": tokenizer_name,
        "template": prompt_template,
        "files": {}
    }
    
    for key, path in data_files.items():
        if path and os.path.exists(path):
            fingerprint["files"][key] = os.path.getmtime(path)
    
    fingerprint_str = json.dumps(fingerprint, sort_keys=True)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    