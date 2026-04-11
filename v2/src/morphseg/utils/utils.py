import torch


def get_device(device_type: str) -> torch.device:
    if "cuda" in device_type and not torch.cuda.is_available():
        raise RuntimeError(
            "Cuda is not available. Found no NVIDIA driver on your system."
        )

    return torch.device(device_type)
