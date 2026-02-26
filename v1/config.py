from typing import Any
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load training configuration from a YAML file.

    The configuration is expected to define model, LoRA, training
    parameters, and filesystem paths.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed training configuration.
    """
    with open(config_path) as f:
        return yaml.load(f)
