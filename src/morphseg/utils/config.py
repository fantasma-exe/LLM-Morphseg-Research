import typing as tp

from omegaconf import DictConfig, OmegaConf


def dictconfig_to_dict(cfg: DictConfig, resolve: bool = True) -> dict[str, tp.Any]:
    return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=True)  # type: ignore
