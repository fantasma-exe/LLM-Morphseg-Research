import typing as tp

from omegaconf import DictConfig, OmegaConf


def dictconfig_to_dict(cfg: DictConfig, resolve: bool = True) -> dict[str, tp.Any]:
    """
    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration.

    resolve : bool, default=True
        Whether resolve keys.

    Returns
    -------
    dict[str, typing.Any]
    """

    return OmegaConf.to_container(cfg, resolve=resolve, throw_on_missing=True)  # type: ignore
