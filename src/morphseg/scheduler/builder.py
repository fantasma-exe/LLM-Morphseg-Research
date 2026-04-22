import hydra
import torch

from omegaconf import DictConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: DictConfig,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Instantiate a learning rate scheduler with support for complex sequential strategies.

    This builder handles both simple schedulers and multi-stage pipelines (like
    SequentialLR). If a sequential scheduler is detected, it automatically
    calculates and injects duration parameters (warmup steps, T_max, and
    milestones) based on the provided total steps and warmup ratio.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.

    schedulers_cfg : omegaconf.DictConfig
        Hydra configuration object.

    total_steps : int
        The total number of training steps (iterations).

    warmup_ratio : float, default=0.1
        The fraction of total steps to be used for the linear warmup phase.
        Must be between in [0, 1].

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        The instantiated PyTorch learning rate scheduler.

    Notes
    -----
    When using `SequentialLR`, the function assumes the first scheduler in the
    list is the Warmup phase (`LinearLR`) and the second one is the main
    decay phase (e.g., `CosineAnnealingLR`).
    """
    if not (0 <= warmup_ratio <= 1):
        raise ValueError("Ratio must be in [0, 1] range.")

    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_cfg._target_ == "torch.optim.lr_scheduler.SequentialLR":
        scheduler_cfg.schedulers[0].total_iters = warmup_steps
        scheduler_cfg.schedulers[1].T_max = total_steps - warmup_steps
        scheduler_cfg.milestones = [warmup_steps]

        scheduler = hydra.utils.call(scheduler_cfg)
        wrapped_scheds = [
            hydra.utils.call(s)(optimizer) for s in scheduler_cfg.schedulers
        ]

        return scheduler(
            optimizer, schedulers=wrapped_scheds, milestones=scheduler_cfg.milestones
        )

    return hydra.utils.instantiate(scheduler_cfg)
