import hydra
import torch

from omegaconf import DictConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: DictConfig,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
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
