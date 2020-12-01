import math
from typing import List
import torch


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        delay_iters: int = 0,
        eta_min_lr: int = 0,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.delay_iters = delay_iters
        self.eta_min_lr = eta_min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        assert self.delay_iters >= self.warmup_iters, "Scheduler delay iters must be larger than warmup iters"
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_iters:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor,
            )
            return [
                base_lr * warmup_factor for base_lr in self.base_lrs
            ]
        elif self.last_epoch <= self.delay_iters:
            return self.base_lrs

        else:
            return [
                self.eta_min_lr + (base_lr - self.eta_min_lr) *
                (1 + math.cos(
                    math.pi * (self.last_epoch - self.delay_iters) / (self.max_iters - self.delay_iters))) / 2
                for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))