import math
from torch.optim.lr_scheduler import LambdaLR

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=None, lr_min_ratio=0.1):
    """
    Linear warmup from 0 -> lr_max over warmup_steps, then cosine decay to lr_min_ratio * lr_max.
    Call sched.step() **once per optimizer step**.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): The total number of steps to schedule.
        warmup_steps (int): The number of steps to warmup.
        lr_min_ratio (float): The minimum ratio of the learning rate to the maximum learning rate.

    Returns:
        LambdaLR: The scheduler.
    """
    if warmup_steps is None:
        warmup_steps = max(int(0.02 * total_steps), 2000)

    def lr_lambda(step):  # step: 0,1,2,... (after each sched.step())
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine   # 1 -> floor

    return LambdaLR(optimizer, lr_lambda)