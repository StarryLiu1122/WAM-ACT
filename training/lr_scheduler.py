"""
Learning Rate Scheduler
学习率调度器

"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    创建带warmup的余弦学习率调度器

    Args:
        optimizer: 优化器
        num_warmup_steps: warmup步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数
        last_epoch: 上次epoch

    Returns:
        scheduler: LambdaLR调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 线性warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # 余弦衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    创建带warmup的线性学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            ),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
