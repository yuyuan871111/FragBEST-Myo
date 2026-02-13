from functools import partial

from torch.optim.lr_scheduler import ConstantLR, LambdaLR

from ..ignite.utils import lambda_lr_scheduler


def set_scheduler(optimizer, config):
    if config.lr_scheduler:
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                partial(
                    lambda_lr_scheduler,
                    lr0=config.lr,
                    n=config.max_epochs * config.le,
                    a=0.9,
                )
            ],
        )
    else:
        lr_scheduler = ConstantLR(optimizer, factor=1.0)
    return lr_scheduler
