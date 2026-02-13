"""Modified by Yu-Yuab (Stuart) Yang - 2024
from ignite generator template - 2024

"""

import os
from pprint import pformat
from typing import Any, cast

import ignite.distributed as idist
from ignite.engine import Events
from ignite.handlers import LRScheduler
from ignite.metrics import Accuracy, ConfusionMatrix, Loss, mIoU
from omegaconf.listconfig import ListConfig
from torch import optim
from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler

from ..seed import set_seed
from .ignite.data import (
    dataset_sample_loader,
    output_transform_mask,
    output_transform_masked,
    output_transform_normal,
    setup_data,
)
from .ignite.losses import setup_loss
from .ignite.models import setup_model
from .ignite.scheduler_opmizers import set_scheduler
from .ignite.trainers import setup_evaluator, setup_trainer
from .ignite.utils import (
    log_metrics,
    resume_from,
    save_config,
    setup_exp_logging,
    setup_handlers,
    setup_logging,
    setup_output_dir,
)


def train_model(dataloader_train, dataloader_val, rank, config, logger):
    # set the length of training data for the LamdaLR scheduler
    config.le = len(dataloader_train)  # how many training data batches

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config))
    optimizer = idist.auto_optim(optim.Adam(model.parameters(), lr=config.lr))
    criterion = setup_loss(config).to(device=device)
    lr_scheduler = set_scheduler(optimizer, config)

    # load from checkpoint
    if (config.checkpoint is not None) and (isinstance(config.checkpoint, ListConfig)):
        to_load = {"model": model}
        logger.info("Loading checkpoints for each kfold sets.")
        resume_from(
            to_load=to_load,
            checkpoint_fp=config.checkpoint[config.fold_num - 1],
            logger=logger,
        )
    elif config.checkpoint is not None:
        to_load = {"model": model}
        logger.info("Loading checkpoint from same checkpoint for all kfold sets.")
        resume_from(to_load=to_load, checkpoint_fp=config.checkpoint, logger=logger)
    else:
        logger.info("No checkpoint found. Training from scratch.")

    # setup metrics
    cm = ConfusionMatrix(
        num_classes=config.num_classes, output_transform=output_transform_mask
    )
    val_metrics = {
        "accuracy": Accuracy(output_transform=output_transform_mask),
        "loss": Loss(
            criterion,
            output_transform=(
                output_transform_masked
                if "masked" in config.loss_fn
                else output_transform_normal
            ),
        ),
        "miou": mIoU(cm),
    }

    # setup ignite trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, criterion, device, dataloader_train.sampler
    )
    train_evaluator = setup_evaluator(config, model, val_metrics, device)
    val_evaluator = setup_evaluator(config, model, val_metrics, device)

    # Setup trainer and evaluator loggers
    trainer.logger = setup_logging(
        config,
        custom_name=(
            "Trainer" if config.fold_num == 0 else f"[Kfold{config.fold_num}] Trainer"
        ),
    )
    train_evaluator.logger = val_evaluator.logger = setup_logging(
        config,
        custom_name=(
            "Evaluator"
            if config.fold_num == 0
            else f"[Kfold{config.fold_num}] Evaluator"
        ),
    )

    # setup ignite learning rate scheduler
    if isinstance(lr_scheduler, PyTorchLRScheduler):
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            lambda engine: cast(PyTorchLRScheduler, lr_scheduler).step(),
        )
    elif isinstance(lr_scheduler, LRScheduler):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # setup ignite checkpoint handlers
    to_save_train = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
        "lr_scheduler": lr_scheduler,
    }
    to_save_eval = {"model": model}
    ckpt_handler_train, ckpt_handler_eval = setup_handlers(
        trainer, val_evaluator, config, to_save_train, to_save_eval
    )

    # setup ignite experiment tracking
    if rank == 0:
        evaluators = {"training": train_evaluator, "validation": val_evaluator}
        exp_logger = setup_exp_logging(config, trainer, optimizer, evaluators)

    # print metrics to the stderr with `add_event_handler` API
    # for training stats
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    # run evaluation at every training epoch end
    # with shortcut `on` decorator API and metrics to the stderr
    # again with `add_event_handler` API
    # for evaluation stats
    @trainer.on(Events.EPOCH_COMPLETED)
    def _():
        train_evaluator.run(dataloader_train, epoch_length=config.eval_epoch_length)
        log_metrics(train_evaluator, "train")
        val_evaluator.run(dataloader_val, epoch_length=config.eval_epoch_length)
        log_metrics(val_evaluator, "val")

    # let's try run evaluation first as a sanity check
    # @trainer.on(Events.STARTED)
    # def _():
    #     evaluator.run(dataloader_val, epoch_length=config.eval_epoch_length)

    # setup if done. let's run the training
    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
        epoch_length=config.train_epoch_length,
    )

    # close logger
    if rank == 0:
        exp_logger.close()

    # show last checkpoint names
    logger.info(f"Last training checkpoint name - {ckpt_handler_train.last_checkpoint}")

    logger.info(
        f"Last evaluation checkpoint name - {ckpt_handler_eval.last_checkpoint}"
    )


def run_training(local_rank: int, config: Any):
    # make a certain seed
    rank = idist.get_rank()
    set_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    check_config_before_training(config)
    if rank == 0:
        save_config(config, output_dir)
    config.output_dir = output_dir

    # setup ignite logger with python logging
    # print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))

    if config.cross_val:
        whole_dataset, indices, splits = setup_data(config)
        for fold_idx, (train_idx, val_idx) in enumerate(splits.split(indices)):
            # cross-validation fold number starts from 1
            config.fold_num = fold_idx + 1
            logger.info(
                f"Fold {config.fold_num}, size: [train] {len(train_idx)}/"
                f"[val] {len(val_idx)}"
            )
            logger.debug(
                f"Fold {config.fold_num}, train indices: {train_idx}, "
                f"val indices: {val_idx}"
            )

            # create dataloaders
            dataloader_train, dataloader_val = dataset_sample_loader(
                whole_dataset, train_idx, val_idx, config
            )

            # run training
            train_model(dataloader_train, dataloader_val, rank, config, logger)

    else:
        # use without the cross-validation
        config.fold_num = 0

        # create dataloaders
        dataloader_train, dataloader_val = setup_data(config)

        # run training
        train_model(dataloader_train, dataloader_val, rank, config, logger)


def check_config_before_training(config: Any):
    # check if the config has all necessary fields
    if config.cross_val:
        # cross-validation
        assert hasattr(config, "fold"), "config.fold is required for cross-validation"
        config.data_split_ratio = None

        # check if checkpoint is provided
        if config.checkpoint is not None:
            assert isinstance(config.checkpoint, ListConfig) or isinstance(
                config.checkpoint, str
            ), "config.checkpoint should be a list or str for cross-validation"
            if isinstance(config.checkpoint, ListConfig):
                assert len(config.checkpoint) == config.fold, (
                    "config.checkpoint should have the same length as config.fold"
                )
                assert all(os.path.exists(c) for c in config.checkpoint), (
                    "config.checkpoint should be a list of valid file paths"
                )
            else:
                assert os.path.exists(config.checkpoint), (
                    "config.checkpoint should be a valid file path"
                )
        else:
            pass
    else:
        # without cross-validation
        assert hasattr(config, "data_split_ratio"), (
            "config.data_split_ratio is required"
        )
        config.fold = 0

    if not hasattr(config, "transform"):
        # config.transform is required
        config.transform = None
    return config
