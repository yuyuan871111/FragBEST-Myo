"""Modified by Yu-Yuab (Stuart) Yang - 2024
from ignite generator template - 2024

"""

import logging
from argparse import ArgumentParser
from collections.abc import Mapping
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import TensorboardLogger, WandBLogger
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping
from ignite.utils import setup_logger
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer


def get_default_parser():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Config file path")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["nccl", "gloo"],
        type=str,
        help="DDP backend",
    )
    return parser


def setup_config(parser=None):
    if parser is None:
        parser = get_default_parser()

    args = parser.parse_args()
    config_path = args.config
    config = OmegaConf.load(config_path)
    config.backend = args.backend

    return config


def log_metrics(engine: Engine, tag: str) -> None:
    """Log `engine.state.metrics` with given `engine` and `tag`.

    Parameters
    ----------
    engine
        instance of `Engine` which metrics to log.
    tag
        a string to add at the start of output.

    """
    metrics_format = (
        f"{tag} [{engine.state.epoch}/{engine.state.iteration}]: {engine.state.metrics}"
    )
    engine.logger.info(metrics_format)


def resume_from(
    to_load: Mapping,
    checkpoint_fp: str | Path,
    logger: Logger,
    strict: bool = True,
    model_dir: str | None = None,
) -> None:
    """Loads state dict from a checkpoint file to resume the training.

    Parameters
    ----------
    to_load
        a dictionary with objects, e.g. {“model”: model, “optimizer”: optimizer, ...}
    checkpoint_fp
        path to the checkpoint file
    logger
        to log info about resuming from a checkpoint
    strict
        whether to strictly enforce that the keys in `state_dict` match the keys
        returned by this module’s `state_dict()` function. Default: True
    model_dir
        directory in which to save the object

    """
    if isinstance(checkpoint_fp, str) and checkpoint_fp.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_fp,
            model_dir=model_dir,
            map_location="cpu",
            check_hash=True,
        )
    else:
        if isinstance(checkpoint_fp, str):
            checkpoint_fp = Path(checkpoint_fp)

        if not checkpoint_fp.exists():
            raise FileNotFoundError(f"Given {str(checkpoint_fp)} does not exist.")
        checkpoint = torch.load(checkpoint_fp, map_location="cpu")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_fp)


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    output_dir = config.output_dir
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-{config.mode}-backend-{config.backend}"
        path = Path(config.output_dir, name)
        path.mkdir(parents=True, exist_ok=True)
        output_dir = path.as_posix()
    return Path(idist.broadcast(output_dir, src=0))


def save_config(config, output_dir):
    """Save configuration to config-lock.yaml for result reproducibility."""
    with open(f"{output_dir}/config-lock.yaml", "w") as f:
        OmegaConf.save(config, f)


def setup_logging(config: Any, custom_name="") -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`

    """
    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset} {custom_name}",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=config.output_dir / f"{config.mode}ing-info.log",
    )
    return logger


def setup_exp_logging(config, trainer, optimizers, evaluators, **kwargs):
    """Setup Experiment Tracking logger from Ignite."""
    log_every_iters = config.log_every_iters

    # this is modified from the ignite.contrib.engines.common.setup_tb_logging
    ###
    logger = TensorboardLogger(log_dir=config.output_dir, **kwargs)
    if optimizers is not None:
        if not isinstance(optimizers, (Optimizer, Mapping)):
            raise TypeError(
                "Argument optimizers should be either a single optimizer "
                "or a dictionary or optimizers"
            )

    if evaluators is not None:
        if not isinstance(evaluators, (Engine, Mapping)):
            raise TypeError(
                "Argument evaluators should be either a single engine "
                "or a dictionary or engines"
            )

    if log_every_iters is None:
        log_every_iters = 1

    # Log training loss iteratively
    logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_every_iters),
        tag="training_step",
        metric_names="all",
    )

    if optimizers is not None:
        # Log optimizer parameters
        if isinstance(optimizers, Optimizer):
            optimizers = {None: optimizers}

        for k, optimizer in optimizers.items():
            logger.attach_opt_params_handler(
                trainer,
                Events.ITERATION_STARTED(every=log_every_iters),
                optimizer,
                param_name="lr",
                tag="training_step" if k is None else f"training_step/{k}",
            )

    if evaluators is not None:
        # Log evaluation metrics
        if isinstance(evaluators, Engine):
            evaluators = {"validation": evaluators}

        event_name = (
            Events.ITERATION_COMPLETED if isinstance(logger, WandBLogger) else None
        )
        gst = global_step_from_engine(trainer, custom_event_name=event_name)
        for k, evaluator in evaluators.items():
            logger.attach_output_handler(
                evaluator,
                event_name=Events.COMPLETED,
                tag=k,
                metric_names="all",
                global_step_transform=gst,
            )
    ###

    return logger


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: Any,
    to_save_train: dict | None = None,
    to_save_eval: dict | None = None,
):
    """Setup Ignite handlers."""
    ckpt_handler_train = ckpt_handler_eval = None
    # checkpointing
    saver = DiskSaver(config.output_dir / "checkpoints", require_empty=False)
    ckpt_handler_train = Checkpoint(
        to_save_train,
        saver,
        filename_prefix=(
            config.filename_prefix
            if config.fold_num == 0
            else f"Kfold{config.fold_num}_{config.filename_prefix}"
        ),
        n_saved=config.n_saved,
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.save_every_iters),
        ckpt_handler_train,
    )
    global_step_transform = None
    if to_save_train.get("trainer", None) is not None:
        global_step_transform = global_step_from_engine(to_save_train["trainer"])
    metric_name = "miou"
    ckpt_handler_eval = Checkpoint(
        to_save_eval,
        saver,
        filename_prefix=(
            "best" if config.fold_num == 0 else f"Kfold{config.fold_num}_best"
        ),
        n_saved=config.n_saved,
        global_step_transform=global_step_transform,
        score_function=Checkpoint.get_default_score_fn(metric_name),
        score_name=metric_name,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_eval)

    # early stopping
    def score_fn(engine: Engine):
        return engine.state.metrics["miou"]

    es = EarlyStopping(config.patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)
    return ckpt_handler_train, ckpt_handler_eval


def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)
