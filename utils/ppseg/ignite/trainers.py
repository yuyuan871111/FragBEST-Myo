from typing import Any

import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events
from ignite.metrics import Metric
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler


def setup_trainer(
    config: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    device: str | torch.device,
    train_sampler: Sampler,
) -> Engine | DeterministicEngine:
    scaler = GradScaler(enabled=config.use_amp)

    def train_function(engine: Engine, batch: Any):
        model.train()
        data = batch["data"]
        data = data.to(device)
        labels = (
            # batch["label"].long().unsqueeze(1)
            batch["label"].long()
        )  # dimension: (batch_size, 1, H, W, D)
        labels = labels.to(device)

        # Forward pass and calculate loss
        with autocast(config.use_amp):
            outputs = model(data)
            if "masked" in config.loss_fn:
                mask = data[:, -1, :, :, :].unsqueeze(1)
                loss = loss_fn(outputs, labels, mask) / config.accumulation_steps
            else:
                loss = loss_fn(outputs, labels) / config.accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights and zero the gradients
        if (engine.state.iteration + 1) % config.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metric = {"epoch": engine.state.epoch, "train_loss": loss.item()}
        engine.state.metrics = metric
        return metric

    trainer = Engine(train_function)

    # set epoch for distributed sampler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)

    return trainer


def setup_evaluator(
    config: Any,
    model: Module,
    metrics: dict[str, Metric],
    device: str | torch.device,
) -> Engine:
    @torch.no_grad()
    def evaluation_function(engine: Engine, batch: Any):
        model.eval()
        data = batch["data"]
        data = data.to(device)
        labels = batch["label"].long()  # dimension: (batch_size, 1, H, W, D)
        labels = labels.to(device)

        with autocast(config.use_amp):
            outputs = model(data)
            mask = data[:, -1, :, :, :].unsqueeze(1)
            # mask's dimension: (batch_size, 1, H, W, D)

        return {"outputs": outputs, "labels": labels, "mask": mask}

    evaluator = Engine(evaluation_function)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
