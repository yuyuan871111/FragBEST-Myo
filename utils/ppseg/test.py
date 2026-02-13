import pickle
from pprint import pformat
from typing import Any

import ignite.distributed as idist
from ignite.metrics import Accuracy, ConfusionMatrix, DiceCoefficient, IoU, mIoU
from torch import load

from ..seed import set_seed
from .ignite.data import output_transform_mask, setup_test_data
from .ignite.models import setup_model
from .ignite.trainers import setup_evaluator
from .ignite.utils import save_config, setup_logging, setup_output_dir


def run_testing(local_rank: int, config: Any):
    # make a certain seed
    rank = idist.get_rank()
    set_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)
    config.output_dir = output_dir

    # create dataloaders
    dataloader_test = setup_test_data(config)
    config.le = len(dataloader_test)

    # setup model, device
    device = idist.device()
    model = setup_model(config)
    model.load_state_dict(load(config.model_path, map_location=device))
    model = idist.auto_model(model)

    # setup metrics
    cm = ConfusionMatrix(
        num_classes=config.num_classes, output_transform=output_transform_mask
    )
    test_metrics = {
        "accuracy": Accuracy(output_transform=output_transform_mask),
        "miou": mIoU(cm),
        "iou": IoU(cm),
        "dice": DiceCoefficient(cm),
        "cm": cm,
    }

    # setup ignite evaluator
    test_evaluator = setup_evaluator(config, model, test_metrics, device)

    # Attach metrics to the evaluators
    for name, metric in test_metrics.items():
        metric.attach(test_evaluator, name)

    # setup ignite logger with python logging
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    test_evaluator.logger = setup_logging(config, custom_name="Evaluator")

    # calculate the results
    results = {
        "filename": [],
        "accuracy": [],
        "miou": [],
        "iou": [],
        "dice": [],
        "cm": [],
    }
    it = iter(dataloader_test)
    for idx in range(config.le):
        batch = next(it)
        state = test_evaluator.run([batch])

        for key in results:
            if key == "filename":
                results[key].append(batch["filename"][0])
            else:
                results[key].append(state.metrics[key])

        logger.info(
            f"{idx + 1} [{results['filename'][idx]}] - "
            f"accuracy: {results['accuracy'][idx]:.3f}, "
            f"mIoU: {results['miou'][idx]:.3f}"
        )

    # save the results
    with open(f"{config.output_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
