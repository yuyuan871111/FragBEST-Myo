"""Modified by Yu-Yuab (Stuart) Yang - 2024
from ignite generator template - 2024

"""

from argparse import Namespace

import ignite.distributed as idist
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from ..dataset import PPSegDataset
from .transforms import setup_transforms


def dataset_sample_loader(dataset, train_idx, val_idx, config: Namespace):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = idist.auto_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
    )
    val_loader = idist.auto_dataloader(
        dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def _read_data(config: Namespace):
    try:
        transform_func = (
            setup_transforms(config)
            if (hasattr(config, "transform")) and (config.transform is not None)
            else None
        )
        whole_dataset = PPSegDataset(
            root_dir=config.data_path,
            except_channels=[3],
            sample_size=config.sample_size,
            seed=config.seed,
            transform=transform_func,
        )
    except RuntimeError as e:
        raise RuntimeError("Dataset not found.") from e
    return whole_dataset


def setup_test_data(config: Namespace):
    test_dataset = _read_data(config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    return test_loader


def setup_data(config: Namespace):
    # read dataset
    whole_dataset = _read_data(config)

    # split training and validation set
    dataset_size = len(whole_dataset)
    indices = list(range(dataset_size))

    if config.cross_val:
        # cross-validation
        config.fold = int(config.fold)
        assert config.fold > 1, "n_splits must be greater than 1 for cross-validation"

        splits = KFold(n_splits=config.fold, shuffle=True, random_state=config.seed)

        return whole_dataset, indices, splits

    else:
        # without cross-validation
        assert config.data_split_ratio < 1.0 and config.data_split_ratio > 0, (
            "data_split_ratio must be less than 1.0 and greater than 0"
        )

        train_idx, val_idx = train_test_split(
            indices,
            train_size=config.data_split_ratio,
            random_state=config.seed,
            shuffle=True,
        )

        train_loader, val_loader = dataset_sample_loader(
            whole_dataset, train_idx, val_idx, config
        )
        return train_loader, val_loader


# Output transform functions
def output_transform_normal(output):
    # return output["outputs"], output["labels"].unsqueeze(1)
    return (
        output["outputs"],  # dimension: (batch_size, C, H, W, D)
        output["labels"],  # dimension: (batch_size, 1, H, W, D)
    )


def output_transform_masked(output):
    return output["outputs"], output["labels"], {"mask": output["mask"]}


def output_transform_mask(output):
    # This implementation mix the data for each batch,
    # because the interest points for each image are different
    # This might lead to a slight difference in the evaluation results
    # However, the results are still acceptable and the training process is faster
    real_bz = output["outputs"].shape[0]
    classes = output["outputs"].shape[1]
    mask = (output["mask"].squeeze(1) == 1).view(real_bz, -1)
    # mask's dimension: (batch_size, H*W*D)
    labels = output["labels"].view(real_bz, -1)
    # label's dimension: (batch_size, H*W*D)
    labels = labels[mask].unsqueeze(0)
    outputs = output["outputs"].view(real_bz, classes, -1)
    outputs = outputs.permute(0, 2, 1)[mask].unsqueeze(0).permute(0, 2, 1)

    return outputs, labels
