import os
import random

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


def list_files(directory, extension):
    """List all files in a directory with a given extension.

    Parameters
    ----------
    directory : str
        The directory to list the files from.
    extension : str
        The extension of the files to list.
    ----------

    Returns
    -------
        files : list
    ----------

    """
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    return natsorted(files)


def write_h5(data, h5_filename, label=None):
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset("raw", data=data)
        if label is not None:
            f.create_dataset("label", data=label)

    return None


def load_h5(h5_filename):
    f = h5py.File(h5_filename, "r")
    data = f["raw"][:]
    label = f["label"][:] if "label" in f else None
    f.close()
    return (data, label)


def normalize_per_channel(data, except_channels: list = []):
    """Normalize the input data per channel.

    Parameters
    ----------
    data : np.ndarray
        The input data to normalize.
    ----------

    Returns
    -------
        data : np.ndarray
    ----------

    """
    assert len(data.shape) == 4, "Data must be 4D (C, H, W, D)"
    if except_channels != []:
        for i in range(data.shape[0]):
            if i not in except_channels:
                data[i] = (data[i] - data[i].mean()) / data[i].std()
    else:
        data = ((data.T - data.mean((1, 2, 3))) / data.std((1, 2, 3))).T
    return data


def preprocess_h5(data: np.array, label: np.array = None, except_channels: list = []):
    """Preprocess the data and label.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    label : np.ndarray
        The input label.
    except_channels : list
        The channels to exclude from normalization.
    ----------

    Returns
    -------
        data : np.ndarray
        label : np.ndarray
    ----------

    """
    # Normalization
    data = np.array(data)  # transform the data into numpy array
    data = normalize_per_channel(data, except_channels=except_channels)

    # Label dimension from (H, W, D) to (1, H, W, D)
    if label is not None:
        label = np.array(label)
        label = np.expand_dims(label, 0)
    return data, label


class PPSegDataset(Dataset):
    def __init__(
        self,
        root_dir,
        extension=".h5",
        except_channels: list = [],
        sample_size: int = None,
        seed: int = 42,
        transform=None,
    ):
        """root_dir : str"""
        self.root_dir = root_dir
        self.data_list = list_files(self.root_dir, extension=extension)
        self.except_channels = except_channels
        self.transform = transform
        random.seed(seed)
        random.shuffle(self.data_list)

        assert isinstance(sample_size, int) or sample_size is None, (
            "sample_size must be an integer or None"
        )
        if sample_size is not None:
            self.data_list = self.data_list[:sample_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, self.data_list[idx])
        assert os.path.exists(filepath), f"{filepath} does not exist"
        data, label = load_h5(filepath)
        data, label = preprocess_h5(data, label, except_channels=self.except_channels)

        sample = {
            "data": torch.FloatTensor(data),
            "label": torch.FloatTensor(label),
            "filename": self.data_list[idx],
        }
        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_mask(data):
    """Get the mask from the data."""
    return data[-1, :, :, :] == 1


def get_mask_col_idx(data):
    """Get the mask from the data.
    input:
        - data [C, H, W, D]
    """
    mask_col_idx = data.shape[0]
    return mask_col_idx - 1
