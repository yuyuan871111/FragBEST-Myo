import numpy as np


def normalize(data, method="minmax+-1"):
    """Normalize the data.

    Parameters
    ----------
    data : np.ndarray
        The data to be normalized.
    method : str, optional
        The normalization method ['minmax+-1', 'minmax', 'zscore'].
        The default is 'minmax+-1'.
    ----------

    Returns
    -------
        normalized_data : np.ndarray
    ----------

    """
    if method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "minmax+-1":
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    elif method == "zscore":
        return (data - np.mean(data)) / np.std(data)
    else:
        raise ValueError("Invalid normalization method")
