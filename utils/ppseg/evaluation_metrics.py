import numpy as np


def calculate_rmsd(ref_coords: np.array, pred_coords: np.array):
    """Calculate the root mean square deviation (RMSD)
    between two sets of coordinates."""
    ref_coords = ref_coords.flatten()
    pred_coords = pred_coords.flatten()
    diff = ref_coords - pred_coords
    rmsd = np.sqrt(np.mean(diff**2))
    return rmsd
