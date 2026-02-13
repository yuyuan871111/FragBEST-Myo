import string
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns

from ...thirdparty.deepdrug3d.visualization import makeLine
from ..voxelization import map_voxel_to_xyz

warnings.filterwarnings("ignore")


_VISUAL_TYPE = Literal["natural", "fragment_idx"]


def save_grid_as_pdb(
    grid_file_path: str,
    grid_w_values_xyz: np.array,
    additional_features: np.array = None,
    prob_in_occ: np.array = None,
):
    # check
    if additional_features is not None:
        assert len(additional_features) == grid_w_values_xyz.shape[0], (
            "The length of the additional features should be the same as "
            "the grid values"
        )
    if prob_in_occ is not None:
        assert len(prob_in_occ) == grid_w_values_xyz.shape[0], (
            "The length of the probability in occupancy should be the "
            "same as the grid values"
        )

    num_range = np.linspace(0, 25, 26, dtype=int)
    chr_range = list(string.ascii_uppercase[:27])
    num_chr_dict = dict(zip(num_range, chr_range))
    with open(grid_file_path, "w") as in_strm:
        for k in range(len(grid_w_values_xyz)):
            temp_coords = grid_w_values_xyz[k, :]
            cl = additional_features[k] if additional_features is not None else 1
            occ = prob_in_occ[k] if prob_in_occ is not None else 1
            temp_string = makeLine(temp_coords, cl, k, num_chr_dict, occ)
            in_strm.write(temp_string)


def save_fragment_for_ligand_using_chainID(
    input_pdb_filepath: str,
    ligand_output_filepath: str,
    ligand_name: str = "2OW",
    mol_f_idx: list = [],
    using_type: _VISUAL_TYPE = "natural",
):
    """Save the fragement information for the ligand using chainID

    Input:
        - `input_pdb_filpath` [str]: path to the input pdb file
        - `ligand_output_filepath` [str]: path to the output ligand pdb file
        - `ligand_name` [str]: name of the ligand
        - `mol_f_idx` [list]: list of the fragment index
            example:
                mol_f_idx = [
                    [0, 1, 2],  # fragment 1
                    [3, 4, 5]   # fragment 2
                ]
        - `using_type` [_VISUAL_TYPE]: type of the using visualisation type
        (natural or fragment_idx)

    Output:
        - save the ligand with the fragment information (visualise by the chainID)
    """
    ref = mda.Universe(input_pdb_filepath)

    ref = fragmentation_from_universe(
        universe=ref,
        ligand_name=ligand_name,
        mol_f_idx=mol_f_idx,
        using_type=using_type,
    )
    ligand = ref.select_atoms(f"resname {ligand_name}")

    with mda.Writer(ligand_output_filepath, ligand.n_atoms) as W:
        W.write(ligand)


def fragmentation_from_universe(
    universe: mda.Universe,
    ligand_name: str,
    mol_f_idx: list[list[int]],
    using_type: _VISUAL_TYPE = "natural",
) -> mda.Universe:
    """Fragmentation for the ligand using the chainID"""
    universe.add_TopologyAttr("chainID")
    ligand = universe.select_atoms(f"resname {ligand_name}", updating=True)
    ligand_atoms_ids = ligand.atoms.ids

    for idx, each_fragment_idx in enumerate(mol_f_idx):
        fragment_list = [f"id {ligand_atoms_ids[each]}" for each in each_fragment_idx]
        fragment_str = " or ".join(fragment_list)
        fragment_atoms = ligand.select_atoms(fragment_str)
        if using_type == "natural":
            fragment_atoms.atoms.chainIDs = string.ascii_uppercase[idx % 26]
        elif using_type == "fragment_idx":
            fragment_atoms.atoms.chainIDs = str(idx + 1)
        else:
            raise ValueError(
                "The using_type should be either 'natural' or 'fragment_idx'"
            )
    return universe


def save_voxels_as_pdb(
    voxel_data: np.array,
    save_pdb_filepath: str,
    pocket_center: np.array = None,
    mask: np.array = None,
    spacing: float = 0.5,
    r: int = 16,
    filter_dummy: bool = True,
    feature_col_idx: int = 0,
    probs_col_idx: int = None,
) -> None:
    """Save the voxel data as pdb file.

    Input:
        - `voxel_data` [np.array]: voxel data with dimension: (C, X, Y, Z)
        - `save_pdb_filepath` [str]: path to the output pdb file
        - `pocket_center` [np.array]: pocket center [x, y, z]
        - `mask` [np.array]: mask for the data with dimension: (X, Y, Z)
        - `spacing` [float]: spacing between the voxels
        - `r` [int]: radius of the voxel
        - `filter_dummy` [bool]: filter the dummy voxels
        - `feature_col_idx` [int]: index of the feature column
        - `probs_col_idx` [int]: index of the feature column

    Output:
        - save the voxel data as pdb file at the `save_pdb_filepath`
    """
    # Map the voxel to xyz
    grids = map_voxel_to_xyz(
        voxel_data,
        spacing=spacing,
        r=r,
        pocket_center=pocket_center,
        filter_dummy=filter_dummy,
        mask=mask,
    )  # (N, 3+features)

    # Sample the grids is not filtering the dummy voxels
    if not filter_dummy and mask is None:
        grids = grids[1::3, :]
    grids_xyz = grids[:, :3]
    features = grids[:, 3 + feature_col_idx]
    probs = grids[:, 3 + probs_col_idx] if probs_col_idx is not None else None

    # Save the grid as pdb
    save_grid_as_pdb(
        grid_file_path=save_pdb_filepath,
        grid_w_values_xyz=grids_xyz,
        additional_features=features,
        prob_in_occ=probs,
    )
    return None


def plot_training_process(data):
    """Visualize the training process"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), dpi=300)
    axes[0].plot(data["training_loss"], label="Train Loss")
    axes[0].plot(data["validation_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(data["training_mIoU"], label="Train mIOU")
    axes[1].plot(data["validation_mIoU"], label="Val mIOU")
    axes[1].plot(data["training_accuracy"], label="Train mACC")
    axes[1].plot(data["validation_accuracy"], label="Val mACC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metrics")
    axes[1].legend()
    return fig, axes


def plot_violin_for_each_class(data, ylabel, palette):
    """Visualize the violin plot"""
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    sns.violinplot(data, ax=axes, inner="quart", palette=palette)
    axes.set_xlabel("Class (chemical fragments)", fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)
    return fig, axes


def plot_metrics_w_trajectory(x_data, y_data):
    """Visualize the metrics with trajectory
    metric: [accuracy, miou]
    """
    fig, axes = plt.subplots(1, 1, figsize=(28, 4), dpi=300)
    axes.plot(x_data, y_data["accuracy"], color="tab:blue")
    axes.plot(x_data, y_data["miou"], color="tab:orange")
    axes.legend(["Accuracy", "mIoU"])
    axes.set_xlabel("Time (ns)", fontsize=16)
    axes.set_ylabel("Metrics", fontsize=16)
    return fig, axes


def plot_confusion_matrix(cf_matrix, normalised_by_row=True, ax=None):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=300) if ax is None else (None, ax)
    group_counts = [f"{value:0.0f}" for value in cf_matrix.flatten()]
    if normalised_by_row:
        percentages = (cf_matrix.T / cf_matrix.sum(axis=1)).T
        group_percentages = [f"{value:.2%}" for value in percentages.flatten()]
    else:
        group_percentages = [
            f"{value:.2%}" for value in cf_matrix.flatten() / np.sum(cf_matrix)
        ]
    annotation = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    annotation = np.asarray(annotation).reshape(7, 7)
    sns.heatmap(cf_matrix, annot=annotation, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    return fig, ax if fig is not None else ax
