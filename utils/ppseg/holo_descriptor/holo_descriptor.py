import json
import os

import numpy as np
import pandas as pd
from natsort import natsorted

from ..myo.default_config import HOLO_DESCRIPTOR_PRESETS, LIGAND_FRAG_INFO_PATH
from .holo_space import HoloSpace
from .pt_ratio import (
    get_class_predprobs,
    get_nonbck_class_pt_ratio,
    get_nonbck_ratio,
    get_num_interest_points,
    get_num_of_class_per_frame,
)


class HoloDescriptor:
    """HoloDescriptor class is used to calculate holo descriptors
    for a conformation based on the predictions from the deep-learning model.

    Arguments:
        ply_path: str, path to the .ply file. Required `pred` and `predprobs`
            in the attrubute.

    Example:
        .. code-block:: python

            from holo_descriptor import HoloDescriptor

            holo_descriptor = HoloDescriptor(ply_path)
            holo_descriptor.run()
            holo_descriptor.save(json_path)

    """

    def __init__(self, ply_path):
        """Initialize the HoloDescriptor class

        Args:
            ply_path (str): Path to the .ply file.
        """
        self.ply_path = ply_path
        self.results = {
            "class_predprobs": None,
            "overall_predprobs": None,
            "class_pt_ratio": None,
            "nonbck_ratio": None,
            "nonbck_class_pt_ratio": None,
            "num_of_classes": None,
            "num_interest_points": None,
            "holospace_volume": None,
            "holospace_frag_volumes": None,
        }

    def run(self):
        """Run to extract holo descriptors

        Args:
            None

        Attributes:
            results (dict): Dictionary containing the results of the descriptors.

        """
        # point-based descriptor
        class_predprobs, overall_predprobs, class_pt_ratio = get_class_predprobs(
            self.ply_path
        )
        self.results["class_predprobs"] = class_predprobs
        self.results["overall_predprobs"] = overall_predprobs
        self.results["class_pt_ratio"] = class_pt_ratio
        self.results["nonbck_ratio"] = get_nonbck_ratio(class_pt_ratio)
        self.results["nonbck_class_pt_ratio"] = get_nonbck_class_pt_ratio(
            class_pt_ratio
        )
        self.results["num_of_classes"] = get_num_of_class_per_frame(class_predprobs)
        self.results["num_interest_points"] = get_num_interest_points(self.ply_path)

        # volume-based descriptor
        self.holospace = HoloSpace(self.ply_path)
        self.results["holospace_volume"] = self.holospace.get_pocket_volume()
        self.results["holospace_frag_volumes"] = (
            self.holospace.get_pocket_frag_volumes()
        )

    def save(self, json_path):
        """Save the results to a json file

        Args:
            json_path (str): Path to the json file.

        """
        results = {k: convert_numpy_types(v) for k, v in self.results.items()}
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)


def save_descriptors(json_path, **kwargs):
    """Save the descriptors to a json file

    Args:
        json_path (str): Path to the json file.
        kwargs: Dictionary containing the descriptors.

    """
    kwargs = {k: convert_numpy_types(v) for k, v in kwargs.items()}
    with open(json_path, "w") as f:
        json.dump(kwargs, f, indent=2)


def convert_numpy_types(obj):
    """Convert numpy types to native python types

    Args:
        obj: Object to be converted

    Returns:
        obj: Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def read_descriptors(json_path):
    """Read the descriptors from a json file

    Args:
        json_path (str): Path to the json file.

    Returns:
        dict: Dictionary containing the descriptors.

    """
    with open(json_path) as f:
        return json.load(f)


class HoloDescriptorAnalyser:
    """HoloDescriptorAnalyser class is used to analyze the holo descriptors
    for a conformation based on the predictions from the deep-learning model.

    Arguments:
        source_path: str, path to the folder containing the json files.
        frag_info_path: str, path to the fragment information json file.
    
    Attributes:
        source_path (str): Path to the folder containing the holo-descriptor json files.
        frag_info_path (str): Path to the fragment information json file.
        files (list): List of json files in the source path (after `list_files`).
        descriptors_df (pd.DataFrame): A curated DataFrame containing the
            descriptors from the json files (after `read()`). `descriptors_df` will
            be added with the ``{colname}_zscore`` column (after `calculate_zscore()`).
            `descriptors_df` will be added with the ``overall_score`` and ``rank`` 
            columns (after `set_rank()`). `descriptors_df` will be added with the 
            ``holospace_frag_score`` column (after `holospace_frag_score()`).
        holospace_frag_volumes (pd.DataFrame): DataFrame containing the
            holospace fragment volumes (after `extract_holospace_frag_volume()`).
            `None` means using the default fragment information file.
    
    .. admonition:: A normal workflow for analyzing holo descriptors
        :class: note

        1. Create an instance of the HoloDescriptorAnalyser class
        2. List the files in the source path (`list_files()`)
        3. Read the descriptors from the json files (`read()`)
        4. Calculate the zscore of the column (`calculate_zscore()`)
        5. Set the rank of the conformations (`set_rank()`)
        6. Get the top n conformations (`top_n()`)
        7. (optional) Extract the holospace fragment volumes \
            (`extract_holospace_frag_volume()`)
     
    Example:
        .. code-block:: python

            from holo_descriptor import HoloDescriptorAnalyser

            # Create an instance of the HoloDescriptorAnalyser class
            holo_descriptor_analyser = HoloDescriptorAnalyser(
                                            source_path, frag_info_path
                                        )
            
            # List the files in the source path
            holo_descriptor_analyser.list_files()
            
            # Read the descriptors from the json files
            holo_descriptor_analyser.read()

            # Calculate the zscore of the column
            holo_descriptor_analyser.calculate_zscore("holospace_volume")
            
            # Set the rank of the conformations
            holo_descriptor_analyser.set_rank()
            
            # Get the top 5 conformations
            holo_descriptor_analyser.top_n(5)

            # (optional) Extract the holospace fragment volumes
            holo_descriptor_analyser.extract_holospace_frag_volume()
           
    """

    def __init__(self, source_path, frag_info_path: str = None):
        """Initialize the HoloDescriptorAnalyser class
        Args:
            source_path (str): Path to the folder containing the json files.
            frag_info_path (str): Path to the fragment information json file. `None`
                means using the default fragment information file.

        """
        self.source_path = source_path
        self.frag_info_path = (
            frag_info_path if frag_info_path is not None else LIGAND_FRAG_INFO_PATH
        )
        try:
            self.load_frag_info(self.frag_info_path) if self.frag_info_path else None
        except Exception as e:
            raise Exception(f"Error loading fragment info: {e}")

    def list_files(self):
        """List the files in the source path (*.json)"""
        files = [
            each for each in os.listdir(self.source_path) if each.endswith(".json")
        ]
        self.files = natsorted(files)
        print(f"Found {len(self.files)} files")

    def read(self, holospace_calc=False):
        """Read the descriptors from the json files

        Args:
            holospace_calc (bool): calculate the holospace fragment score at
                the same time, which might need more time (default: False)

        Returns:
            pd.DataFrame: DataFrame containing the descriptors

        """
        assert hasattr(self, "files"), "Please run list_files() first"
        descriptors = []
        for filename in self.files:
            json_path = f"{self.source_path}/{filename}"
            temp = read_descriptors(json_path)
            temp["filename"] = filename
            descriptors.append(temp)

        self.descriptors_df = pd.DataFrame(descriptors)

        # check the number of interest points
        self.descriptors_df["warnings"] = [
            "# Too few interest points (<200) " if the_filter else ""
            for the_filter in (
                (self.descriptors_df["num_interest_points"] < 200)
                | (self.descriptors_df["num_interest_points"].isna())
            )
        ]

        # calculate the holospace fragment score (if applicable)
        if hasattr(self, "fragment_vol") and holospace_calc:
            self.extract_holospace_frag_volume()
            self.holospace_frag_score()

        return self.descriptors_df

    def extract_holospace_frag_volume(self, num_frag=6):
        """Extract the holospace fragment volumes (if needed)

        Args:
            num_frag (int): Number of fragments (default: 6)

        Returns:
            pd.DataFrame: DataFrame containing the holospace
                fragment volumes

        """
        assert hasattr(self, "descriptors_df"), "Please run read() first"
        data = self.descriptors_df["holospace_frag_volumes"].tolist()
        data = [each if each is not None else [0] * num_frag for each in data]
        self.holospace_frag_volumes = pd.DataFrame(
            data,
            columns=[f"holospace_frag_vol_{i}" for i in range(1, 1 + num_frag)],
        )
        return self.holospace_frag_volumes

    def calculate_zscore(self, colname, use_presets: str = None):
        """Calculate the zscore of the column (need specify the column name)

        Args:
            colname (str): Column name to be calculated
            use_presets (str): Use presets `pr` or `pps` ("pr" for post-rigor state
                myosin; "pps" for pre-powerstroke state myosin) for mean and std
                (default: None). If `None`, the mean and std will be calculated
                from the data.

        """
        # santity checks
        assert hasattr(self, "descriptors_df"), "Please run read() first"
        assert colname in self.descriptors_df.columns, f"{colname} not found"
        data = self.descriptors_df[colname].values
        data = np.nan_to_num(data)

        # use presets for mean and std
        if use_presets is not None:
            assert HOLO_DESCRIPTOR_PRESETS.get(use_presets) is not None, (
                f"Presets {use_presets} not found. Available presets: "
                f"{list(HOLO_DESCRIPTOR_PRESETS.keys())}"
            )
            preset = HOLO_DESCRIPTOR_PRESETS[use_presets]

            assert preset.get(colname) is not None, (
                f"{colname} not found in presets {use_presets}"
            )
            data_mean = preset[colname]["mean"]
            data_std = preset[colname]["std"]

        # check if the column has only one unique value (to prevent division by zero)
        elif len(np.unique(data)) == 1:
            print(f"Warning: {colname} has only one unique value. Skipping...")
            return

        else:
            data_mean = data.mean()
            data_std = data.std()

        self.descriptors_df[f"{colname}_zscore"] = (data - data_mean) / data_std

    def set_rank(
        self,
        weights: np.ndarray = None,
        zscore_columns: list = None,
        filter_warning=True,
    ):
        """Set the rank of the conformations (based on the zscore columns)

        Args:
            weights (np.ndarray): weights for the zscore columns
                (default = ``None``, equal weights for all zscore columns)
            zscore_columns (list): zscore columns to be used: aligned to weights,
                (default = ``None``, use all zscore columns in the data frame)
            filter_warning (bool): sort by warnings first

        """
        assert hasattr(self, "descriptors_df"), "Please run read() first"

        zscore_columns_exist = [
            each for each in self.descriptors_df.columns if each.endswith("zscore")
        ]
        if zscore_columns is None:
            zscore_columns = zscore_columns_exist
        else:
            assert all(each in zscore_columns_exist for each in zscore_columns), (
                f"Zscore columns not found: {zscore_columns}"
            )

        assert len(zscore_columns) > 0, (
            "No zscore columns found, please run calculate_zscore() first"
        )

        if weights is None:
            weights = np.ones(len(zscore_columns)) / len(zscore_columns)
        else:
            assert len(weights) == len(zscore_columns), (
                f"Weights (len: {len(weights)}) must match zscore columns length "
                f"({len(zscore_columns)})"
            )

        # calculate overall score
        descriptors_df_overall_score = np.sum(
            self.descriptors_df[zscore_columns].values * weights,
            axis=1,
        )
        self.descriptors_df["overall_score"] = descriptors_df_overall_score

        # rank the conformations
        if filter_warning:
            self.descriptors_df.sort_values(
                ["warnings", "overall_score"], ascending=[True, False], inplace=True
            )
        else:
            self.descriptors_df.sort_values(
                ["overall_score"], ascending=False, inplace=True
            )
        self.descriptors_df["rank"] = range(1, len(self.descriptors_df) + 1)
        self.descriptors_df.sort_index(inplace=True)

    def top_n(self, n=5):
        """Get the top n conformations.

        Args:
            n (int): Number of top conformations (default: 5)

        Returns:
            pd.DataFrame: DataFrame containing the top n conformations

        """
        assert hasattr(self, "descriptors_df"), "Please run read() first"
        assert "rank" in self.descriptors_df.columns, "Please run set_rank() first"
        descriptors_df = self.descriptors_df.sort_values(by="rank", ascending=True)
        return descriptors_df.head(n)

    def load_frag_info(self, frag_info_path):
        """Load the fragment information

        Args:
            frag_info_path (str): Path to the fragment information json file.

        Returns:
            dict: Dictionary containing the fragment information

            .. note::
                - `self.fragment_info`: dict
                - `self.fragment_vol`: np.ndarray (fragment volumes)

        """
        self.fragment_info = read_descriptors(frag_info_path)
        self.fragment_vol = (
            pd.DataFrame(self.fragment_info).T["fragment_vol"][1:].values
        )
        return self.fragment_info

    def holospace_frag_score(self):
        """Calculate the holospace fragment score

        Returns:
            pd.DataFrame: DataFrame containing the holospace
                fragment score (`holospace_frag_score`)

        """
        assert hasattr(self, "fragment_vol"), "Please run load_frag_info() first"
        assert hasattr(self, "descriptors_df"), "Please run read() first"
        assert hasattr(self, "holospace_frag_volumes"), (
            "Please run extract_holospace_frag_volume() first"
        )

        # check how many times the holospace is greater than the fragment volume
        holospace_frag_vol_fold = self.holospace_frag_volumes / self.fragment_vol

        # holospace 10-fold greater than frag vol, generate a warning
        warning_msg = []
        frag_nums = np.array(
            [
                each.replace("holospace_frag_vol_", "")
                for each in holospace_frag_vol_fold.columns
            ]
        )

        for each_row in holospace_frag_vol_fold.values:
            match_items = frag_nums[each_row >= 10]
            if len(match_items) > 0:
                warning_msg.append(
                    f"# HoloSpace too big (>10-fold) at frags: {', '.join(match_items)}"
                )
            else:
                warning_msg.append("")

        self.descriptors_df["warnings"] = [
            f"{each}{warn}"
            for each, warn in zip(self.descriptors_df["warnings"], warning_msg)
        ]

        # set the score
        holospace_frag_vol_fold[holospace_frag_vol_fold >= 1] = 1
        self.holospace_each_frag_score = holospace_frag_vol_fold
        score = self.holospace_each_frag_score.mean(axis=1)
        self.descriptors_df["holospace_frag_score"] = score

        return self.descriptors_df
