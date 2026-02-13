import os

# import self defined functions
from ..ppseg.ignite.utils import save_config
from .general import apo_data_preparation_recipe


def setup_configs(config):
    # configuration manager (output settings)
    os.makedirs(config.output_path, exist_ok=True)

    config.p_filename = f"PPS_{config.traj_name}_aligned"
    config.output_aligned_traj_filepath = (
        f"{config.output_path}/PPS_Apo_{config.traj_name}_aligned.xtc"
    )
    config.output_p_data_folderpath = (
        f"{config.output_path}/{config.traj_name}_aligned_protein"
    )
    config.output_index_path = (
        f"{config.output_p_data_folderpath}/{config.p_filename}_index.txt"
    )

    save_config(config, config.output_path)

    return config


def run_preparation(config):
    """Run the dataset preparation with configuration and logging manager"""
    config = setup_configs(config)

    apo_data_preparation_recipe(config)
