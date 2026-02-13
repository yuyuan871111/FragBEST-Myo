import os

# import self defined functions
from ..ppseg.ignite.utils import save_config
from .general import checking_workflow


def setup_configs(config):
    # configuration manager (output settings)
    os.makedirs(config.output_report_folderpath, exist_ok=True)
    config.check_index_path = (
        f"{config.input_check_path}/PPS_{config.traj_name}_aligned_index.txt"
    )
    config.protein_filename = f"PPS_{config.traj_name}_aligned"
    save_config(config, config.output_report_folderpath)

    return config


def run_checking(config):
    config = setup_configs(config)

    checking_workflow(config)
