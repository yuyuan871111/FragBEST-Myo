import os
import subprocess
from pathlib import Path
from random import randint
from time import strftime


def generate_pse(pdb_file, ply_file, pse_output, pymol_path, logger=None):
    """Generate a PyMOL session file (``.pse``) visualizing the PDB and PLY files.

    Args:
        pdb_file (str): Path to the PDB file.
        ply_file (str): Path to the PLY file.
        pse_output (str): Path to save the output .pse file.
        pymol_path (str): Path to the PyMOL executable.
        logger (logging.Logger, optional): Logger for logging messages.
            Defaults to ``None``.

    Returns:
        ``None``

    """
    pymol_py_script = f"""

import os
import sys


def find_project_root(marker=".git"):
    current_path = os.getcwd()
    while current_path != os.path.dirname(current_path):  # Stop at the filesystem root
        if marker in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)
    return None  # Return None if the marker is not found


project_root = find_project_root()
sys.path.append(project_root)

from pymol import cmd
from utils.thirdparty.FragBEST_pymol_plugin.FragBEST_pymol_plugin.loadPLY import \
    load_ply


cmd.load('{pdb_file}')
load_ply('{ply_file}', custom_name='{Path(ply_file).name}')
cmd.save('{pse_output}')
cmd.quit()

"""
    now = strftime("%y%m%d%H%M%S")
    random_int = randint(10, 99)
    temp_filename = f".temp_pymolscript_{Path(ply_file).name}_{now}_{random_int}.py"
    with open(temp_filename, "w") as f:
        f.write(pymol_py_script)

    subprocess.run([pymol_path, "-cq", temp_filename])
    os.remove(temp_filename)

    if logger:
        logger.info(f"Generated PyMOL session file: {Path(pse_output).name}")


def merge_pse(
    pse_file_list: list, merged_pse_output: str, pymol_path: str, logger=None
):
    """Merge multiple PyMOL session files into a single session file.

    Args:
        pse_file_list (list): List of paths to the ``.pse`` files to merge.
        merged_pse_output (str): Path to save a merged ``.pse`` file.
        pymol_path (str): Path to the PyMOL executable.
        logger (logging.Logger, optional): Logger for logging messages.
            Defaults to ``None``.

    Returns:
        ``None``

    """
    merge_script = f"""
from pymol import cmd


for pse_file in {pse_file_list}:
    cmd.load(pse_file, partial=1)
cmd.save('{merged_pse_output}')
cmd.quit()

"""

    now = strftime("%y%m%d%H%M%S")
    random_int = randint(10, 99)
    temp_filename = (
        f".temp_pymolscript_{Path(merged_pse_output).name}_{now}_{random_int}.py"
    )
    with open(temp_filename, "w") as f:
        f.write(merge_script)

    subprocess.run([pymol_path, "-cq", temp_filename])
    os.remove(temp_filename)

    if logger:
        logger.info(f"Generated PyMOL session file: {Path(merged_pse_output).name}")
