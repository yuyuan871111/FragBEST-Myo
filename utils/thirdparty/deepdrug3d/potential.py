# This file is derived from DeepDrug3D 
# Original copyright (c) 2019 Pu et al.

# Modifications:
# - Adapted for Python 3.10 compatibility 
# - Integrated into FragBEST-Myo workflow 

# Modified by Yu-Yuan (Stuart) Yang, 2024 
# Licensed under GPL-3.0 (see LICENSE)

import os
import os.path as osp
import random
import string
import subprocess
import time

import numpy as np
import pandas as pd
from openbabel import pybel

try:
    from .global_vars import dligand_bin
except ImportError:
    from global_vars import dligand_bin

"""
The following functions create a new dummy mol2 file for the DFIRE calculation
"""


def preparation_mol2_file(input_pdb_path: str, output_folder: str):
    # using opebabel's pybel to write mol2 file
    pdb_filename = osp.basename(input_pdb_path).split(".")[0]
    output_trans_mol2_path = osp.join(output_folder, pdb_filename + ".mol2")
    print(f"Saving the binding pocket aligned mol2 file to: {pdb_filename}.mol2")
    mol = next(pybel.readfile("pdb", input_pdb_path))
    mol.write("mol2", output_trans_mol2_path, overwrite=True)
    return None


"""
The following functions calculate the DFIRE potentials using the dligand program proivded in the DFIRE paper
"""


# replace the coordinates in the original string with new coordinates
def replace_coord(original_string, new_coord):
    temp = "{:>8}  {:>8}  {:>8}".format(new_coord[0], new_coord[1], new_coord[2])
    new_string = original_string.replace(" 50.0000   51.0000   52.0000", temp)
    return new_string


# replace the atom type in the original string with the new atom type
def replace_type(original_string, new_type):
    temp = "{:6}".format(new_type)
    new_string = original_string.replace("N.3   ", temp)
    return new_string


# USING THE DFIRE FUNCTION
def single_potEnergy(loc1, ld_type_list, mol2_in_string, protein_file):
    temp_loc = loc1.round(4)
    Es = []
    append = Es.append
    r1 = replace_coord(mol2_in_string, temp_loc)
    random_string = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(11)
    )
    temp_filename = (
        "/dev/shm/" + random_string + ".mol2"
    )  # TODO: this controls the place to generate the temporary mol2 file
    for item in ld_type_list:
        rrr = replace_type(r1, item)
        f = open(temp_filename, "w")
        f.write(rrr)
        f.close()
        breakpoint()
        child = subprocess.Popen(
            [f"{dligand_bin}/dligand-linux", temp_filename, protein_file],
            stdout=subprocess.PIPE,
            shell=True,
        )
        child.wait()
        out = child.communicate()
        out = out[0].decode("utf-8")
        a = out.replace("\n", "")
        if a != "":
            b = float(a)
            append(b)
    Es = np.array(Es)
    os.remove(temp_filename)
    return Es


def minmax_scale(X, axis=0):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (X.max() - X.min()) + X.min()
    return X_scaled


def potEnergy(binding_site, mol2_in_path, protein_file):
    ld_type_list = [
        "C.2",
        "C.3",
        "C.ar",
        "F",
        "N.am",
        "N.2",
        "O.co2",
        "N.ar",
        "S.3",
        "O.2",
        "O.3",
        "N.4",
        "P.3",
        "N.pl3",
    ]
    total_potE = {"loc": [], "potE": []}
    mol2_in_file = open(mol2_in_path)
    mol2_in_string = mol2_in_file.read()
    potEs = np.array(
        [
            single_potEnergy(loc1, ld_type_list, mol2_in_string, protein_file)
            for loc1 in binding_site
        ]
    )
    total_potE["potE"] = minmax_scale(potEs, axis=0)
    total_potE["loc"] = binding_site
    return total_potE


def cal_DFIRE_potential(new_site, input_trans_mol2_path):
    ss = time.time()
    print("Calculating of the binding site potential energy")
    total_potE = potEnergy(new_site, "dummy_mol2.mol2", input_trans_mol2_path)
    print(
        "The total time of binding site potential energy computation is: {:.4f} seconds".format(
            time.time() - ss
        )
    )
    df1 = pd.DataFrame(total_potE["loc"], columns=["x", "y", "z"])
    df2 = pd.DataFrame(
        total_potE["potE"],
        columns=[
            "C.2",
            "C.3",
            "C.ar",
            "F",
            "N.am",
            "N.2",
            "O.co2",
            "N.ar",
            "S.3",
            "O.2",
            "O.3",
            "N.4",
            "P.3",
            "N.pl3",
        ],
    )
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
    return df
