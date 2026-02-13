# global_vars.py: Global variables used by DeepDrug3D
# This file was adapted from MaSIF's default global_vars.py, 
# MaSIF Copyright 2019 - Gainza P, Sverrisson F, Monti F, Rodola, Bronstein MM, Correia BE
# The original MaSIF code is under Apache License 2.0.

# Added to DeepDrug3D by Yu-Yuan Yang (2024):
# - Added this file to store global variables for DeepDrug3D. 
# - Integrated into FragBEST-Myo workflow 

# DeepDeug3D is under GNU General Public License v3.0 License.

import configparser
import sys

# Read config file.
config = configparser.ConfigParser()
path_args = __file__.split("/")[0:-1]
root_path = "/".join(path_args)
config.read(f"{root_path}/../../config.cfg")
config.sections()

# Set the environment variables for the programs used by DeepDrug3D.
msms_bin = ""
if "DLIGAND_BIN" in config["ThirdParty"]:
    dligand_bin = config["ThirdParty"]["DLIGAND_BIN"]
else:
    breakpoint()
    print("ERROR: DLIGAND_BIN not set. Variable should point to DLIGAND program.")
    sys.exit(1)


class NoSolutionError(Exception):
    pass
