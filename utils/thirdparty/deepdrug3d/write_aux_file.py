# This file is derived from DeepDrug3D 
# Original copyright (c) 2019 Pu et al.

# Modifications:
# - Adapted for Python 3.10 compatibility 
# - Integrated into FragBEST-Myo workflow 

# Added by Yu-Yuan (Stuart) Yang, 22 Jan, 2024 
# Licensed under GPL-3.0 (see LICENSE)


def write_aux_file(
    aux_filepath, binding_residue_ids: str = None, binding_site_center: str = None
):
    """
    Write auxiliary file for DeepDrug3D.
    :param binding_residue_ids: str (e.g. "1 2 3 4 5")
    :param binding_site_center: str (e.g. "0.0 0.0 0.0")
    :return: None
    """
    with open(aux_filepath, "w") as f:
        f.write(
            f"BindingResidueIDs:{binding_residue_ids}\n"
        ) if binding_residue_ids else f.write("BindingResidueIDs:\n")
        f.write(
            f"BindingSiteCenter:{binding_site_center}\n"
        ) if binding_site_center else f.write("BindingSiteCenter:\n")
