from rdkit import Chem


def fragment_on_bonds(
    ligand_file_path: str,
    break_bond_idx: list,
    removeHs: bool = False,
    with_star: bool = False,
) -> [tuple, list]:
    """ligand_file_path [str]: path to the ligand file
    break_bond_idx [int]: index of the bond to be broken
    removeHs [bool]: whether to remove the hydrogen atoms or not
    with_star [bool]: whether to include the star atoms in the fragment or not
    """
    mol = Chem.MolFromPDBFile(ligand_file_path, removeHs=removeHs)

    mol_f = Chem.FragmentOnBonds(mol, break_bond_idx)
    mol_fragment = Chem.GetMolFrags(mol_f, asMols=True, sanitizeFrags=False)
    mol_f_idx = Chem.GetMolFrags(mol_f, asMols=False, sanitizeFrags=False)

    star_idx = [
        idx for idx, atom in enumerate(mol_f.GetAtoms()) if atom.GetSymbol() == "*"
    ]

    # check whether to return the fragment index list with removing star
    # atoms from the fragment
    if not with_star:
        new_mol_f_idx = []
        for each_f_idx in mol_f_idx:
            f_idx_without_star = [each for each in each_f_idx if each not in star_idx]
            new_mol_f_idx.append(f_idx_without_star)
        return mol_fragment, new_mol_f_idx

    else:
        return mol_fragment, mol_f_idx


def fragment_idx_label_dict(labels_info: dict) -> dict:
    """labels_info [dict]: dictionary containing the label information
    output: dictionary containing the fragment index and label

    Example:
        labels_info = {
            0: {"name": "out of the threshold"},
            1: {"name": "fragment 1", "fragments_idx": [0, 3, 4]},
            2: {"name": "fragment 2", "fragments_idx": [1, 2]}
        }
        fragidx_label_dict = fragment_idx_label_dict(labels_info=labels_info)
    Output:
        fragidx_label_dict = {
            0: 1,
            1: 2,
            2: 2,
            3: 1,
            4: 1
        }

    """
    fragidx_label_dict = {}
    for each_label in labels_info:
        if (each_label == 0) or (each_label == "0"):
            pass
        else:
            for each_frag_idx in labels_info[each_label]["fragments_idx"]:
                fragidx_label_dict[each_frag_idx] = each_label
    return fragidx_label_dict
