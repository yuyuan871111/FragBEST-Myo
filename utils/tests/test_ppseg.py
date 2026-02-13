import os

from ..ppseg.fragment import fragment_on_bonds


def test_fragment_on_bonds():
    # Set the path to the ligand file
    ligand_file_path = f"{os.getcwd()}/tests/test_data/ref_ligand_OMB.pdb"

    # Set the index of the bond to be broken
    break_bond_idx = [3, 7, 8, 13, 17]

    # Test function
    _, mol_f_idx = fragment_on_bonds(
        ligand_file_path=ligand_file_path,
        break_bond_idx=break_bond_idx,
        removeHs=False,
        with_star=False,
    )

    # Set the expected output
    expected_output = [
        [0, 1, 2, 3, 30, 45, 52],
        [4, 5, 6, 7, 26, 27, 29, 31, 32, 43, 44, 46, 47, 50],
        [8, 33, 48],
        [9, 10, 11, 12, 13, 25, 28, 34, 35, 36],
        [14, 15, 16, 17, 37, 38],
        [18, 19, 20, 21, 22, 23, 24, 39, 40, 41, 42, 49, 51],
    ]
    assert mol_f_idx == expected_output, (
        "The function fragment_on_bonds does not return the expected output."
    )
