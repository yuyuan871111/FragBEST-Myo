import os

# import dataset processing code
from ..ppseg.workflow import voxelization_workflow


def test_voxelization_workflow():
    # read test BioLiP dataset
    current_dir = os.getcwd()
    dataset_dir = f"{current_dir}/tests/test_data"

    # extract the first row of index_df as an example
    query = "8pnw"
    pdbfilename = "8pnwA"
    pdb_file_path = f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}.pdb"
    BS_string = "58 155 159 188 190 191 192 212 215 216 252"

    # test the voxelization_workflow function
    voxelization_workflow(
        pdb_file_path, BS_string, prot_method="propka", radius_for_grid=5, spacing=1.0
    )

    # check if the output files are created
    voxel_file_path = (
        f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton_voxel_10.npy"
    )
    grid_file_path = (
        f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton_grids_10.pdb"
    )
    assert os.path.exists(voxel_file_path)
    assert os.path.exists(grid_file_path)

    # remove the output files
    test_files_path = f"{dataset_dir}/{query}/receptor_nr/{pdbfilename}_proton*"
    os.system(f"rm {test_files_path}")
