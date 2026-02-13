import os

from ..datasets.traj_handler import TrajectoryHandler


def test_traj_handler():
    """Test the TrajHandler class"""
    # Set the path to the ligand file
    current_dir = os.getcwd()
    top_filepath = f"{current_dir}/tests/test_data/PPS_OMB_0.pdb"
    traj_filepath = f"{current_dir}/tests/test_data/PPS_OMB_MD1_aligned_test.xtc"

    complex_traj_handler = TrajectoryHandler(
        top_path=top_filepath,
        trajectory_path=traj_filepath,
        ligand_name="2OW",
        radius_of_interest=16,
        spacing=0.5,
        distance_cutoff=5,
        warning_check=False,
    )

    complex_traj_handler.get_pocket_center()

    assert (
        complex_traj_handler.residues_at_pocket_str
        == "120 146 147 160 163 164 167 168 170 492 497 666 667 710 711 712 713 "
        "721 722 762 765 770 771 774"
    )
    assert complex_traj_handler.pocket_center == [
        48.18664509001232,
        107.5514571598598,
        64.30374107360839,
    ]

    complex_traj_handler.get_complex()

    assert hasattr(complex_traj_handler, "complex")
    assert hasattr(complex_traj_handler, "ligand")
    assert hasattr(complex_traj_handler, "protein")
